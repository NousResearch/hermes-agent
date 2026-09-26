"""Tests for cron iteration limit behavior (#921).

Acceptance criteria:
1. An iteration-limited cron run is distinguishable in the durable executions ledger (executions.db)
   from a run that finished its work (status='failed' with explicit error instead of status='completed', error=null).
2. The warning logged names the job, the execution id, and the limit that was hit.
3. A cron job's iteration budget (max_turns / max_iterations) is settable per job and overrides agent.max_turns.
4. Truncated output is saved and delivered, but not marked as an unqualified success.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from cron import executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig


def _mock_runtime():
    return {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }


def test_run_job_iteration_limit_returns_failure_and_partial_response(tmp_path, monkeypatch, caplog):
    """An iteration-limited run returns success=False with partial response and logs details."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    job = {
        "id": "job-iter-test",
        "name": "Heavy report",
        "prompt": "generate report",
        "max_turns": 5,
    }
    fake_db = MagicMock()

    handoff_result = {
        "final_response": "Here is what I completed before hitting the iteration limit.",
        "completed": False,
        "failed": False,
        "interrupted": False,
        "turn_exit_reason": "max_iterations_reached(5/5)",
    }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_mock_runtime()), \
         patch("run_agent.AIAgent") as mock_agent_cls:

        mock_agent = MagicMock()
        mock_agent.max_iterations = 5
        mock_agent.run_conversation.return_value = handoff_result
        mock_agent_cls.return_value = mock_agent

        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            success, output, final_response, error = scheduler.run_job(job, execution_id="exec-iter-42")

    # 1. Distinguishable from success: success is False, error is explicit
    assert success is False
    assert error == "Job reached iteration limit (5)"

    # 2. Truncated output is saved and returned
    assert final_response == "Here is what I completed before hitting the iteration limit."
    assert "## Response\n\nHere is what I completed before hitting the iteration limit." in output
    assert "**Execution status:** Failed (Job reached iteration limit (5))" in output

    # 3. Warning names the job, execution id, and the limit
    warning_records = [r for r in caplog.records if "reached the iteration limit" in r.message]
    assert len(warning_records) >= 1
    msg = warning_records[0].message
    assert "Heavy report" in msg
    assert "exec-iter-42" in msg
    assert "5" in msg


def test_run_one_job_iteration_limit_marks_failed_in_ledger_and_delivers_partial(tmp_path, monkeypatch):
    """run_one_job marks status='failed' in executions.db while delivering the partial fallback response."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")

    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)

    sent = []

    async def mock_send(platform, pconfig, chat_id, text, **kwargs):
        sent.append(text)
        return {"success": True, "message_id": "receipt-iter-1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", mock_send)

    partial_text = "Progress: step 1 and 2 finished, step 3 hit iteration cap."
    handoff_output = (
        "# Heavy task\n\n## Response\n\n"
        + partial_text
        + "\n\n**Execution status:** Failed (Job reached iteration limit (10))\n"
    )

    def mock_run_job(job, **kwargs):
        return (False, handoff_output, partial_text, "Job reached iteration limit (10)")

    monkeypatch.setattr(scheduler, "run_job", mock_run_job)

    with jobs.use_cron_store(tmp_path):
        created_job = jobs.create_job(
            prompt="run long task", schedule="every 1h", deliver="telegram:chat123", max_turns=10,
        )

        execution = executions.create_execution(created_job["id"], source="builtin")
        created_job["execution_id"] = execution["id"]

        # Run the job
        completed = scheduler.run_one_job(created_job)
        assert completed is True

        # Check delivered message contains the partial response
        assert len(sent) == 1
        assert partial_text in sent[0]

        # Check job store status: error recorded
        updated_job = jobs.get_job(created_job["id"])
        assert updated_job["last_status"] == "error"
        assert updated_job["last_error"] == "Job reached iteration limit (10)"

        # Check executions ledger: must be status='failed', NOT 'completed'
        exec_record = executions.get_execution(execution["id"])
        assert exec_record is not None
        assert exec_record["status"] == "failed"
        assert exec_record["error"] == "Job reached iteration limit (10)"
        assert exec_record["delivery_outcome"] == "delivered"

        from cron.incidents import get_incident, list_incidents

        incident = list_incidents()[0]
        assert incident["job_id"] == created_job["id"]
        assert incident["state"] == "alerted"
        assert scheduler._repeat_alert_withheld(incident) is True

        # A second capped run has fresh progress despite the same incident signature.
        partial_text = "Progress: step 3 finished, step 4 hit iteration cap."
        handoff_output = "# Heavy task\n\n## Response\n\n" + partial_text
        second_execution = executions.create_execution(created_job["id"], source="builtin")
        updated_job["execution_id"] = second_execution["id"]
        assert scheduler.run_one_job(updated_job) is True
        assert len(sent) == 2
        assert partial_text in sent[1]
        assert sent[0] != sent[1]
        second_record = executions.get_execution(second_execution["id"])
        assert second_record["status"] == "failed"
        assert second_record["error"] == "Job reached iteration limit (10)"
        assert second_record["delivery_outcome"] == "delivered"
        assert len(list_incidents()) == 1
        assert get_incident(incident["id"])["state"] == "alerted"


def test_cron_job_max_turns_config_and_overrides(tmp_path):
    """Per-job max_turns is persisted, updated, validated, and overrides agent.max_turns."""
    with jobs.use_cron_store(tmp_path):
        # 1. create_job with max_turns
        job1 = jobs.create_job(prompt="p1", schedule="every 1h", max_turns=15)
        assert job1.get("max_turns") == 15
        assert jobs.get_job(job1["id"]).get("max_turns") == 15

        # 2. create_job with max_iterations alias
        job2 = jobs.create_job(prompt="p2", schedule="every 1h", max_iterations=20)
        assert job2.get("max_turns") == 20
        assert jobs.get_job(job2["id"]).get("max_turns") == 20

        # 3. update_job updates max_turns
        updated = jobs.update_job(job1["id"], {"max_turns": 25})
        assert updated.get("max_turns") == 25
        assert jobs.get_job(job1["id"]).get("max_turns") == 25

        # 4. update_job with max_iterations alias
        updated2 = jobs.update_job(job1["id"], {"max_iterations": 30})
        assert updated2.get("max_turns") == 30
        assert jobs.get_job(job1["id"]).get("max_turns") == 30

        # 5. update_job clearing max_turns
        updated_cleared = jobs.update_job(job1["id"], {"max_turns": ""})
        assert updated_cleared.get("max_turns") is None
        assert jobs.get_job(job1["id"]).get("max_turns") is None

        # 6. Invalid values raise ValueError
        with pytest.raises(ValueError, match="positive integer"):
            jobs.create_job(prompt="p3", schedule="every 1h", max_turns=0)

        with pytest.raises(ValueError, match="positive integer"):
            jobs.create_job(prompt="p4", schedule="every 1h", max_turns=-5)

        with pytest.raises(ValueError, match="positive integer"):
            jobs.create_job(prompt="p5", schedule="every 1h", max_turns="invalid")

        with pytest.raises(ValueError, match="positive integer"):
            jobs.create_job(prompt="p6", schedule="every 1h", max_turns=True)

        with pytest.raises(ValueError, match="positive integer"):
            jobs.update_job(job2["id"], {"max_turns": -1})


def test_resolve_cron_agent_setup_precedence(tmp_path):
    """_resolve_cron_agent_setup prioritizes job max_turns over agent.max_turns."""
    jc = MagicMock()
    jc.model = "test-model"
    jc.cfg = {"agent": {"max_turns": 100}, "max_turns": 50}

    # Job with max_turns overrides global agent.max_turns
    job_with_cap = {"id": "j1", "name": "j1", "max_turns": 12}
    with patch("cron.scheduler._load_prefill_messages", return_value=None), \
         patch("cron.scheduler._guard_job_credential_exfil"), \
         patch("cron.scheduler._preflight_or_block", return_value=None), \
         patch("cron.scheduler._job_fallback_chain", return_value=None), \
         patch("cron.scheduler._load_credential_pool", return_value=None), \
         patch("cron.scheduler._init_cron_mcp_tools"), \
         patch("cron.scheduler._resolve_job_runtime", return_value=({}, "test-model")), \
         patch("cron.scheduler._cron_preflight_enabled", return_value=False):
        setup = scheduler._resolve_cron_agent_setup(job_with_cap, "j1", "j1", jc)
        assert setup.max_iterations == 12

    # Job with max_iterations alias overrides global agent.max_turns
    job_with_alias = {"id": "j2", "name": "j2", "max_iterations": 8}
    with patch("cron.scheduler._load_prefill_messages", return_value=None), \
         patch("cron.scheduler._guard_job_credential_exfil"), \
         patch("cron.scheduler._preflight_or_block", return_value=None), \
         patch("cron.scheduler._job_fallback_chain", return_value=None), \
         patch("cron.scheduler._load_credential_pool", return_value=None), \
         patch("cron.scheduler._init_cron_mcp_tools"), \
         patch("cron.scheduler._resolve_job_runtime", return_value=({}, "test-model")), \
         patch("cron.scheduler._cron_preflight_enabled", return_value=False):
        setup = scheduler._resolve_cron_agent_setup(job_with_alias, "j2", "j2", jc)
        assert setup.max_iterations == 8

    # Job without override falls back to agent.max_turns (100)
    job_no_cap = {"id": "j3", "name": "j3"}
    with patch("cron.scheduler._load_prefill_messages", return_value=None), \
         patch("cron.scheduler._guard_job_credential_exfil"), \
         patch("cron.scheduler._preflight_or_block", return_value=None), \
         patch("cron.scheduler._job_fallback_chain", return_value=None), \
         patch("cron.scheduler._load_credential_pool", return_value=None), \
         patch("cron.scheduler._init_cron_mcp_tools"), \
         patch("cron.scheduler._resolve_job_runtime", return_value=({}, "test-model")), \
         patch("cron.scheduler._cron_preflight_enabled", return_value=False):
        setup = scheduler._resolve_cron_agent_setup(job_no_cap, "j3", "j3", jc)
        assert setup.max_iterations == 100


def test_cron_cli_args_parsing():
    """CLI subcommands accept --max-turns and --max-iterations."""
    import argparse
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    build_cron_parser(subparsers, cmd_cron=lambda args: None)

    args_create = parser.parse_args(["cron", "create", "every 1h", "prompt", "--max-turns", "15"])
    assert args_create.max_turns == "15"

    args_create_alias = parser.parse_args(["cron", "create", "every 1h", "prompt", "--max-iterations", "25"])
    assert args_create_alias.max_turns == "25"

    args_edit = parser.parse_args(["cron", "edit", "job-123", "--max-turns", "30"])
    assert args_edit.max_turns == "30"

    args_edit_alias = parser.parse_args(["cron", "edit", "job-123", "--max-iterations", "40"])
    assert args_edit_alias.max_turns == "40"
