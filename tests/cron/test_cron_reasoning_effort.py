"""Per-job cron reasoning override contracts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    from cron import jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@contextmanager
def captured_cron_agents(home):
    """Run the real scheduler path while replacing external/runtime edges."""
    runtime = {
        "provider": "test-provider",
        "api_key": None,
        "base_url": "https://example.invalid/v1",
        "api_mode": "chat_completions",
    }
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", home), patch(
        "cron.scheduler._resolve_origin", return_value=None
    ), patch("hermes_cli.env_loader.load_hermes_dotenv"), patch(
        "hermes_cli.env_loader.reset_secret_source_cache"
    ), patch("hermes_state.SessionDB", return_value=fake_db), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime
    ), patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), patch(
        "run_agent.AIAgent"
    ) as agent_class:
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "synthetic result"}
        agent_class.return_value = agent
        yield agent_class


def _agent_job(job_id: str, **overrides):
    return {
        "id": job_id,
        "name": f"Synthetic {job_id}",
        "prompt": "Produce a synthetic result.",
        "model": "test-model",
        "provider": "test-provider",
        "deliver": "local",
        **overrides,
    }


def test_create_update_clear_and_persistence_reload(isolated_store):
    from cron.jobs import create_job, get_job, load_jobs, update_job

    job = create_job(
        prompt="Produce a synthetic result.",
        schedule="every 1h",
        name="Synthetic reasoning job",
        model="test-model",
        provider="test-provider",
        reasoning_effort="HIGH",
    )
    assert job["reasoning_effort"] == "high"
    assert get_job(job["id"])["reasoning_effort"] == "high"
    assert load_jobs()[0]["reasoning_effort"] == "high"

    renamed = update_job(job["id"], {"name": "Synthetic renamed job"})
    assert renamed["reasoning_effort"] == "high"

    disabled = update_job(job["id"], {"reasoning_effort": "none"})
    assert disabled["reasoning_effort"] is False
    assert get_job(job["id"])["reasoning_effort"] is False

    cleared = update_job(job["id"], {"reasoning_effort": None})
    assert cleared["reasoning_effort"] is None
    assert get_job(job["id"])["reasoning_effort"] is None


def test_old_record_without_reasoning_loads_without_rewrite(isolated_store):
    from cron.jobs import get_job

    jobs_file = isolated_store / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    original = {
        "jobs": [
            {
                "id": "legacy-job",
                "name": "Legacy synthetic job",
                "prompt": "Produce a synthetic result.",
                "enabled": True,
                "schedule_display": "every 1h",
            }
        ],
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    jobs_file.write_text(json.dumps(original, indent=2), encoding="utf-8")

    loaded = get_job("legacy-job")
    assert "reasoning_effort" not in loaded
    assert json.loads(jobs_file.read_text(encoding="utf-8")) == original


@pytest.mark.parametrize("value", ["turbo", "", True, 42])
def test_invalid_values_fail_with_shared_vocabulary(isolated_store, value):
    from cron.jobs import create_job

    with pytest.raises(ValueError) as exc_info:
        create_job(
            prompt="Produce a synthetic result.",
            schedule="every 1h",
            reasoning_effort=value,
        )
    message = str(exc_info.value)
    assert "Invalid cron reasoning_effort" in message
    assert "none, minimal, low, medium, high, xhigh, max, ultra" in message


def test_scheduled_execution_uses_job_override(isolated_store):
    from cron.jobs import create_job
    from cron.scheduler import run_one_job

    (isolated_store / "config.yaml").write_text(
        "model: test-model\nagent:\n  reasoning_effort: low\n",
        encoding="utf-8",
    )
    job = create_job(
        prompt="Produce a synthetic result.",
        schedule="every 1h",
        name="Synthetic scheduled job",
        model="test-model",
        provider="test-provider",
        reasoning_effort="xhigh",
        deliver="local",
    )

    with captured_cron_agents(isolated_store) as agent_class:
        assert run_one_job(job) is True

    assert agent_class.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_manual_run_uses_same_job_override(isolated_store):
    from cron.jobs import create_job
    from tools.cronjob_tools import cronjob

    (isolated_store / "config.yaml").write_text(
        "model: test-model\nagent:\n  reasoning_effort: low\n",
        encoding="utf-8",
    )
    job = create_job(
        prompt="Produce a synthetic result.",
        schedule="every 1h",
        name="Synthetic manual job",
        model="test-model",
        provider="test-provider",
        reasoning_effort="high",
        deliver="local",
    )

    with captured_cron_agents(isolated_store) as agent_class:
        result = json.loads(cronjob(action="run", job_id=job["id"]))

    assert result["success"] is True
    assert result["job"]["executed"] is True
    assert result["job"]["reasoning_effort"] == "high"
    assert agent_class.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }


def test_job_override_precedes_per_model_and_stays_isolated(isolated_store):
    from cron.scheduler import run_job

    (isolated_store / "config.yaml").write_text(
        "model: test-model\n"
        "agent:\n"
        "  reasoning_effort: medium\n"
        "  reasoning_overrides:\n"
        "    test-model: high\n",
        encoding="utf-8",
    )
    jobs = [
        _agent_job("inherit"),
        _agent_job("disabled", reasoning_effort=False),
        _agent_job("explicit", reasoning_effort="ultra"),
        _agent_job("inherit-again", reasoning_effort=None),
        # A hand-edited legacy record may contain an invalid/unhashable value.
        # It must inherit rather than crashing the scheduler.
        _agent_job("invalid-legacy", reasoning_effort=[]),
    ]

    with captured_cron_agents(isolated_store) as agent_class:
        for job in jobs:
            assert run_job(job)[0] is True

    configs = [call.kwargs["reasoning_config"] for call in agent_class.call_args_list]
    assert configs == [
        {"enabled": True, "effort": "high"},
        {"enabled": False},
        {"enabled": True, "effort": "ultra"},
        {"enabled": True, "effort": "high"},
        {"enabled": True, "effort": "high"},
    ]


def test_no_agent_job_preserves_override_without_constructing_llm(isolated_store):
    from cron.jobs import create_job, get_job
    from cron.scheduler import run_job

    job = create_job(
        prompt="",
        schedule="every 1h",
        name="Synthetic script-only job",
        script="synthetic-check.py",
        no_agent=True,
        reasoning_effort="ultra",
        deliver="local",
    )
    assert get_job(job["id"])["reasoning_effort"] == "ultra"

    with patch(
        "cron.scheduler._run_job_script_with_claim_heartbeat",
        return_value=(True, "synthetic output"),
    ), patch("run_agent.AIAgent") as agent_class:
        success, _doc, final_response, error = run_job(job)

    assert success is True
    assert final_response == "synthetic output"
    assert error is None
    agent_class.assert_not_called()
