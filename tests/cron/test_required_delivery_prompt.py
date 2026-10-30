"""The same job contract controls prompt guidance and final-result acceptance."""

import pytest

from cron.scheduler import _forbidden_final_response
from cron.scheduler_prompt import _build_job_prompt


def test_job_forbidden_silence_removes_suppression_guidance():
    watcher = {"prompt": "Check for updates"}
    required = {**watcher, "forbidden_final_responses": ["[SILENT]"]}
    watcher_prompt = _build_job_prompt(watcher)
    required_prompt = _build_job_prompt(required)
    assert "[SILENT]" in watcher_prompt
    assert "[SILENT]" not in required_prompt
    assert watcher["prompt"] in required_prompt
    # A different forbidden token does not change ordinary watcher behavior.
    assert _build_job_prompt({**watcher, "forbidden_final_responses": ["DEBUG"]}) == watcher_prompt


@pytest.mark.parametrize("response", ["[SILENT]", "silent", "NO_REPLY", "NO REPLY", "[SILENT]\nNothing new"])
def test_forbidden_silence_uses_the_canonical_protocol_matcher(response):
    job = {"forbidden_final_responses": ["[SILENT]"]}
    assert _forbidden_final_response(job, response) == "[SILENT]"
    assert _forbidden_final_response({}, response) is None
    assert _forbidden_final_response(job, "The log contains [SILENT], but the task is ready.") is None


@pytest.mark.parametrize("no_agent", [False, True])
def test_successful_script_gate_is_not_a_forbidden_agent_reply(tmp_path, monkeypatch, no_agent, cron_owner):
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch
    from cron import scheduler as sched

    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "gate.py"
    script.write_text('print(\'{"wakeAgent": false}\')\n')
    job = {
        "id": "sleep-control", "name": "sleep-control", "prompt": "Prepare a result",
        "script": str(script), "no_agent": no_agent, "deliver": "local",
        "forbidden_final_responses": ["[SILENT]"],
    }
    monkeypatch.setattr(sched, "_launch_external_cron_worker", lambda _job: False)
    with patch.object(sched, "_construct_cron_agent") as agent:
        success, output, final, error = sched.run_job(job)
    assert success and error is None
    agent.assert_not_called()
    fence = MagicMock()
    fence.side_effect_fence.side_effect = lambda: nullcontext(True)
    fence.lost.return_value = False
    delivery = sched._RunDelivery(job=job, success=success, error=error)
    with patch.object(sched, "save_job_output", return_value=str(tmp_path / "output.md")), \
         patch.object(sched, "_is_interrupted", return_value=False), \
         patch.object(sched, "_deliver_result") as send:
        sched._save_compose_deliver(delivery, fence, final, output,
                                   adapters=None, loop=None, verbose=False, execution_token=None)
    assert delivery.success, delivery.error
    assert not delivery.should_deliver
    assert delivery.terminal_status is None
    send.assert_not_called()

    # Exercise the real execution owner as well: the independent watchdog reads
    # this durable outcome, not the local delivery object's success bit.
    from cron import executions
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    execution = executions.create_execution(job["id"], source="builtin")
    with patch.object(sched, "claim_dispatch", return_value=True), \
         patch.object(sched, "mark_job_run", return_value=True), \
         patch.object(sched, "save_job_output", return_value=str(tmp_path / "output.md")), \
         patch.object(sched, "_deliver_result") as send:
        assert sched.run_one_job({**job, "execution_id": execution["id"]})
    persisted = executions.get_execution(execution["id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["delivery_outcome"] == "suppressed"
    assert persisted["error"] is None
    send.assert_not_called()


def test_agent_silence_cannot_spoof_script_gate_provenance(tmp_path):
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch
    from cron import scheduler as sched

    job = {"id": "awake-control", "deliver": "local",
           "forbidden_final_responses": ["[SILENT]"]}
    fence = MagicMock()
    fence.side_effect_fence.side_effect = lambda: nullcontext(True)
    fence.lost.return_value = False
    delivery = sched._RunDelivery(job=job, success=True, error=None)
    with patch.object(sched, "save_job_output", return_value=str(tmp_path / "output.md")), \
         patch.object(sched, "_is_interrupted", return_value=False), \
         patch.object(sched, "_deliver_result") as send:
        sched._save_compose_deliver(
            delivery, fence, "[SILENT]", "Script gate returned `wakeAgent=false` — agent skipped.",
            adapters=None, loop=None, verbose=False, execution_token=None)
    assert not delivery.success
    assert delivery.terminal_status == "forbidden_final_response"
    send.assert_not_called()
