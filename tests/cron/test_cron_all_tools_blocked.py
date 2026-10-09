"""run_job's runtime dead-turn signal (#135544).

``run_conversation`` exports ``tool_calls_attempted``/``tool_calls_blocked`` on the
turn result (pinned in tests/agent/test_turn_tool_call_stats.py). ``run_job`` turns
the all-blocked shape into a job flag — ``job["_all_tool_calls_blocked"]`` — that the
fire path consumes for failure bookkeeping, mirroring ``_model_unreachable``.

Consumption contract (fire path records the run as failed) is pinned in
tests/cron/test_run_one_job.py; this module pins WHEN the flag is set:

  * attempted == blocked  -> flag set to the attempted count
  * some call ran         -> no flag (partial blocks are the model's to handle)
  * text-only turn        -> no flag (no tool calls, nothing blocked)
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job


@contextlib.contextmanager
def _run_job_patches(tmp_path, run_conversation_result):
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = run_conversation_result
    base = [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler_delivery._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state_registry.acquire", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-" + "key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent", return_value=mock_agent),
    ]
    with contextlib.ExitStack() as stack:
        for cm in base:
            stack.enter_context(cm)
        yield mock_agent


def _job():
    return {"id": "all-blocked", "name": "t", "prompt": "hi"}


def test_run_job_flags_a_turn_whose_every_call_was_blocked(tmp_path):
    result = {
        "final_response": "report says no data could be collected",
        "tool_calls_attempted": 6,
        "tool_calls_blocked": 6,
    }
    job = _job()
    with _run_job_patches(tmp_path, result):
        success, _output, _final, error = run_job(job)

    # run_job's own scope: the turn completed; the FIRE path is what fails it.
    assert success is True
    assert error is None
    assert job["_all_tool_calls_blocked"] == 6


def test_run_job_does_not_flag_a_turn_where_some_call_ran(tmp_path):
    result = {
        "final_response": "partial report",
        "tool_calls_attempted": 6,
        "tool_calls_blocked": 3,
    }
    job = _job()
    with _run_job_patches(tmp_path, result):
        run_job(job)

    assert "_all_tool_calls_blocked" not in job


def test_run_job_does_not_flag_a_text_only_turn(tmp_path):
    result = {"final_response": "plain answer"}
    job = _job()
    with _run_job_patches(tmp_path, result):
        run_job(job)

    assert "_all_tool_calls_blocked" not in job
