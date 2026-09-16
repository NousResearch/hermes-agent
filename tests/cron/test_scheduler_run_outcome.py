"""Cron run success follows nested child outcomes, not parent-turn completion.

Regression for #112426: a parent cron agent can finish its own turn after
delegate_task returns status=failed. The scheduler used to record that fire as
ok because only parent exceptions/timeouts flipped success.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from cron.scheduler_prompt import _build_job_prompt
from cron.scheduler_run_outcome import nested_child_failure


def _delegate_messages(*entries: dict, tool_name: str = "delegate_task") -> list[dict]:
    call_id = "call_delegate_1"
    payload = {"results": list(entries), "total_duration_seconds": 1.2}
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def test_failed_delegate_child_is_cron_failure_evidence():
    result = {
        "final_response": "The delegated child could not finish the report.",
        "completed": True,
        "failed": False,
        "messages": _delegate_messages(
            {"status": "failed", "error": "child could not finish the report"}
        ),
    }

    error = nested_child_failure(result)

    assert error is not None
    assert "child could not finish the report" in error


def test_completed_delegate_child_is_not_failure_evidence():
    result = {
        "final_response": "Report ready.",
        "completed": True,
        "messages": _delegate_messages({"status": "completed", "summary": "done"}),
    }

    assert nested_child_failure(result) is None


def test_prose_that_mentions_failure_without_child_result_is_not_evidence():
    result = {
        "final_response": 'Docs mention {"status": "failed"} but this run recovered.',
        "completed": True,
        "messages": [],
    }

    assert nested_child_failure(result) is None


def test_terminal_nonzero_exit_is_not_treated_as_delegated_child_failure():
    result = {
        "final_response": "grep found no matches",
        "completed": True,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "term1",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "term1",
                "content": json.dumps({"exit_code": 1, "output": "not found"}),
            },
        ],
    }

    assert nested_child_failure(result) is None


def test_mixed_batch_fails_closed_on_any_failed_child():
    result = {
        "final_response": "one child recovered, one did not",
        "completed": True,
        "messages": _delegate_messages(
            {"status": "completed", "summary": "ok"},
            {"status": "failed", "error": "second child timed out"},
        ),
    }

    error = nested_child_failure(result)

    assert error is not None
    assert "second child timed out" in error


def test_content_block_tool_payload_is_still_parsed():
    call_id = "call_blocks"
    payload = {"results": [{"status": "failed", "error": "blocked child"}]}
    result = {
        "final_response": "child failed",
        "completed": True,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": call_id, "function": {"name": "delegate_task"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [{"type": "text", "text": json.dumps(payload)}],
            },
        ],
    }

    error = nested_child_failure(result)

    assert error is not None
    assert "blocked child" in error


def test_background_dispatch_handle_is_not_failure_evidence():
    result = {
        "final_response": "spawned a child",
        "completed": True,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "bg1",
                        "function": {"name": "delegate_task", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bg1",
                "content": json.dumps(
                    {
                        "status": "dispatched",
                        "delegation_id": "deleg_test",
                        "note": "running in the background",
                    }
                ),
            },
        ],
    }

    assert nested_child_failure(result) is None


def test_run_job_records_failed_child_as_failed_cron_run(tmp_path):
    """Parent turn completion must not mark the fire ok when a child failed."""
    from cron.scheduler import run_job

    job = {"id": "nested-fail", "name": "delegate", "prompt": "run the audit"}
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    child_error = "delegated child could not finish the report"
    conversation = {
        "final_response": "The delegated child failed.",
        "completed": True,
        "failed": False,
        "messages": _delegate_messages({"status": "failed", "error": child_error}),
    }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = conversation
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(job)

    assert success is False
    assert error is not None and child_error in error
    assert "The delegated child failed." in final_response
    assert "The delegated child failed." in output


def test_run_job_keeps_healthy_parent_success_without_failed_children(tmp_path):
    from cron.scheduler import run_job

    job = {"id": "nested-ok", "name": "delegate", "prompt": "run the audit"}
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    conversation = {
        "final_response": "Report ready.",
        "completed": True,
        "failed": False,
        "messages": _delegate_messages({"status": "completed", "summary": "done"}),
    }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = conversation
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert final_response == "Report ready."
    assert "Report ready." in output


def test_cron_prompt_tells_agent_failed_children_fail_the_run():
    prompt = _build_job_prompt({"prompt": "Check delegated work"})

    assert "[SILENT]" in prompt
    assert "status=failed" in prompt
    assert "recorded as failed" in prompt
    assert "[CRON_FAILURE]" not in prompt
    assert "[FAILURE:" not in prompt

