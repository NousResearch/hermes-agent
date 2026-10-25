"""Lifecycle-hook isolation for persistence-disabled forks, at the API-request granularity.

Mirrors test_detached_fork_lifecycle_hooks.py's session/turn-level guard (#107069), extended to
the three API-request hooks (pre_api_request, post_api_request, api_request_error) that PR left
unguarded — a detached fork (background_review) shares its parent's real session_id, so without
this guard a plugin like plugins/observability/langfuse attributes per-request trace events to
the live session even though nothing was persisted.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.api_request_hooks import ApiRequestHooksMixin
from agent.turn_api_request import _fire_pre_api_request_hook
from agent.turn_response_intake import _fire_post_api_request_hook


def _agent(*, persist_disabled: bool):
    return SimpleNamespace(
        _persist_disabled=persist_disabled,
        session_id="shared-session",
        platform="cli",
        model="test/model",
        provider="test-provider",
        base_url="https://api.test",
        api_mode="chat",
        max_tokens=1024,
        tools=[],
        _last_api_first_chunk_at=None,
        _api_request_payload_for_hook=Mock(return_value={"messages": []}),
        _api_response_payload_for_hook=Mock(return_value={"content": "hi"}),
        _usage_summary_for_api_request_hook=Mock(return_value={"total_tokens": 10}),
    )


def _assistant_message():
    return SimpleNamespace(role="assistant", content="hi", tool_calls=None)


def _fire_pre(agent):
    _fire_pre_api_request_hook(
        agent, {"messages": []}, [], [], messages=[], original_user_message="hi",
        approx_tokens=10, total_chars=10, retry_count=0, api_call_count=1,
        api_request_id="req-1", api_start_time=0.0, effective_task_id="task-1", turn_id="turn-1",
    )


def _fire_post(agent):
    _fire_post_api_request_hook(
        agent, response=SimpleNamespace(model="test/model"), assistant_message=_assistant_message(),
        finish_reason="stop", api_messages=[], api_call_count=1, api_duration=0.5,
        api_start_time=0.0, api_request_id="req-1", effective_task_id="task-1", turn_id="turn-1",
    )


def _fire_error(agent):
    ApiRequestHooksMixin._invoke_api_request_error_hook(
        agent, task_id="task-1", turn_id="turn-1", api_request_id="req-1", api_call_count=1,
        api_start_time=0.0, api_kwargs={}, error_type="RateLimitError", error_message="boom",
    )


def test_persist_disabled_fork_skips_all_three_api_request_hooks():
    agent = _agent(persist_disabled=True)
    with (
        patch("hermes_cli.lifecycle.invoke_hook") as lifecycle_hook,
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
    ):
        _fire_pre(agent)
        _fire_post(agent)
        _fire_error(agent)

    lifecycle_hook.assert_not_called()


def test_persisted_agent_still_fires_all_three_api_request_hooks():
    agent = _agent(persist_disabled=False)
    with (
        patch("hermes_cli.lifecycle.invoke_hook") as lifecycle_hook,
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch("agent.conversation_loop._system_prompt_for_hooks", return_value="sys"),
        patch("agent.conversation_loop._moa_reference_metrics_for_hook", return_value=None),
    ):
        _fire_pre(agent)
        _fire_post(agent)
        _fire_error(agent)

    fired = [call.args[0] for call in lifecycle_hook.call_args_list]
    assert fired == ["pre_api_request", "post_api_request", "api_request_error"]
