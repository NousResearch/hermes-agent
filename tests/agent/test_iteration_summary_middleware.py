"""Iteration summaries obey execution middleware on every provider path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import chat_completion_helpers as helpers
from hermes_cli import plugins


@pytest.fixture
def summary_agent(monkeypatch):
    request = {"model": "review-model", "messages": [{"role": "user", "content": "private draft"}]}
    transport = SimpleNamespace(
        build_kwargs=lambda **kw: dict(request),
        normalize_response=lambda response, **kw: SimpleNamespace(content=response, tool_calls=[]),
    )
    agent = SimpleNamespace(
        provider="custom", model="review-model", base_url="https://example.invalid/v1",
        session_id="summary-session", platform="cli", max_tokens=100,
        _current_task_id="summary-task", _current_turn_id="summary-turn",
        reasoning_config=None, _is_anthropic_oauth=False,
        _anthropic_preserve_dots=lambda: False,
        _get_transport=lambda: transport, _build_api_kwargs=lambda messages: dict(request),
        _interruptible_api_call=Mock(return_value="summary"),
    )
    monkeypatch.setattr(helpers, "sanitize_outbound_kwargs", lambda *args: None)
    monkeypatch.setattr(helpers, "is_router_timeout_shim", lambda response: False)
    return agent


@pytest.mark.parametrize("mode", ["codex_responses", "anthropic_messages", "chat_completions"])
@pytest.mark.parametrize("has_turn", [True, False])
def test_summary_middleware_can_refuse_without_provider_call(monkeypatch, summary_agent, mode, has_turn):
    agent = summary_agent
    agent.api_mode = mode
    seen = []
    if not has_turn:
        del agent._current_task_id, agent._current_turn_id

    def refuse(*, task_id, turn_id, api_call_count, middleware_trace, **context):
        assert task_id == ("summary-task" if has_turn else "")
        assert turn_id == ("summary-turn" if has_turn else "")
        assert api_call_count == 7 and middleware_trace == []
        seen.append(context)
        return "policy refused summary"

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: SimpleNamespace(
        _middleware={"llm_execution": [refuse]}, _report_hook_failure=lambda *args, **kw: None))
    builder = helpers._SUMMARY_ATTEMPT_BUILDERS.get(mode, helpers._chat_summary_attempt)
    attempt = builder(agent, [], "iteration-summary:test", api_call_count=7)
    for retry in (0, 1):
        assert attempt(retry) == "policy refused summary"
    agent._interruptible_api_call.assert_not_called()
    assert [context["retry_count"] for context in seen] == [0, 1]
    assert all(context["provider"] == agent.provider and context["base_url"] == agent.base_url
               and context["api_mode"] == mode and context["session_id"] == agent.session_id
               for context in seen)


@pytest.mark.parametrize("mode", ["codex_responses", "anthropic_messages", "chat_completions"])
def test_summary_middleware_forwards_changed_request_once(monkeypatch, summary_agent, mode):
    from agent import relay_llm

    agent = summary_agent
    agent.api_mode = mode
    relay_calls = []

    def relay(request, callback, **kwargs):
        relay_calls.append(kwargs)
        return callback(request)

    def replace(**context):
        return context["next_call"]({**context["request"], "model": "approved-model"})

    monkeypatch.setattr(relay_llm, "execute_current", relay)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: SimpleNamespace(
        _middleware={"llm_execution": [replace]}))
    builder = helpers._SUMMARY_ATTEMPT_BUILDERS.get(mode, helpers._chat_summary_attempt)
    assert builder(agent, [], "iteration-summary:test")(0) == "summary"
    agent._interruptible_api_call.assert_called_once()
    assert agent._interruptible_api_call.call_args.args[0]["model"] == "approved-model"
    assert len(relay_calls) == (0 if mode == "codex_responses" else 1)
