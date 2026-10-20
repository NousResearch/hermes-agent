"""Iteration summaries obey execution middleware on every provider path."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import chat_completion_helpers as helpers
from hermes_cli import plugins


@pytest.fixture
def summary_agent(monkeypatch):
    request = {"model": "review-model", "messages": [{"role": "user", "content": "private draft"}]}
    transport = SimpleNamespace(
        build_kwargs=lambda **kw: deepcopy(request),
        normalize_response=lambda response, **kw: SimpleNamespace(content=response, tool_calls=[]),
    )
    agent = SimpleNamespace(
        provider="custom", model="review-model", base_url="https://example.invalid/v1",
        session_id="summary-session", platform="cli", max_tokens=100,
        _current_task_id="summary-task", _current_turn_id="summary-turn",
        reasoning_config=None, _is_anthropic_oauth=False,
        _anthropic_preserve_dots=lambda: False,
        _get_transport=lambda: transport, _build_api_kwargs=lambda messages: deepcopy(request),
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

    # Exactly the ordinary execution callback fields, with no **kwargs escape.
    def refuse(*, request, original_request, next_call, telemetry_schema_version,
               middleware_schema_version, task_id, turn_id, api_request_id,
               session_id, platform, model, provider, base_url, api_mode,
               api_call_count, middleware_trace):
        assert task_id == ("summary-task" if has_turn else "")
        assert turn_id == ("summary-turn" if has_turn else "")
        assert api_call_count == 7 and middleware_trace == []
        seen.append(dict(provider=provider, base_url=base_url, api_mode=api_mode,
                         session_id=session_id, api_request_id=api_request_id))
        return "policy refused summary"

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: SimpleNamespace(
        _middleware={"llm_execution": [refuse]}, _report_hook_failure=lambda *args, **kw: None))
    builder = helpers._SUMMARY_ATTEMPT_BUILDERS.get(mode, helpers._chat_summary_attempt)
    attempt = builder(agent, [], "iteration-summary:test", api_call_count=7)
    for retry in (0, 1):
        assert attempt(retry) == "policy refused summary"
    agent._interruptible_api_call.assert_not_called()
    assert len(seen) == 2
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


@pytest.mark.parametrize("mode", ["codex_responses", "anthropic_messages", "chat_completions"])
def test_summary_mutation_preserves_original_and_retry_request(monkeypatch, summary_agent, mode):
    from agent import relay_llm

    agent = summary_agent
    agent.api_mode = mode
    seen = []
    originals = []

    def mutate(**context):
        request = context["request"]
        seen.append(deepcopy(request))
        originals.append(context["original_request"])
        request["model"] = "approved-model"
        request["messages"][0]["content"] = "approved draft"
        return context["next_call"](request)

    monkeypatch.setattr(relay_llm, "execute_current", lambda request, callback, **kw: callback(request))
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: SimpleNamespace(
        _middleware={"llm_execution": [mutate]}))
    builder = helpers._SUMMARY_ATTEMPT_BUILDERS.get(mode, helpers._chat_summary_attempt)
    attempt = builder(agent, [], "iteration-summary:test")
    assert attempt(0) == attempt(1) == "summary"
    assert len(seen) == 2
    assert seen[0] == seen[1] == originals[0] == originals[1]
    assert seen[0]["model"] == "review-model"
    assert seen[0]["messages"][0]["content"] == "private draft"
    assert agent._interruptible_api_call.call_count == 2
    for call in agent._interruptible_api_call.call_args_list:
        assert call.args[0]["model"] == "approved-model"
        assert call.args[0]["messages"][0]["content"] == "approved draft"
