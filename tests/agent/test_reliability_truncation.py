"""Exhausted incomplete action retries may advance, never execute or replay tools."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.chat_completion_helpers import _build_partial_stream_stub
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _Trunc, recover_from_truncation
from hermes_constants import PARTIAL_STREAM_STUB_ID


def state(stub=False, available=True, activates=True):
    messages = [
        {"role": "user", "content": "fixture request"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "already-done", "type": "function", "function": {"name": "fixture", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "already-done", "content": "completed once"},
    ]
    agent = SimpleNamespace(
        provider="fixture", model="fixture", log_prefix="", api_mode="chat_completions",
        _fallback_index=0, _fallback_chain=[{}] if available else [],
        _try_activate_fallback=Mock(return_value=activates),
        _buffer_vprint=Mock(), _vprint=Mock(), _flush_status_buffer=Mock(),
        _emit_diagnostic_status=Mock(), _cleanup_task_resources=Mock(), _persist_session=Mock(),
        _ephemeral_max_output_tokens=32768, _ephemeral_reasoning_off=True,
        _get_transport=ChatCompletionsTransport,
        _requested_output_cap_from_api_kwargs=Mock(return_value=None), max_tokens=4096,
        _build_assistant_message=lambda message, reason: {"role": "assistant", "content": message.content},
        _strip_think_blocks=lambda content: content,
    )
    response = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID if stub else "fixture-response",
        choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                id="incomplete", type="function",
                function=SimpleNamespace(name="fixture", arguments='{'),
            )]),
        )],
    )
    st = _Trunc(
        agent=agent, response=response,
        finish_reason="length", messages=messages, conversation_history=[],
        api_call_count=5, effective_task_id="fixture-task", current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[], truncated_tool_call_retries=4,
        retry_count=2, compression_attempts=1,
    )
    return st


def recover(st, retry):
    return recover_from_truncation(
        st.agent, st.response, st.finish_reason, retry,
        **{name: getattr(st, name) for name in (
            "messages", "conversation_history", "api_call_count", "effective_task_id",
            "current_turn_user_idx", "length_continue_retries", "truncated_response_parts",
            "truncated_tool_call_retries", "retry_count", "compression_attempts")},
        api_kwargs={},
    )


@pytest.mark.parametrize("stub", [False, True])
def test_exhausted_incomplete_action_advances_without_replaying_completed_tools(stub):
    st = state(stub=stub)
    original = deepcopy(st.messages)
    retry = TurnRetryState()
    verdict = recover(st, retry)
    assert verdict.action == "break"
    assert retry.restart_with_rebuilt_messages is True
    assert verdict.messages == original
    assert verdict.truncated_tool_call_retries == 0
    assert verdict.retry_count == 0
    assert st.agent._ephemeral_max_output_tokens is None
    assert st.agent._ephemeral_reasoning_off is False
    st.agent._try_activate_fallback.assert_called_once()
    st.agent._cleanup_task_resources.assert_not_called()
    st.agent._persist_session.assert_not_called()


@pytest.mark.parametrize("available,activates", [(False, False), (True, False)])
def test_exhausted_chain_keeps_fail_closed_terminal_result(available, activates):
    st = state(available=available, activates=activates)
    retry = TurnRetryState()
    verdict = recover(st, retry)
    assert verdict.action == "return"
    assert verdict.result is not None
    assert verdict.result["completed"] is False
    assert verdict.result["partial"] is True
    assert retry.restart_with_rebuilt_messages is False
    assert all(not message.get("tool_calls") or message["tool_calls"][0]["id"] == "already-done"
               for message in verdict.messages)
    st.agent._cleanup_task_resources.assert_called_once()
    st.agent._persist_session.assert_called_once()


@pytest.mark.parametrize("dropped", [False, True])
def test_real_fallback_chain_advances_once_per_route_and_then_stops(dropped, monkeypatch):
    from tests.agent.test_provider_fallback import _make_agent, _mock_client

    agent = _make_agent(fallback_model=[
        {"provider": "fixture-a", "model": "fixture-model-a"},
        {"provider": "fixture-b", "model": "fixture-model-b"},
    ])
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                        lambda *args, **kwargs: (_mock_client("https://fixture.invalid/v1"), kwargs["model"]))
    monkeypatch.setattr("agent.chat_completion_helpers._update_fallback_context_compressor", lambda agent: None)
    st = state()
    if dropped:
        st.response = _build_partial_stream_stub(
            "assistant", "Preparing the action.", "", "fixture", None, dropped_tool_names=["fixture"],
        )
    st.agent = agent
    original = deepcopy(st.messages)
    for expected_index in (1, 2):
        st.truncated_tool_call_retries = 4
        st.length_continue_retries = 3 if dropped else 0
        retry = TurnRetryState()
        st = recover(st, retry)
        assert st.action == "break"
        assert agent._fallback_index == expected_index
        assert st.messages == original
    st.truncated_tool_call_retries = 4
    st.length_continue_retries = 3 if dropped else 0
    terminal = recover(st, TurnRetryState())
    assert terminal.action == "return"
    assert terminal.result is not None
    assert terminal.result["completed"] is False
    assert agent._fallback_index == 2


@pytest.mark.parametrize("clean_eof", [False, True])
@pytest.mark.parametrize("content", [None, "Preparing the action."])
def test_real_dropped_tool_stub_uses_bounded_action_retries_then_fallback(clean_eof, content):
    st = state(stub=True)
    st.response = _build_partial_stream_stub(
        "assistant", content, "", "fixture", None,
        dropped_tool_names=["fixture"], clean_eof=clean_eof,
    )
    normalized = st.agent._get_transport().normalize_response(st.response)
    assert not normalized.tool_calls
    original = deepcopy(st.messages)
    retry = TurnRetryState()
    for count in range(4):
        st.length_continue_retries = count
        retry.restart_with_length_continuation = False
        verdict = recover(st, retry)
        assert verdict.messages[:len(original)] == original
        if count < 3:
            assert verdict.action == "break"
            assert retry.restart_with_length_continuation
            assert verdict.messages[-1]["_length_continuation_nudge"]
            assert "too large" in verdict.messages[-1]["content"]
            st.agent._try_activate_fallback.assert_not_called()
        else:
            assert verdict.action == "break"
            assert retry.restart_with_rebuilt_messages
            assert not retry.restart_with_length_continuation
            assert verdict.messages == original
            assert verdict.length_continue_retries == 0
            assert verdict.truncated_tool_call_retries == 0
            assert verdict.truncated_response_parts == []
            st.agent._try_activate_fallback.assert_called_once()
    st.agent._cleanup_task_resources.assert_not_called()
    st.agent._persist_session.assert_not_called()


@pytest.mark.parametrize("available,activates", [(False, False), (True, False)])
def test_real_dropped_tool_stub_exhaustion_is_incomplete_not_success(available, activates):
    st = state(stub=True, available=available, activates=activates)
    st.length_continue_retries = 3
    st.response = _build_partial_stream_stub(
        "assistant", None, "", "fixture", None, dropped_tool_names=["fixture"],
    )
    retry = TurnRetryState()
    verdict = recover(st, retry)
    assert verdict.action == "return"
    assert verdict.result is not None
    assert verdict.result["completed"] is False
    assert verdict.result["partial"] is True
    assert not retry.restart_with_rebuilt_messages
    assert not retry.restart_with_length_continuation
    assert all(not message.get("tool_calls") or message["tool_calls"][0]["id"] == "already-done"
               for message in verdict.messages)


@pytest.mark.parametrize("clean_eof", [False, True])
def test_real_text_only_stub_keeps_existing_continuation_behavior(clean_eof):
    st = state(stub=True)
    st.response = _build_partial_stream_stub(
        "assistant", "A partial explanation.", "", "fixture", None, clean_eof=clean_eof,
    )
    retry = TurnRetryState()
    verdict = recover(st, retry)
    assert verdict.action == "break"
    assert retry.restart_with_length_continuation
    assert not retry.restart_with_rebuilt_messages
    assert verdict.messages[-1]["_length_continuation_nudge"]
    st.agent._try_activate_fallback.assert_not_called()


def response_loop_state(st, prompt):
    """Exercise the phase verdict copier, restart consumer and request builder together."""
    return SimpleNamespace(
        response=st.response, _retry=TurnRetryState(), thinking_spinner=None,
        messages=st.messages, api_messages=[{"role": "system", "content": prompt}], api_kwargs={},
        active_system_prompt=prompt, conversation_history=[], finish_reason=None,
        retry_count=2, max_retries=8, compression_attempts=1, max_compression_attempts=3,
        length_continue_retries=3, truncated_response_parts=[], truncated_tool_call_retries=4,
        current_turn_user_idx=0, api_call_count=5, api_request_id=None, api_start_time=0,
        effective_task_id="fixture-task", turn_id=None, _preflight_compression_blocked=True,
        _last_preflight_pressure=None, interrupted=False, user_message="fixture request",
        final_response=None, restart_count=0, _turn_exit_reason=None,
    )


@pytest.mark.parametrize("dropped", [False, True])
@pytest.mark.parametrize("decorated", [False, True])
def test_incomplete_action_rebuild_propagates_route_identity_to_wire(dropped, decorated, monkeypatch):
    from agent.conversation_loop import _run_phase
    from agent.prompt_caching import apply_anthropic_cache_control
    from agent.turn_context import build_api_messages
    from agent.turn_iteration_prep import apply_retry_restarts
    from agent.turn_response_check import check_api_response
    from tests.agent.test_provider_fallback import _make_agent, _mock_client

    agent = _make_agent(fallback_model=[
        {"provider": "fixture-a", "model": "fixture-model-a"},
        {"provider": "fixture-b", "model": "fixture-model-b"},
    ])
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                        lambda *args, **kwargs: (_mock_client("https://fixture.invalid/v1"), kwargs["model"]))
    monkeypatch.setattr("agent.chat_completion_helpers._update_fallback_context_compressor", lambda agent: None)
    static = "Stable instructions.\nModel: memory-decoy\nProvider: memory-decoy\n\n"
    prompt = static + "Model: fixture-primary\nProvider: fixture-primary"
    agent._cached_system_prompt = prompt
    agent._cached_system_prompt_static = static
    agent.ephemeral_system_prompt = "Ephemeral instructions."
    agent._current_turn_timestamp = "2026-10-19T20:00:00+00:00"
    st = state()
    st.agent = agent
    if dropped:
        st.response = _build_partial_stream_stub(
            "assistant", "Preparing the action.", "", "fixture", None, dropped_tool_names=["fixture"],
        )
    loop = response_loop_state(st, prompt)
    history = deepcopy(st.messages)
    for expected_index in (1, 2):
        loop._retry = TurnRetryState()
        loop.truncated_tool_call_retries = 4
        loop.length_continue_retries = 3
        if decorated:
            loop.api_messages = apply_anthropic_cache_control(
                loop.api_messages, cache_ttl="5m", native_anthropic=True, static_system_prefix=static,
            )
        checked = _run_phase(check_api_response, agent, loop)
        assert checked.action == "break"
        assert agent._fallback_index == expected_index
        assert loop.active_system_prompt == agent._cached_system_prompt
        expected = static + f"Model: {agent.model}\nProvider: {agent.provider}"
        assert loop.active_system_prompt == expected
        effective = expected + "\n\nEphemeral instructions."
        content = loop.api_messages[0]["content"]
        if decorated:
            assert isinstance(content, list)
            assert content[0]["text"] == static
            assert "".join(part["text"] for part in content) == effective
            assert any("cache_control" in part for part in content)
        else:
            assert content == effective
        assert _run_phase(apply_retry_restarts, agent, loop).action == "continue"
        assert not loop._preflight_compression_blocked
        wire, effective_system = build_api_messages(
            agent, loop.messages, current_turn_user_idx=0,
            ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
            active_system_prompt=loop.active_system_prompt,
        )
        assert effective_system == effective
        assert wire[0] == {"role": "system", "content": effective}
        assert loop.messages == history
        assert agent._cached_system_prompt_static == static
        loop.api_messages = wire


@pytest.mark.parametrize("mode", ["retry", "exhausted", "activation-failed"])
def test_incomplete_action_without_route_change_keeps_loop_prompt(mode, monkeypatch):
    from agent.conversation_loop import _run_phase
    from agent.turn_response_check import check_api_response
    from tests.agent.test_provider_fallback import _make_agent

    routes = [] if mode == "exhausted" else [{"provider": "fixture", "model": "fixture-model"}]
    agent = _make_agent(fallback_model=routes)
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *args, **kwargs: (None, None))
    prompt = "Stable instructions.\nModel: fixture-primary\nProvider: fixture-primary"
    agent._cached_system_prompt = prompt
    st = state()
    st.agent = agent
    loop = response_loop_state(st, prompt)
    if mode == "retry":
        loop.truncated_tool_call_retries = 0
    verdict = _run_phase(check_api_response, agent, loop)
    assert verdict.action == ("continue" if mode == "retry" else "return")
    assert not loop._retry.restart_with_rebuilt_messages
    assert loop.active_system_prompt == prompt
    assert agent._cached_system_prompt == prompt
    assert loop.api_messages == [{"role": "system", "content": prompt}]
