"""Typed turns fail closed at pressure/overflow boundaries without rewriting history."""
import copy
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression, compress_after_tool_results
from agent.turn_overflow import recover_from_overflow, FailoverReason
from agent.compression_facade import CompressionFacadeMixin
from gateway.internal_events import create_gateway_system_event


def fixture():
    _, marker = create_gateway_system_event(content="host metadata", session_key="key", expected_session_id="physical",
        event_id="delivery", event_kind="external_tool_completed", plugin_id="fixture",
        expected_route={"profile_name":"default", "platform":"telegram", "user_id":"42", "chat_id":"42", "topic_id":""},
        eligibility_check=lambda: True)
    history = [{"role":"user", "content":"original"}, {"role":"assistant", "content":"answer"}]
    messages = history + [{"role":"developer", "content":"host metadata"}, {"role":"tool", "content":"new tool evidence", "tool_call_id":"call"}]
    agent = MagicMock(_gateway_system_event=marker, session_id="physical", _cached_system_prompt="SYSTEM",
                      compression_enabled=True, model="fixture", base_url="https://fixture.invalid", log_prefix="")
    agent.context_compressor.threshold_tokens = 2000
    agent.context_compressor.should_compress.return_value = True
    agent.context_compressor.get_active_compression_failure_cooldown.return_value = None
    return agent, history, messages


@pytest.mark.parametrize("pressure,provider_overflow,expected", [(True,False,"return"), (False,True,"return"), (False,False,"fallthrough")])
def test_typed_preflight_preserves_context_and_never_compresses(pressure, provider_overflow, expected):
    agent, history, messages = fixture()
    before = copy.deepcopy(messages)
    agent.context_compressor.should_compress.return_value = pressure
    v = PreflightGateVerdict(**{f.name:None for f in fields(PreflightGateVerdict)})
    v.messages, v.conversation_history, v.active_system_prompt = messages, history, "SYSTEM"
    v.compression_attempts, v.api_call_count, v._preflight_compression_blocked = 0, 1, False
    with patch('agent.turn_preflight._review_fork_first_request_pending', return_value=False):
        result = run_preflight_compression(agent, v, compressor=agent.context_compressor, request_pressure_tokens=1000,
            provider_overflow_preflight=provider_overflow, defer_preflight=lambda n:False, moa_prepared_request=None,
            system_message="SYSTEM", user_message="host metadata", max_compression_attempts=3, effective_task_id="task")
    assert result.action == expected
    if expected == "return":
        assert result.result['completed'] is False
        assert result.result['failed'] is True
        agent._persist_session.assert_called_once_with(messages, history)
    assert result.messages is messages and messages == before
    assert result.conversation_history is history and result.active_system_prompt == "SYSTEM"
    agent._compress_context.assert_not_called()
    assert agent.session_id == "physical" and agent._cached_system_prompt == "SYSTEM"


def test_typed_post_tool_keeps_new_evidence_without_compression_or_pruning():
    agent, history, messages = fixture()
    before = copy.deepcopy(messages)
    result = compress_after_tool_results(agent, messages=messages, system_message="SYSTEM", user_message="host metadata",
        active_system_prompt="SYSTEM", conversation_history=history, compression_attempts=0, max_compression_attempts=3,
        effective_task_id="task", final_response=None, turn_exit_reason=None)
    assert result.messages is messages and messages == before
    assert result.conversation_history is history and result.active_system_prompt == "SYSTEM"
    agent._compress_context.assert_not_called()
    agent.context_compressor.prune_tool_results_only.assert_not_called()


@pytest.mark.parametrize("reason", [FailoverReason.context_overflow, FailoverReason.payload_too_large])
def test_typed_provider_overflow_never_rotates_or_updates_context_model(reason):
    agent, history, messages = fixture()
    before = copy.deepcopy(messages)
    result = recover_from_overflow(agent, RuntimeError("too many tokens"), SimpleNamespace(reason=reason), MagicMock(),
        status_code=413, error_msg="too many tokens", wrapped_output_cap_budget=None, messages=messages,
        api_messages=messages, system_message="SYSTEM", active_system_prompt="SYSTEM", conversation_history=history,
        approx_tokens=1000, compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="task")
    assert result.action == "return" and result.result['completed'] is False
    assert result.result['failed'] is True
    agent._persist_session.assert_called_once_with(messages, history)
    assert not result.result.get('compression_exhausted')
    assert messages == before and result.messages is messages and result.conversation_history is history
    agent._compress_context.assert_not_called()
    agent.context_compressor.update_model.assert_not_called()
    assert agent.session_id == "physical" and agent._cached_system_prompt == "SYSTEM"


def test_shared_compression_entry_refuses_typed_retry_call_before_any_mutation():
    agent, history, messages = fixture()
    before = copy.deepcopy(messages)
    with pytest.raises(ValueError, match="typed gateway"):
        CompressionFacadeMixin._compress_context(agent, messages, "SYSTEM")
    assert messages == before and agent.session_id == "physical" and agent._cached_system_prompt == "SYSTEM"


@pytest.mark.parametrize("failed", [False, True])
def test_typed_turn_releases_compression_guard_on_every_exit(monkeypatch, failed):
    from agent import conversation_loop, conversation_compression
    agent, history, messages = fixture()
    marker = agent._gateway_system_event
    agent._gateway_system_event = None
    agent._conversation_root_id.return_value = None

    def turn(*args, **kwargs):
        agent._gateway_system_event = kwargs["gateway_system_event"]
        with pytest.raises(ValueError, match="typed gateway"):
            CompressionFacadeMixin._compress_context(agent, messages, "SYSTEM", force=True)
        if failed:
            raise RuntimeError("typed turn failed")
        return {"completed": True, "messages": messages}

    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", turn)
    if failed:
        with pytest.raises(RuntimeError, match="typed turn failed"):
            conversation_loop.run_conversation(agent, "completion", gateway_system_event=marker)
    else:
        conversation_loop.run_conversation(agent, "completion", gateway_system_event=marker)
    compressed = (history, "SYSTEM")
    compressor = MagicMock(return_value=compressed)
    monkeypatch.setattr(conversation_compression, "compress_context", compressor)
    monkeypatch.setattr(conversation_compression, "resolve_context_compression_timeouts", lambda: (0, 0))
    with patch("agent.compression_facade._mirror_result_onto_live_lists"), \
         patch("agent.compression_facade._rebind_caller_session_context"), \
         patch("agent.prompt_cache_scope.declared_conversation_scope_safe", return_value=None):
        assert CompressionFacadeMixin._compress_context(agent, messages, "SYSTEM", force=True) == compressed
    compressor.assert_called_once()
