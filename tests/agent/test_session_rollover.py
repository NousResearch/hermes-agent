"""Behavioral coverage for the opt-in session rollover pressure guard."""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from agent.session_rollover import (
    RolloverBudgetUnavailable,
    RolloverGuard,
    measure_rollover_pressure,
)
from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor
from run_agent import AIAgent
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


def _tool_definitions(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent(*tool_names: str, session_db=None) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_definitions(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = True
    agent.save_trajectories = False
    agent.max_tokens = None
    agent.context_compressor.context_length = 20_000
    agent.context_compressor.threshold_tokens = 15_000
    return agent


def _response(content="done", *, tool_calls=None, usage=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    response = SimpleNamespace(choices=[choice], model="test/model")
    response.usage = SimpleNamespace(**usage) if usage else None
    return response


def _tool_call():
    return SimpleNamespace(
        id="rollover-tool-1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"synthetic.txt"}'),
    )


def test_pressure_snapshot_uses_provider_anchor_plus_appended_growth_without_mutating_prefix():
    base = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    growth = {"role": "tool", "content": "x" * 3_000}
    messages = [*base, growth]
    cached_system = "byte-stable system prefix"
    tools = [{"type": "function", "function": {"name": "fixed_tool"}}]
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=20_000, threshold_tokens=8_000),
        compression_enabled=True,
        api_mode="chat_completions",
        max_tokens=1_000,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        tools=tools,
        _cached_system_prompt=cached_system,
        _usage_anchor=capture_usage_anchor(6_000, 250, base),
    )
    guard = RolloverGuard(1_000, 250, 9_000, 100)

    expected = anchored_context_tokens(messages, runtime._usage_anchor)
    assert expected == 6_250 + estimate_messages_tokens_rough([growth])
    snapshot = guard.inspect(runtime, messages, system_prompt=cached_system)

    assert snapshot.pressure_tokens == expected
    assert snapshot.context_window == 20_000
    assert snapshot.threshold_tokens == 8_000
    assert snapshot.emergency_threshold_tokens == 8_000
    assert snapshot.rollover_threshold_tokens == 6_750
    assert snapshot.rollover_requested is True
    assert guard.phase == "requested"
    assert runtime._cached_system_prompt == cached_system
    assert runtime.tools == tools
    assert messages == [*base, growth]

def test_resolved_output_limit_caps_emergency_before_the_window():
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=20_000, threshold_tokens=18_000),
        compression_enabled=True,
        api_mode="chat_completions",
        max_tokens=5_000,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        tools=[],
        _cached_system_prompt="system",
        _usage_anchor=None,
    )
    guard = RolloverGuard(500, 500, 19_000, 100)

    snapshot = guard.inspect(
        runtime, [{"role": "user", "content": "hello"}], system_prompt="system"
    )

    assert snapshot.threshold_tokens == 18_000
    assert snapshot.emergency_threshold_tokens == 15_000
    assert snapshot.rollover_threshold_tokens == 14_000

def test_message_limit_defers_hygiene_and_invalidates_authoring_before_the_valve():
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=20_000, threshold_tokens=9_000),
        compression_enabled=True,
        api_mode="chat_completions",
        max_tokens=None,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        tools=[],
        _cached_system_prompt="system",
        _usage_anchor=None,
    )
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    pending = {"role": "user", "content": "accepted"}
    guard = RolloverGuard(500, 500, 10_000, 3)

    snapshot = guard.inspect(
        runtime, messages, system_prompt="system", pending_message=pending
    )

    assert snapshot.message_count == 3
    assert snapshot.rollover_requested is True
    assert snapshot.skip_hygiene is True
    assert snapshot.reason == "message_limit"

    guard.phase = "authoring"
    emergency = guard.inspect(
        runtime, messages, system_prompt="system", pending_message=pending
    )
    assert emergency.rollover_fallback is True
    assert emergency.skip_hygiene is False
    assert guard.phase == "invalidated"


def test_real_loop_yields_before_compaction_and_persists_accepted_input(session_db):
    agent = _agent(session_db=session_db)
    history = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent._usage_anchor = capture_usage_anchor(14_000, 100, history)
    agent._session_rollover = RolloverGuard(500, 500, 18_000, 100)
    system_prefix = agent._cached_system_prompt
    tool_prefix = copy.deepcopy(agent.tools)

    with (
        patch.object(agent, "_compress_context") as compress,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "accepted input survives rollover",
            conversation_history=history,
            persist_user_message="accepted input survives rollover",
        )

    assert agent.client.chat.completions.create.call_count == 0
    compress.assert_not_called()
    assert result["rollover_requested"] is True
    assert result["rollover_fallback"] is False
    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert any(
        message.get("role") == "user"
        and message.get("content") == "accepted input survives rollover"
        for message in session_db.get_messages(agent.session_id)
    )
    assert result["messages"][-1]["role"] == "assistant"
    assert agent._cached_system_prompt == system_prefix
    assert agent.tools == tool_prefix

def test_pre_api_guard_yields_after_request_only_growth_and_refunds_iteration():
    agent = _agent()
    history = [
        {"role": "user", "content": "prior request"},
        {"role": "assistant", "content": "prior answer"},
    ]
    agent._usage_anchor = capture_usage_anchor(10_500, 100, history)
    agent._session_rollover = RolloverGuard(1_000, 1_000, 18_000, 100)
    system_prefix = agent._cached_system_prompt
    tool_prefix = copy.deepcopy(agent.tools)

    with (
        patch("agent.turn_context._maybe_title_session_at_turn_start"),
        patch("agent.turn_context._collect_pre_llm_call_context", return_value="x" * 12_000),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue the approved task", conversation_history=history)

    assert agent.client.chat.completions.create.call_count == 0
    assert result["rollover_requested"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert agent.iteration_budget.used == 0
    assert agent._cached_system_prompt == system_prefix
    assert agent.tools == tool_prefix


def test_authoring_turn_skips_proactive_compaction_but_keeps_provider_path():
    agent = _agent()
    history = [
        {"role": "user", "content": "prior request"},
        {"role": "assistant", "content": "prior answer"},
    ]
    agent._usage_anchor = capture_usage_anchor(10_000, 100, history)
    agent._session_rollover = RolloverGuard(500, 500, 18_000, 100)
    agent._session_rollover.phase = "authoring"
    agent.compression_idle_compact_after_seconds = 1
    agent._last_activity_ts = time.time() - 3_600
    agent.context_compressor._micro_compact_enabled = True
    system_prefix = agent._cached_system_prompt
    tool_prefix = copy.deepcopy(agent.tools)
    agent.client.chat.completions.create.return_value = _response(
        "The handoff packet is ready.",
        usage={"prompt_tokens": 10_500, "completion_tokens": 50, "total_tokens": 10_550},
    )

    with (
        patch("agent.turn_context._maybe_title_session_at_turn_start"),
        patch.object(agent, "_compress_context") as compress,
        patch.object(agent.context_compressor, "_micro_compact") as micro_compact,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("author the handoff", conversation_history=history)

    micro_compact.assert_not_called()

    compress.assert_not_called()
    assert agent.client.chat.completions.create.call_count == 1
    assert result["completed"] is True
    assert result["rollover_requested"] is False
    assert result["rollover_fallback"] is False
    assert agent._cached_system_prompt == system_prefix
    assert agent.tools == tool_prefix


def test_oversized_tool_result_invalidates_before_emergency_compaction():
    agent = _agent("read_file")
    agent.context_compressor.context_length = 20_000
    agent.context_compressor.threshold_tokens = 15_000
    agent._session_rollover = RolloverGuard(1_000, 1_000, 18_000, 1_000)
    history = [{"role": "user", "content": "earlier work"}, {"role": "assistant", "content": "recorded"}]
    agent._usage_anchor = capture_usage_anchor(10_500, 100, history)
    agent.client.chat.completions.create.side_effect = [
        _response(
            content=None,
            tool_calls=[_tool_call()],
            usage={"prompt_tokens": 11_000, "completion_tokens": 100, "total_tokens": 11_100},
        ),
        _response(
            "Search complete.",
            usage={"prompt_tokens": 1_000, "completion_tokens": 50, "total_tokens": 1_050},
        ),
    ]
    compaction_phases = []

    def compact(messages, system_message, **_kwargs):
        compaction_phases.append(agent._session_rollover.phase)
        compacted = [
            dict(message, content="[tool result compacted]")
            if isinstance(message, dict) and message.get("role") == "tool"
            else dict(message) if isinstance(message, dict) else message
            for message in messages
        ]
        agent.context_compressor.awaiting_real_usage_after_compression = True
        return compacted, system_message

    with (
        patch("agent.turn_context._maybe_title_session_at_turn_start"),
        # Below the native 24k-character turn cap, but larger than the remaining context runway.
        patch("model_tools.handle_function_call", return_value="r" * 20_000),
        patch.object(agent, "_compress_context", side_effect=compact),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("read the synthetic file", conversation_history=history)

    assert compaction_phases == ["invalidated"]
    assert result["rollover_requested"] is False
    assert result["rollover_fallback"] is True
    assert result["completed"] is True


def test_unsupported_native_runtime_budget_fails_closed():
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=50_000, threshold_tokens=30_000),
        compression_enabled=True,
        api_mode="codex_app_server",
        codex_app_server_auto_compaction="native",
        max_tokens=None,
        provider="openai-codex",
        base_url="",
        tools=[],
        _cached_system_prompt="system",
        _usage_anchor=None,
    )
    guard = RolloverGuard(1_000, 500, 40_000, 1_000)

    with pytest.raises(RolloverBudgetUnavailable):
        guard.inspect(runtime, [], system_prompt="system")

def test_missing_agent_output_limit_fails_closed():
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=20_000, threshold_tokens=15_000),
        compression_enabled=True,
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        tools=[],
        _cached_system_prompt="system",
        _usage_anchor=None,
    )
    guard = RolloverGuard(1_000, 500, 18_000, 100)

    with pytest.raises(RolloverBudgetUnavailable, match="agent output-token limit"):
        guard.inspect(runtime, [], system_prompt="system")


def test_unanchored_measurement_without_resolved_system_prompt_fails_closed():
    runtime = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=20_000, threshold_tokens=15_000),
        compression_enabled=True,
        api_mode="chat_completions",
        tools=[],
        _cached_system_prompt=None,
        _usage_anchor=None,
    )

    with pytest.raises(RolloverBudgetUnavailable, match="requires the resolved system prompt"):
        measure_rollover_pressure(runtime, [{"role": "user", "content": "hello"}])


def test_invalidated_guard_allows_native_turn_after_runtime_budget_shrinks(session_db):
    agent = _agent(session_db=session_db)
    guard = RolloverGuard(1_000, 500, 18_000, 100)
    agent._session_rollover = guard
    guard.invalidate("native_overflow", agent=agent)
    agent.context_compressor.context_length = 1_000
    agent.context_compressor.threshold_tokens = 850
    agent.client.chat.completions.create.return_value = _response()

    with patch("agent.turn_context._maybe_title_session_at_turn_start"):
        result = agent.run_conversation("preserve this input during recovery", conversation_history=[])

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert result["rollover_requested"] is False
    assert result["rollover_fallback"] is True
    assert any(message["role"] == "user" and message["content"] == "preserve this input during recovery"
               for message in session_db.get_messages(agent.session_id))
