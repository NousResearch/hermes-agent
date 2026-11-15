"""Unit tests for ``agent.turn_context_compaction`` (turn-start compaction extracted
from ``build_turn_context``)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_context_compaction import (
    CompactionOutcome,
    _codex_native_auto_compaction,
    run_turn_start_compaction,
    should_defer_preflight_to_real_usage,
)

def _agent(**kw):
    compressor = SimpleNamespace(
        protect_first_n=3, protect_last_n=3, threshold_tokens=1_000, context_length=8_000,
        summary_target_ratio=0.5,
    )
    base = dict(
        compression_enabled=False, context_compressor=compressor, session_id="s1",
        model="m", _clear_context_overflow_warn=MagicMock(),
    )
    base.update(kw)
    return SimpleNamespace(**base)

def test_codex_native_auto_compaction_gate():
    assert _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="native")
    )
    assert _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="OFF")
    )
    assert not _codex_native_auto_compaction(
        SimpleNamespace(api_mode="codex_app_server", codex_app_server_auto_compaction="hermes")
    )
    assert not _codex_native_auto_compaction(SimpleNamespace(api_mode="chat_completions"))

def test_disabled_compression_rearms_overflow_warn_when_under_window():
    agent = _agent()
    msgs = [{"role": "user", "content": "hi"}]
    out = run_turn_start_compaction(
        agent, messages=msgs, system_message=None, active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=0, user_message="hi",
        effective_task_id="t",
    )
    assert isinstance(out, CompactionOutcome)
    assert out.messages is msgs and out.current_turn_user_idx == 0
    assert out.compressed is False and out.blocked is False
    agent._clear_context_overflow_warn.assert_called_once()
    assert agent._turn_received_provider_response is False
    assert agent._turn_preflight_display_snapshot is None


def test_should_defer_preflight_to_real_usage_respects_180k_pricing_cliff_cap():
    cc = SimpleNamespace(
        threshold_tokens=157_286,
        context_length=1_048_576,
        last_real_prompt_tokens=140_000,
        awaiting_real_usage_after_compression=False,
        _provider_omits_usage=False,
    )
    agent = SimpleNamespace(context_compressor=cc, _request_pressure_anchored=False)
    # Between 157k threshold and 180k cliff cap: un-anchored rough estimate defers once.
    assert should_defer_preflight_to_real_usage(agent, 165_000) is True
    # At or above 180k cliff cap: never defers past the 200k long-context pricing cliff.
    assert should_defer_preflight_to_real_usage(agent, 180_000) is False
    assert should_defer_preflight_to_real_usage(agent, 205_000) is False
    # Anchored estimate never defers at any size.
    assert should_defer_preflight_to_real_usage(agent, 165_000, anchored=True) is False


def test_turn1_oversized_tool_batch_excluded_from_protected_head():
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor(model="google/gemini-4-argon", quiet_mode=True, protect_first_n=3, protect_last_n=6)
    cc.tail_token_budget = 10_000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial user task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "skill_view", "arguments": '{"name":"s"}'}}
                for i in range(6)
            ],
        },
    ]
    for i in range(6):
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "S" * 6_000})
    for i in range(8):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + ("t" * 400)})

    compress_start, compress_end = cc._compress_window(messages)
    # System prompt (0) and initial user message (1) remain protected, while the oversized Turn-1 tool batch starts at 2.
    assert compress_start == 2
    assert compress_end > compress_start

