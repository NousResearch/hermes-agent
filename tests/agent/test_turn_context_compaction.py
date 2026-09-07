"""Unit tests for ``agent.turn_context_compaction`` (turn-start compaction extracted
from ``build_turn_context``)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_context_compaction import (
    CompactionOutcome,
    _codex_native_auto_compaction,
    _rearm_uncompressed_overflow_warn,
    run_turn_start_compaction,
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


def test_multimodal_content_forces_real_estimate():
    agent = _agent()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    with patch(
        "agent.turn_context._preflight_request_tokens", return_value=9_999
    ) as est:
        _rearm_uncompressed_overflow_warn(agent, msgs, "sys")
    est.assert_called_once()
    agent._clear_context_overflow_warn.assert_not_called()


def test_preflight_gate_skips_small_transcripts():
    agent = _agent(compression_enabled=True)
    agent.context_compressor.should_compress = MagicMock()
    msgs = [{"role": "user", "content": "hi"}]
    with patch("agent.turn_context._preflight_request_tokens") as est:
        out = run_turn_start_compaction(
            agent, messages=msgs, system_message=None, active_system_prompt="sys",
            conversation_history=None, current_turn_user_idx=0, user_message="hi",
            effective_task_id="t",
        )
    est.assert_not_called()
    agent.context_compressor.should_compress.assert_not_called()
    assert out.messages is msgs
