"""Unit tests for ``agent.turn_context_compaction`` (turn-start compaction extracted
from ``build_turn_context``)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_context_compaction import (
    CompactionOutcome,
    _codex_native_auto_compaction,
    _idle_compaction,
    _rearm_uncompressed_overflow_warn,
    _run_preflight_passes,
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


def test_idle_compaction_resets_astra_segment_only_after_real_rewrite():
    messages = [{"role": "user", "content": "old"}]
    compressed = [{"role": "assistant", "content": "summary"}]
    compressor = SimpleNamespace(
        threshold_tokens=100, summary_target_ratio=0.5,
        get_active_compression_failure_cooldown=lambda: None,
        last_compression_rough_tokens=0,
    )
    agent = _agent(
        compression_enabled=True, compression_idle_compact_after_seconds=1,
        _last_activity_ts=0, context_compressor=compressor,
        _emit_status=MagicMock(), _compress_context=MagicMock(return_value=(compressed, "sys")),
    )
    out = CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=list(messages),
        current_turn_user_idx=0,
    )
    with (
        patch("agent.turn_context._preflight_request_tokens", return_value=1_000),
        patch("agent.turn_context._should_idle_compact", return_value=True),
        patch("agent.turn_context_compaction.automatic_compaction_status_message", return_value=None),
        patch("agent.turn_context_compaction.conversation_history_after_compression", return_value=compressed),
        patch("agent.turn_context_compaction._reset_astra_segment_after_compaction") as reset,
        patch("agent.turn_context_compaction._reanchor", return_value=0),
    ):
        _idle_compaction(agent, out, "sys", "old", "task")

    reset.assert_called_once_with(agent)


def test_preflight_compaction_resets_astra_segment_once_across_passes():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "long"},
    ]
    compressed = [{"role": "assistant", "content": "summary"}]
    compressor = SimpleNamespace(
        threshold_tokens=100, context_length=1_000,
        should_compress=MagicMock(return_value=False),
    )
    agent = _agent(
        compression_enabled=True, context_compressor=compressor, max_compression_attempts=2,
        _emit_status=MagicMock(), _compress_context=MagicMock(return_value=(compressed, "sys")),
    )
    out = CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=list(messages),
        current_turn_user_idx=0,
    )
    with (
        patch("agent.turn_context._preflight_request_tokens", return_value=50),
        patch("agent.turn_context.compression_made_progress", return_value=True),
        patch("agent.turn_context_compaction.automatic_compaction_status_message", return_value=None),
        patch("agent.turn_context_compaction.conversation_history_after_compression", return_value=compressed),
        patch("agent.turn_context_compaction._reset_astra_segment_after_compaction") as reset,
    ):
        _run_preflight_passes(agent, out, compressor, 1_000, "sys", "task")

    reset.assert_called_once_with(agent)
