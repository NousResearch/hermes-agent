"""Structural no-op backoff (#93022).

A compression attempt that finds nothing eligible inside the protection
window (too few messages / empty window / post-handoff residue) is "nothing
to compress right now", not an ineffective attempt: it must defer retries
transiently instead of arming the permanent anti-thrash breaker, so a short
session can still auto-compact after it grows real compressible material.
"""

import time
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _compressor(protect_first_n: int = 1) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def test_insufficient_messages_backs_off_without_strike():
    """Too few messages -> structural backoff, breaker stays untouched."""
    compressor = _compressor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]

    result = compressor.compress(messages, current_tokens=90_000)

    assert result == messages
    assert compressor._ineffective_compression_count == 0
    assert compressor._structural_no_op_backoff_until > 0.0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "insufficient_messages"


def test_no_compressible_window_backs_off_without_strike():
    """Transcript inside the tail budget -> backoff, breaker untouched."""
    compressor = _compressor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "turn three"},
        {"role": "assistant", "content": "answer three"},
        {"role": "user", "content": "latest request in protected tail"},
    ]

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        result = compressor.compress(messages, current_tokens=90_000)

    assert result == messages
    assert compressor._ineffective_compression_count == 0
    assert compressor._structural_no_op_backoff_until > 0.0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "no_compressible_window"


def test_gate_blocked_during_backoff_then_resumes():
    """should_compress defers during the backoff and recovers after it lapses.

    The transcript sits over the compression threshold the whole time; only
    the clock changes, proving the block is transient rather than a latched
    breaker state.
    """
    compressor = _compressor()

    # While the structural backoff is live the gate must say blocked.
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0
    with patch.object(
        compressor, "_automatic_compression_blocked", return_value=True
    ):
        should, reason = compressor.should_compress_info(prompt_tokens=300_000)
        assert should is False
        assert reason is not None
        assert reason.startswith("structural_backoff:")
        assert compressor._compression_block_reason().startswith(
            "structural_backoff:"
        )

    # After the backoff lapses nothing blocks: same over-threshold
    # transcript compresses again (real gate, real state).
    compressor._structural_no_op_backoff_until = (
        time.monotonic() - 1.0
    )
    should, reason = compressor.should_compress_info(prompt_tokens=300_000)
    assert should is True
    assert reason is None


SUMMARY_RESPONSE = "fresh replacement summary body"


def _messages_with_old_handoff():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": (
            "CONTEXT SUMMARY (from previous session):\nold summary body"
        )},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_forced_attempt_and_success_lift_the_backoff():
    """Manual /compress clears an active backoff; a completed boundary lifts it.

    Both are proof the transcript is being actively worked on — neither may
    leave auto-compaction deferred by a stale structural no-op.
    """
    compressor = _compressor()
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(SUMMARY_RESPONSE),
    ):
        compressed = compressor.compress(
            _messages_with_old_handoff(), force=True
        )

    assert compressor._structural_no_op_backoff_until == 0.0
    # The forced attempt actually committed a boundary.
    assert len(compressed) < len(_messages_with_old_handoff())


def test_real_attempt_underperformance_still_strikes_breaker():
    """Only genuine attempted-but-underperformed compressions strike.

    A real summary pass that saves <10% goes through the ineffective
    verdict (persisted); structural no-ops must not touch that counter —
    that distinction IS this fix.
    """
    compressor = _compressor()
    before = compressor._ineffective_compression_count
    compressor._record_ineffective_compression_verdict(before + 1)
    assert compressor._ineffective_compression_count == before + 1

    compressor._record_structural_no_op("test reason")
    assert compressor._structural_no_op_backoff_until > 0.0
    assert compressor._ineffective_compression_count == before + 1
