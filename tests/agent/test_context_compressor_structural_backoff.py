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


def _sidecar_messages():
    """Transcript whose real bill rides in encrypted reasoning sidecars (#125920).

    The visible text is tiny (every row fits the tail budget, so the middle
    window is empty), while the two assistant rows before the last user turn
    carry stale ``codex_reasoning_items`` the provider re-bills on every replay.
    The newest assistant row holds the active turn's sidecar, which must stay.
    """
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {
            "role": "assistant",
            "content": "answer one",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "STALE-BLOB-ONE"}
            ],
        },
        {"role": "user", "content": "turn two"},
        {
            "role": "assistant",
            "content": "answer two",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "STALE-BLOB-TWO"}
            ],
        },
        {"role": "user", "content": "latest request in protected tail"},
        {
            "role": "assistant",
            "content": "latest answer",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "ACTIVE-TURN-BLOB"}
            ],
        },
    ]


def test_no_compressible_window_prunes_stale_replay_instead_of_backing_off():
    """Empty middle + stale replay sidecars -> prune them, do not arm the backoff.

    The local estimator prices encrypted_content at zero, so the tail budget
    swallows the whole transcript and the middle window is empty. Returning
    unchanged here used to skip the success-path replay prune AND lock it out
    for 300s via the structural backoff (#125920).
    """
    compressor = _compressor()
    messages = _sidecar_messages()

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        result = compressor.compress(messages, current_tokens=300_000)

    # Stale sidecars (before the last user turn) are stripped from the result.
    assert "codex_reasoning_items" not in result[2]
    assert "codex_reasoning_items" not in result[4]
    # The active turn's sidecar survives: the Responses API replays that chain.
    assert result[6]["codex_reasoning_items"][0]["encrypted_content"] == "ACTIVE-TURN-BLOB"
    # Real progress was made: no structural backoff may lock out the next try.
    assert compressor._last_compression_made_progress is True
    assert compressor._structural_no_op_backoff_until == 0.0
    assert compressor._ineffective_compression_count == 0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") is None
    assert telemetry.get("pruned_stale_replay_messages") == 2


def test_insufficient_messages_prunes_stale_replay():
    """Short transcript with replay-heavy rows: prune still runs (#125920).

    "Too few messages" says nothing about the sidecar bill: the visible chat
    can be two turns long while stale replay blobs dwarf the threshold.
    """
    compressor = _compressor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "answer to the head user turn",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "STALE-BLOB"}
            ],
        },
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "latest answer"},
    ]

    result = compressor.compress(messages, current_tokens=300_000)

    assert "codex_reasoning_items" not in result[1]
    assert compressor._last_compression_made_progress is True
    assert compressor._structural_no_op_backoff_until == 0.0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") is None
    assert telemetry.get("pruned_stale_replay_messages") == 1


def test_prune_rescue_keeps_input_untouched():
    """The rescue prunes a private copy; the caller's transcript is never mutated in place."""
    compressor = _compressor()
    messages = _sidecar_messages()
    snapshot = [dict(msg) for msg in messages]

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        compressor.compress(messages, current_tokens=300_000)

    assert messages == snapshot


def test_rescue_prune_does_not_strike_the_breaker():
    """The rescue prune shrinks the provider bill, not the local estimate (ciphertext
    prices at zero, #100611): arming the real-usage verdict on it would strike the
    breaker on the very next count >= threshold, and two rescues disarm
    auto-compaction on exactly the short sessions the prune keeps alive (AI review
    probe: strikes 1, 2, tripped by turn 1 on the unfixed head)."""
    compressor = _compressor()
    messages = _sidecar_messages()

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        compressor.compress(messages, current_tokens=300_000)

    assert compressor._last_progress_was_sidecar_prune is True
    # The committed boundary records it: backoff lifted, verdict NOT armed.
    compressor.record_completed_compaction(sidecar_prune=True)
    assert compressor._structural_no_op_backoff_until == 0.0
    assert compressor._verify_compaction_cleared_threshold is False

    # Real usage still >= threshold (the prune cannot lower the estimate): no strike.
    compressor.update_from_response({"prompt_tokens": compressor.threshold_tokens + 5_000})
    assert compressor._ineffective_compression_count == 0
    assert compressor._tripped() is False

    # A second rescue round must stay strike-free too.
    compressor.update_from_response({"prompt_tokens": compressor.threshold_tokens + 5_000})
    assert compressor._ineffective_compression_count == 0

    # Control: a real rewrite that arms the verdict still strikes when the count
    # stays over threshold — the breaker itself is untouched by this change.
    compressor._verify_compaction_cleared_threshold = True
    compressor.update_from_response({"prompt_tokens": compressor.threshold_tokens + 5_000})
    assert compressor._ineffective_compression_count == 1


def test_rescue_prune_strips_persistence_markers():
    """The rescue is a NEW compress() exit with changed content, so it owes the
    same final scrub as the success path (#57491): a leaked ``_db_persisted``
    marker makes the rotation flush skip the row, dropping the pruned transcript
    from state.db."""
    compressor = _compressor()
    messages = _sidecar_messages()
    for row in messages:
        if isinstance(row, dict):
            row["_db_persisted"] = True

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        result = compressor.compress(messages, current_tokens=300_000)

    assert compressor._last_progress_was_sidecar_prune is True
    assert all("_db_persisted" not in row for row in result if isinstance(row, dict))
    # The prune itself still happened on this run.
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("pruned_stale_replay_messages") == 2
