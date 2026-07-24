"""Anti-thrash recovery: the tripped guard must not be permanent (#14694).

When two consecutive compactions each fail to clear the threshold, the
anti-thrashing breaker blocks automatic compaction. Before this fix the block
was permanent for the life of the session: nothing ever decremented
``_ineffective_compression_count`` (or ``_fallback_compression_streak``)
while blocked, so a session whose middle region was briefly too small to
compact never auto-compacted again — it grew unbounded until the provider's
hard context limit, and only ``/new`` or ``/reset`` recovered it.

The recovery contract pinned here:

* After ``_ANTI_THRASH_RECOVERY_SECONDS`` of continuous block, the gate
  grants exactly ONE probation probe: tripped counters drop to 1 strike
  (persisted) and the gate reports unblocked once.
* An ineffective probe re-trips the guard on the very next verdict, and the
  next recovery waits a FULL fresh window (no immediate re-probe loop).
* An effective probe (or any fitting real-usage reading) fully clears the
  counters through the existing ``update_from_response`` path.
* The recovery clock is armed lazily on the first blocked evaluation and is
  NOT durable: a process restart that loads a durable tripped counter
  (#69872) starts a full fresh window blocked — a restart must never disarm
  or shorten the guard (#54923).
* The protection itself is preserved: inside the window the gate stays
  blocked exactly as before.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor(threshold_tokens: int = 10_000) -> ContextCompressor:
    cc = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    cc.threshold_tokens = threshold_tokens
    return cc


def _trip(cc: ContextCompressor) -> None:
    """Arm the breaker exactly as two ineffective real-usage verdicts do."""
    cc._record_ineffective_compression_verdict(2)


class TestRecoveryWindow:
    def test_blocked_within_window_unblocked_after(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            # First blocked evaluation arms the clock and stays blocked.
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS - 1,
        ):
            # Still inside the window: protection intact.
            assert cc.should_compress(cc.threshold_tokens + 1) is False
            assert cc._ineffective_compression_count == 2
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            # Window elapsed: exactly one probe is granted.
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        # Probation, not amnesty: one strike remains armed.
        assert cc._ineffective_compression_count == 1

    def test_ineffective_probe_re_trips_and_waits_a_full_fresh_window(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        probe_time = base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1
        with patch(
            "agent.context_compressor.time.monotonic", return_value=probe_time
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
            # The probe compaction completes but does not clear the threshold.
            cc._verify_compaction_cleared_threshold = True
            cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
            assert cc._ineffective_compression_count == 2
            # Re-tripped: blocked again immediately (arms a new clock).
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=probe_time + cc._ANTI_THRASH_RECOVERY_SECONDS - 5,
        ):
            # No immediate re-probe loop: the second window is full length,
            # measured from the re-trip, not the original trip.
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=probe_time + cc._ANTI_THRASH_RECOVERY_SECONDS + 5,
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True

    def test_effective_probe_clears_the_guard_completely(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
            cc._verify_compaction_cleared_threshold = True
            cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 500})
        assert cc._ineffective_compression_count == 0
        assert cc._anti_thrash_recovery_deadline == 0.0

    def test_fallback_streak_breaker_recovers_too(self):
        cc = _compressor()
        cc._fallback_compression_streak = 2
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert cc._fallback_compression_streak == 1

    def test_under_threshold_never_arms_the_clock(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            # Under threshold: gate never evaluated, clock untouched.
            assert cc.should_compress(cc.threshold_tokens - 1) is False
        assert cc._anti_thrash_recovery_deadline == 0.0

    def test_untripped_guard_disarms_a_stale_clock(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert cc._anti_thrash_recovery_deadline > 0.0
        # A fitting real-usage reading clears the counter mid-window.
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 500})
        with patch("agent.context_compressor.time.monotonic", return_value=base + 1):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        # The stale clock was disarmed, so a LATER trip starts a full window.
        assert cc._anti_thrash_recovery_deadline == 0.0


class TestRestartSemantics:
    def test_restart_with_durable_tripped_counter_waits_a_full_window(self, tmp_path):
        """#69872 x #14694: a restart must not disarm OR shorten the guard."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        db.set_compression_ineffective_count("sess-1", 2)

        cc = _compressor()
        cc.bind_session_state(session_db=db, session_id="sess-1")
        assert cc._ineffective_compression_count == 2
        # The recovery clock is process-local and must come up disarmed.
        assert cc._anti_thrash_recovery_deadline == 0.0
        base = 5000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.monotonic",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        # The probation reset is durable, so sibling agents on the same
        # session row (gateway hygiene) unblock too.
        assert db.get_compression_ineffective_count("sess-1") == 1

    def test_session_reset_disarms_the_recovery_clock(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.monotonic", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert cc._anti_thrash_recovery_deadline > 0.0
        cc.on_session_reset()
        assert cc._anti_thrash_recovery_deadline == 0.0
        assert cc._ineffective_compression_count == 0
