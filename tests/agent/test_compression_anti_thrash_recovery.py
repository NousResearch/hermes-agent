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
* The recovery clock is armed lazily on the first blocked evaluation and
  persisted on the session row as a wall-clock deadline (#100185): a fresh
  compressor that loads a durable tripped counter (#69872) with NO stored
  deadline starts a full window blocked — a restart must never disarm or
  shorten the guard (#54923) — while one that loads an armed deadline
  resumes that window instead of restarting it, so gateway agent rebuilds
  cannot block a session forever.
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


    def test_effective_probe_clears_the_guard_completely(self):
        cc = _compressor()
        _trip(cc)
        base = 1000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.time",
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
        with patch("agent.context_compressor.time.time", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.time",
            return_value=base + cc._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert cc._fallback_compression_streak == 1




class TestRestartSemantics:
    def test_restart_with_durable_tripped_counter_waits_a_full_window(self, tmp_path):
        """#69872 x #14694: a restart must not disarm OR shorten the guard."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        db.set_compression_ineffective_count("sess-1", 2)

        cc = _compressor()
        cc.bind_session_state(session_db=db, session_id="sess-1")
        assert cc._ineffective_compression_count == 2
        # No stored deadline yet -> the clock comes up disarmed.
        assert cc._anti_thrash_recovery_deadline == 0.0
        base = 5000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        with patch(
            "agent.context_compressor.time.time",
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
        with patch("agent.context_compressor.time.time", return_value=base):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert cc._anti_thrash_recovery_deadline > 0.0
        cc.on_session_reset()
        assert cc._anti_thrash_recovery_deadline == 0.0
        assert cc._ineffective_compression_count == 0


class TestDurableDeadline:
    """#100185: the gateway rebuilds the compressor on every cache eviction."""

    def _bound(self, db, session_id="sess-1"):
        cc = _compressor()
        cc.bind_session_state(session_db=db, session_id=session_id)
        return cc

    def test_fresh_compressors_resume_the_same_window(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        db.set_compression_ineffective_count("sess-1", 2)
        base = 5000.0
        first = self._bound(db)
        with patch("agent.context_compressor.time.time", return_value=base):
            assert first.should_compress(first.threshold_tokens + 1) is False
        # Deadline is durable, as a wall-clock epoch.
        assert db.get_compression_recovery_deadline("sess-1") == (
            base + first._ANTI_THRASH_RECOVERY_SECONDS
        )
        # Fresh compressor (gateway rebuilt the agent) well past the window:
        # before the fix it re-armed a new window and stayed blocked forever.
        second = self._bound(db)
        assert second._anti_thrash_recovery_deadline == (
            base + first._ANTI_THRASH_RECOVERY_SECONDS
        )
        with patch(
            "agent.context_compressor.time.time",
            return_value=base + first._ANTI_THRASH_RECOVERY_SECONDS + 1,
        ):
            assert second.should_compress(second.threshold_tokens + 1) is True
        assert db.get_compression_ineffective_count("sess-1") == 1
        assert db.get_compression_recovery_deadline("sess-1") == 0.0

    def test_fresh_compressor_inside_window_stays_blocked(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        db.set_compression_ineffective_count("sess-1", 2)
        base = 5000.0
        first = self._bound(db)
        with patch("agent.context_compressor.time.time", return_value=base):
            assert first.should_compress(first.threshold_tokens + 1) is False
        second = self._bound(db)
        with patch("agent.context_compressor.time.time", return_value=base + 10):
            assert second.should_compress(second.threshold_tokens + 1) is False
        assert db.get_compression_ineffective_count("sess-1") == 2

    def test_backward_clock_jump_is_bounded_to_one_window(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        db.set_compression_ineffective_count("sess-1", 2)
        window = ContextCompressor._ANTI_THRASH_RECOVERY_SECONDS
        db.set_compression_recovery_deadline("sess-1", 1_000_000.0)
        cc = self._bound(db)
        # Wall clock now far BEFORE the stored deadline (clock stepped back).
        with patch("agent.context_compressor.time.time", return_value=100.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert db.get_compression_recovery_deadline("sess-1") == 100.0 + window

    def test_clearing_the_guard_disarms_the_durable_deadline(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        db.set_compression_ineffective_count("sess-1", 2)
        cc = self._bound(db)
        with patch("agent.context_compressor.time.time", return_value=5000.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert db.get_compression_recovery_deadline("sess-1") > 0.0
        cc._record_ineffective_compression_verdict(0)
        with patch("agent.context_compressor.time.time", return_value=5001.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert db.get_compression_recovery_deadline("sess-1") == 0.0

    def test_session_db_round_trip(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        assert db.get_compression_recovery_deadline("sess-1") == 0.0
        db.set_compression_recovery_deadline("sess-1", 1234.5)
        assert db.get_compression_recovery_deadline("sess-1") == 1234.5
        db.set_compression_recovery_deadline("sess-1", 0.0)
        assert db.get_compression_recovery_deadline("sess-1") == 0.0
        assert db.get_compression_recovery_deadline("missing") == 0.0
