"""Anti-thrash state must survive process restarts (#54923).

The guard in ``_automatic_compression_blocked_locally()`` trips only after two
consecutive compactions that fail to bring the real prompt under the
threshold. Historically ``_ineffective_compression_count`` was in-memory only:
a fresh compressor bound to a resumed (already-compacted) session started at
``compression_count == 0`` with a disarmed guard, so a near-threshold session
could legally re-compact once per process restart, forever — the exact
residual @lanyusea identified in #54923.

The counter now round-trips the durable session-state channel exactly like
``compression_failure_cooldown_until`` (#54465) and the fallback streak
(af7dceaf7):

* every verdict (strike or clear) writes through to the session row,
* ``bind_session_state()`` loads the persisted value, so a fresh compressor
  on a resumed session inherits an armed (1) or tripped (2) guard,
* the reset semantics are unchanged — a real provider reading below the
  threshold still clears the counter (update_from_response), and that clear
  is durable too.
"""
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor(db: SessionDB | None = None, session_id: str = "") -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        cc = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    if db is not None:
        cc.bind_session_state(db, session_id)
    return cc


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


class TestCounterRoundTripsBindSessionState:
    def test_fresh_compressor_inherits_tripped_guard_after_restart(self, tmp_path):
        """A restart must not disarm a tripped anti-thrash breaker."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        first = _compressor(db, "s1")
        # Two compactions that failed to clear the threshold — judged on the
        # provider's real prompt counts, exactly as conversation_loop drives it.
        for _ in range(2):
            first._verify_compaction_cleared_threshold = True
            first.update_from_response({"prompt_tokens": first.threshold_tokens + 1})
        assert first._ineffective_compression_count == 2
        assert first.should_compress(10**9) is False

        # Process restart: a brand-new compressor binds the same session.
        second = _compressor(db, "s1")
        assert second.compression_count == 0  # the #54923 precondition
        assert second._ineffective_compression_count == 2
        assert second.should_compress(10**9) is False, (
            "a fresh compressor on a resumed session must inherit the "
            "tripped anti-thrash guard instead of re-compacting"
        )

    def test_fresh_compressor_inherits_armed_single_strike(self, tmp_path):
        """One strike before the restart still counts toward the trip."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        first = _compressor(db, "s1")
        first._verify_compaction_cleared_threshold = True
        first.update_from_response({"prompt_tokens": first.threshold_tokens + 1})
        assert first._ineffective_compression_count == 1

        second = _compressor(db, "s1")
        assert second._ineffective_compression_count == 1
        # One inherited strike does not block yet...
        assert second.should_compress(10**9) is True
        # ...but the next ineffective pass trips the guard cross-process.
        second._verify_compaction_cleared_threshold = True
        second.update_from_response({"prompt_tokens": second.threshold_tokens + 1})
        assert second._ineffective_compression_count == 2
        assert second.should_compress(10**9) is False

    def test_rebind_to_other_session_does_not_leak_counter(self, tmp_path):
        """The counter is per-session: switching sessions must not carry it."""
        db = _db(tmp_path)
        db.create_session("hot", source="cli")
        db.create_session("cold", source="cli")
        db.set_compression_ineffective_count("hot", 2)

        cc = _compressor(db, "hot")
        assert cc._ineffective_compression_count == 2

        cc.bind_session_state(db, "cold")
        assert cc._ineffective_compression_count == 0

    def test_unbound_compressor_keeps_in_memory_behavior(self):
        """No session DB bound (plugins/tests): everything still works."""
        cc = _compressor()
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert cc._ineffective_compression_count == 1
        cc.update_from_response({"prompt_tokens": 1})
        assert cc._ineffective_compression_count == 0


class TestResetSemanticsPreserved:
    def test_real_dip_below_threshold_clears_counter_durably(self, tmp_path):
        """The L1466-1474 contract survives: any real provider reading below
        the threshold clears the latch — and now clears the durable copy, so
        a restart cannot resurrect voided strikes."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        cc = _compressor(db, "s1")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert cc._ineffective_compression_count == 1
        assert db.get_compression_ineffective_count("s1") == 1

        # An ordinary fitting response (not post-compaction) clears the latch.
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 1})
        assert cc._ineffective_compression_count == 0
        assert db.get_compression_ineffective_count("s1") == 0

        # And a restart sees the cleared state.
        fresh = _compressor(db, "s1")
        assert fresh._ineffective_compression_count == 0
        assert fresh.should_compress(10**9) is True

    def test_post_compaction_clearing_reading_resets_durably(self, tmp_path):
        """The post-compaction success verdict (real tokens under threshold)
        also zeroes the durable strike count."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        cc = _compressor(db, "s1")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert db.get_compression_ineffective_count("s1") == 1

        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens - 1})
        assert cc._ineffective_compression_count == 0
        assert db.get_compression_ineffective_count("s1") == 0

    def test_update_model_reset_writes_through(self, tmp_path):
        """update_model() voids strikes judged against the old threshold; the
        durable copy must not resurrect them on the next restart."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        cc = _compressor(db, "s1")
        db.set_compression_ineffective_count("s1", 2)
        cc._ineffective_compression_count = 2

        cc.update_model("other/model", 100_000)

        assert cc._ineffective_compression_count == 0
        assert db.get_compression_ineffective_count("s1") == 0


class TestStrikesPersistFromEveryVerdictSite:
    def test_no_op_compaction_branches_write_through(self, tmp_path):
        """The insufficient-messages no-op branch records its strike durably."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")

        cc = _compressor(db, "s1")
        # 3 tiny messages < minimum window → the #40803 no-op branch.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = cc.compress(msgs, current_tokens=10**9)
        assert out == msgs
        assert cc._ineffective_compression_count == 1
        assert db.get_compression_ineffective_count("s1") == 1

    def test_persist_failure_is_swallowed_and_memory_still_advances(self, tmp_path):
        """A DB write failure must not break the in-memory guard."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        cc = _compressor(db, "s1")

        with patch.object(
            db,
            "set_compression_ineffective_count",
            side_effect=Exception("disk full"),
        ):
            cc._verify_compaction_cleared_threshold = True
            cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})

        assert cc._ineffective_compression_count == 1
        assert db.get_compression_ineffective_count("s1") == 0


class TestCompressionBoundaryCarry:
    def test_rotation_boundary_carries_counter_onto_child_row(self, tmp_path):
        """Session-id rotation must not launder an armed guard through the
        fresh child row (same carry contract as the fallback streak)."""
        db = _db(tmp_path)
        db.create_session("parent", source="cli")
        cc = _compressor(db, "parent")
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert db.get_compression_ineffective_count("parent") == 1

        db.create_session("child", source="cli", parent_session_id="parent")
        cc.on_session_start(
            "child",
            boundary_reason="compression",
            old_session_id="parent",
            session_db=db,
        )

        assert cc._session_id == "child"
        assert cc._ineffective_compression_count == 1
        # Persisted onto the child row so a restart right after rotation
        # still inherits the armed guard.
        assert db.get_compression_ineffective_count("child") == 1
