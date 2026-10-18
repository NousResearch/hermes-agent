"""Anti-thrash recovery: the tripped guard must not be permanent (#14694).

When two consecutive compactions each fail to clear the threshold, the
anti-thrashing breaker blocks automatic compaction. Before this fix the block
was permanent for the life of the session: nothing ever decremented
``_ineffective_compression_count``
while blocked, so a session whose middle region was briefly too small to
compact never auto-compacted again — it grew unbounded until the provider's
hard context limit, and only ``/new`` or ``/reset`` recovered it.

The recovery contract pinned here:

* After ``_ANTI_THRASH_RECOVERY_SECONDS`` of continuous block, the gate
  grants exactly ONE token-fenced probation probe. Durable counters stay
  tripped to fence siblings; only the winning compressor mirrors one strike.
* An ineffective probe re-trips the guard on the very next verdict, and the
  next recovery waits a FULL fresh window (no immediate re-probe loop).
* An effective probe (or any fitting real-usage reading) fully clears the
  count through the existing ``update_from_response`` path.
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

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import CompressionCommitFence, compress_context
from hermes_state import SessionDB
from tests.agent.test_compression_attempt_lifecycle import _build_agent


def test_upstream_deadline_and_fenced_probe_share_one_state(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="cross-reader", source="cli")
    db.set_compression_breaker_state(
        "cross-reader", ineffective_count=2, fallback_streak=0, recovery_at=1900)
    assert db.get_compression_recovery_deadline("cross-reader") == 1900
    db.set_compression_recovery_deadline("cross-reader", 2400)
    assert db.get_compression_breaker_state("cross-reader")["recovery_at"] == 2400
    assert not db.claim_compression_recovery_probe(
        "cross-reader", now=2399, recovery_seconds=900)["claimed"]
    claim = db.claim_compression_recovery_probe(
        "cross-reader", now=2401, recovery_seconds=900)
    assert claim["claimed"]
    assert db.get_compression_recovery_deadline("cross-reader") == 0
    assert db.release_compression_recovery_probe(
        "cross-reader", probe_token=claim["probe_token"], now=2402)
    assert db.get_compression_recovery_deadline("cross-reader") == 2402
    db.close()


def test_pre_column_recovery_state_migrates_without_rearming(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="legacy-recovery", source="cli")
    db.set_compression_ineffective_count("legacy-recovery", 2)
    db.patch_session_model_config(
        "legacy-recovery", {"_compression_anti_thrash_recovery_at": 1900})
    assert db.get_compression_recovery_deadline("legacy-recovery") == 1900
    claim = db.claim_compression_recovery_probe(
        "legacy-recovery", now=1901, recovery_seconds=900)
    assert claim["claimed"]
    assert db.get_compression_recovery_deadline("legacy-recovery") == 0
    assert db.get_session_model_config_value(
        "legacy-recovery", "_compression_anti_thrash_recovery_at") is None
    assert not db.claim_compression_recovery_probe(
        "legacy-recovery", now=1902, recovery_seconds=900)["claimed"]
    db.close()


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

    def test_recovery_window_is_fifteen_minutes(self):
        assert _compressor()._ANTI_THRASH_RECOVERY_SECONDS == 900.0


    def test_effective_probe_clears_the_guard_completely(self):
        cc = _compressor()
        base = 1000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(cc)
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

    def test_fallback_boundaries_wait_for_real_effectiveness_verdict(self):
        cc = _compressor()
        for prompt_tokens in (4_000, 5_000):
            cc.record_completed_compaction(used_fallback=True)
            cc.update_from_response({"prompt_tokens": prompt_tokens})
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert cc._fallback_compression_streak == 2
        assert cc._ineffective_compression_count == 0

        for _ in range(2):
            cc.record_completed_compaction(used_fallback=True)
            cc.update_from_response({"prompt_tokens": cc.threshold_tokens + 1})
        assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert cc._ineffective_compression_count == 2




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
        # Counters remain tripped durably while the single winner owns the
        # probe; only its in-process mirror falls to one strike.
        state = db.get_compression_breaker_state("sess-1")
        assert state["ineffective_count"] == 2
        assert state["probe_token"]

    def test_session_reset_disarms_the_recovery_clock_durably(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        cc = _compressor()
        cc.bind_session_state(session_db=db, session_id="sess-1")
        base = 1000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(cc)
            assert cc.should_compress(cc.threshold_tokens + 1) is False
            cc.record_completed_compaction(used_fallback=True)
            cc.record_completed_compaction(used_fallback=True)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert cc._anti_thrash_probe_until > 0.0
        assert cc._anti_thrash_probe_token
        cc.on_session_reset()
        assert cc._anti_thrash_recovery_deadline == 0.0
        assert cc._ineffective_compression_count == 0
        assert db.get_compression_ineffective_count("sess-1") == 0
        state = db.get_compression_breaker_state("sess-1")
        assert state["fallback_streak"] == 0
        assert state["recovery_at"] == 0.0
        assert state["probe_until"] == 0.0
        assert state["probe_token"] == ""


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
        # Deadline is durable in the atomic breaker tuple.
        assert db.get_compression_breaker_state("sess-1")["recovery_at"] == base + first._ANTI_THRASH_RECOVERY_SECONDS
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
        state = db.get_compression_breaker_state("sess-1")
        assert state["ineffective_count"] == 2
        assert state["probe_until"] > 0.0
        sibling = _compressor()
        with patch(
            "agent.context_compressor.time.time",
            return_value=base + first._ANTI_THRASH_RECOVERY_SECONDS + 2,
        ):
            sibling.bind_session_state(session_db=db, session_id="sess-1")
            assert sibling.should_compress(sibling.threshold_tokens + 1) is False

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
        window = ContextCompressor._ANTI_THRASH_RECOVERY_SECONDS
        db.set_compression_breaker_state(
            "sess-1", ineffective_count=2, fallback_streak=0, recovery_at=1_000_000.0,
        )
        cc = self._bound(db)
        # Wall clock now far BEFORE the stored deadline (clock stepped back).
        with patch("agent.context_compressor.time.time", return_value=100.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert db.get_compression_breaker_state("sess-1")["recovery_at"] == 100.0 + window
        with patch("agent.context_compressor.time.time", return_value=100.0 + window + 1):
            assert cc.should_compress(cc.threshold_tokens + 1) is True

    def test_clearing_the_guard_disarms_the_durable_deadline(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        db.set_compression_ineffective_count("sess-1", 2)
        cc = self._bound(db)
        with patch("agent.context_compressor.time.time", return_value=5000.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is False
        assert db.get_compression_breaker_state("sess-1")["recovery_at"] > 0.0
        cc._record_ineffective_compression_verdict(0)
        with patch("agent.context_compressor.time.time", return_value=5001.0):
            assert cc.should_compress(cc.threshold_tokens + 1) is True
        assert db.get_compression_breaker_state("sess-1")["recovery_at"] == 0.0

    def test_session_db_round_trip(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        assert db.get_compression_recovery_deadline("sess-1") == 0.0
        db.set_compression_recovery_deadline("sess-1", 1234.5)
        assert db.get_compression_recovery_deadline("sess-1") == 1234.5
        db.set_compression_recovery_deadline("sess-1", 0.0)
        assert db.get_compression_recovery_deadline("sess-1") == 0.0
        assert db.get_compression_recovery_deadline("missing") == 0.0


class TestAtomicRecoveryProbe:
    def test_fresh_agents_claim_exactly_one_probe(self, tmp_path):
        db_path = tmp_path / "state.db"
        setup = SessionDB(db_path=db_path)
        setup.create_session(session_id="sess-1", source="telegram")
        owner = _compressor()
        owner.bind_session_state(setup, "sess-1")
        base = 10_000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(owner)

        barrier = Barrier(2)

        def attempt() -> bool:
            db = SessionDB(db_path=db_path)
            compressor = _compressor()
            compressor.bind_session_state(db, "sess-1")
            barrier.wait()
            return compressor.should_compress(compressor.threshold_tokens + 1)

        with patch("agent.context_compressor.time.time", return_value=base + 901.0), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))

        assert sorted(results) == [False, True]
        state = setup.get_compression_breaker_state("sess-1")
        assert state["ineffective_count"] == 2
        assert state["probe_until"] == base + 1801.0
        assert state["probe_token"]

    def test_abandoned_probe_can_be_reclaimed_after_one_full_window(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        base = 20_000.0
        first = _compressor()
        first.bind_session_state(session_db=db, session_id="sess-1")
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(first)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert first.should_compress(first.threshold_tokens + 1) is True

        replacement = _compressor()
        with patch("agent.context_compressor.time.time", return_value=base + 1802.0):
            replacement.bind_session_state(session_db=db, session_id="sess-1")
            assert replacement.should_compress(replacement.threshold_tokens + 1) is True

    def test_reclaimed_probe_fences_the_stale_owner(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        base = 40_000.0
        old_owner = _compressor()
        old_owner.bind_session_state(session_db=db, session_id="sess-1")
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(old_owner)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert old_owner.should_compress(old_owner.threshold_tokens + 1) is True
        old_token = old_owner._anti_thrash_probe_token

        new_owner = _compressor()
        with patch("agent.context_compressor.time.time", return_value=base + 1802.0):
            new_owner.bind_session_state(session_db=db, session_id="sess-1")
            assert new_owner.should_compress(new_owner.threshold_tokens + 1) is True
        assert new_owner._anti_thrash_probe_token != old_token

        with patch("agent.context_compressor.time.time", return_value=base + 1803.0):
            assert old_owner.should_compress(old_owner.threshold_tokens + 1) is False
            assert new_owner.should_compress(new_owner.threshold_tokens + 1) is True

    def test_stale_release_cannot_clear_replacement_probe(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        old = _compressor()
        old.bind_session_state(db, "sess-1")
        base = 20_000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(old)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert old.should_compress(old.threshold_tokens + 1)
        stale_token = old._anti_thrash_probe_token

        replacement = _compressor()
        replacement.bind_session_state(db, "sess-1")
        with patch("agent.context_compressor.time.time", return_value=base + 1802.0):
            assert replacement.should_compress(replacement.threshold_tokens + 1)
        replacement_token = replacement._anti_thrash_probe_token

        assert not db.release_compression_recovery_probe("sess-1", probe_token=stale_token, now=base + 1803.0)
        state = db.get_compression_breaker_state("sess-1")
        assert state["probe_token"] == replacement_token
        assert state["probe_until"] == base + 2702.0

    def test_committed_fence_keeps_the_probe_claim(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        owner = _compressor()
        owner.bind_session_state(session_db=db, session_id="sess-1")
        base = 70_000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(owner)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert owner.should_compress(owner.threshold_tokens + 1) is True
        fence = CompressionCommitFence()
        fence.register_cancelled_recovery_probe_release(owner.release_anti_thrash_recovery_probe)
        assert fence.begin_commit() is True
        fence.release_cancelled_compression_lock()
        assert db.get_compression_breaker_state("sess-1")["probe_token"]
        fence.finish_commit()

    def test_forward_clock_jump_grants_one_bounded_probe(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        owner = _compressor()
        owner.bind_session_state(session_db=db, session_id="sess-1")
        with patch("agent.context_compressor.time.time", return_value=1_000.0):
            _trip(owner)
        with patch("agent.context_compressor.time.time", return_value=20_000.0):
            assert owner.should_compress(owner.threshold_tokens + 1) is True

    def test_cancelled_fence_releases_matching_probe_for_immediate_reclaim(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="telegram")
        owner = _compressor()
        owner.bind_session_state(db, "sess-1")
        base = 30_000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(owner)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert owner.should_compress(owner.threshold_tokens + 1)
            fence = CompressionCommitFence()
            fence.register_cancelled_recovery_probe_release(owner.release_anti_thrash_recovery_probe)
            assert fence.try_cancel_before_commit() is True
            fence.release_cancelled_compression_lock()

        state = db.get_compression_breaker_state("sess-1")
        assert state["probe_until"] == 0.0
        assert state["probe_token"] == ""
        assert state["recovery_at"] == base + 901.0

        successor = _compressor()
        successor.bind_session_state(db, "sess-1")
        with patch("agent.context_compressor.time.time", return_value=base + 902.0):
            assert successor.should_compress(successor.threshold_tokens + 1)

    def test_compress_context_publishes_probe_release_before_provider_work(self, tmp_path):
        db, agent = _build_agent(tmp_path, "probe-cancel")
        base = 35_000.0
        db.set_compression_breaker_state(
            "probe-cancel", ineffective_count=2, fallback_streak=0, recovery_at=base,
        )
        agent.context_compressor.bind_session_state(db, "probe-cancel")
        agent.context_compressor.protect_first_n = 3
        agent.context_compressor.protect_last_n = 5
        entered, resume = Event(), Event()

        def held_summary(*_args, **_kwargs):
            entered.set()
            assert resume.wait(3.0)
            return "summary"

        agent.context_compressor._generate_summary = held_summary
        fence = CompressionCommitFence()
        messages = [
            {"role": "user", "content": f"historical turn {index} with durable detail " * 50}
            for index in range(40)
        ]
        with patch("agent.context_compressor.time.time", return_value=base + 901.0), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(compress_context, agent, messages, "sys", approx_tokens=500_000, commit_fence=fence)
            assert entered.wait(3.0)
            assert fence._cancelled_probe_release is not None
            assert fence.try_cancel_before_commit() is True
            fence.release_cancelled_compression_lock()
            state = db.get_compression_breaker_state("probe-cancel")
            assert state["probe_until"] == 0.0
            assert state["probe_token"] == ""
            resume.set()
            returned, _ = future.result(timeout=5.0)
        assert returned == messages
        fresh = _compressor()
        with patch("agent.context_compressor.time.time", return_value=base + 902.0):
            fresh.bind_session_state(db, "probe-cancel")
            assert fresh.should_compress(fresh.threshold_tokens + 1) is True

    def test_rotation_carries_the_original_recovery_deadline(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli")
        owner = _compressor()
        owner.bind_session_state(session_db=db, session_id="parent")
        with patch("agent.context_compressor.time.time", return_value=7_000.0):
            _trip(owner)
        parent_deadline = owner._anti_thrash_recovery_deadline

        owner.on_session_start(
            "child", boundary_reason="compression", old_session_id="parent", session_db=db,
        )

        assert owner._anti_thrash_recovery_deadline == parent_deadline
        assert db.get_compression_breaker_state("child")["recovery_at"] == parent_deadline

    def test_rotation_copies_claim_but_only_winner_keeps_local_ownership(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="parent", source="telegram")
        db.create_session(session_id="child", source="telegram")
        owner = _compressor()
        owner.bind_session_state(db, "parent")
        base = 40_000.0
        with patch("agent.context_compressor.time.time", return_value=base):
            _trip(owner)
        with patch("agent.context_compressor.time.time", return_value=base + 901.0):
            assert owner.should_compress(owner.threshold_tokens + 1)
            owner.on_session_start("child", boundary_reason="compression", old_session_id="parent", session_db=db)

        state = db.get_compression_breaker_state("child")
        assert state["ineffective_count"] == 2
        assert state["probe_token"] == owner._anti_thrash_probe_token
        assert owner._owns_anti_thrash_probe
        sibling = _compressor()
        sibling.bind_session_state(db, "child")
        with patch("agent.context_compressor.time.time", return_value=base + 902.0):
            assert not sibling.should_compress(sibling.threshold_tokens + 1)


def test_recovery_owner_survives_guard_refresh_and_loses_a_replaced_claim(tmp_path):
    from agent.conversation_compression import _refresh_persisted_compression_guards

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="refresh-fixture", source="cli")
    owner = _compressor()
    owner.bind_session_state(db, "refresh-fixture")
    with patch("agent.context_compressor.time.time", return_value=1_000.0):
        _trip(owner)
    with patch("agent.context_compressor.time.time", return_value=1_901.0):
        assert owner.should_compress(owner.threshold_tokens + 1)
        sibling = _compressor()
        sibling.bind_session_state(db, "refresh-fixture")
        for include_cooldown in (True, False):
            _refresh_persisted_compression_guards(owner, include_cooldown=include_cooldown)
            assert not owner._automatic_compression_blocked()
            assert not sibling.should_compress(sibling.threshold_tokens + 1)

        state = db.get_compression_breaker_state("refresh-fixture")
        state["probe_token"] = "replacement-owner"
        db.set_compression_breaker_state("refresh-fixture", **state)
        _refresh_persisted_compression_guards(owner)
        assert owner._automatic_compression_blocked()
