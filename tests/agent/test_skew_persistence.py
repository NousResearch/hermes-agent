"""Skew-calibration persistence across process restarts (2026-07-10 incident).

The P2 skew history (real/rough ratios that calibrate the preflight
compression trigger) lived only in process memory on the singleton engine.
A gateway restart wiped it, so the first post-restart preflight ran with
skew=1.0 (raw rough): on a large session whose rough estimate over-counts
(live numbers: raw ~767k vs real ~476k on a 1M window, threshold 750k) that
FALSE-FIRED compression at ~48% real usage.

Fix: persist the last-k ratios to the session row (sessions.compression_
skew_history, JSON) on every fresh pair; seed an EMPTY in-memory history
from the persisted row when ``bind_session_state`` binds the session; clear
at real session boundaries (reset/end) alongside ``reset_skew_calibration``.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _isolated_skew_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


class FakeSessionDB:
    """Duck-typed stand-in for HermesState (record/get/clear round-trip)."""

    def __init__(self):
        self.rows: dict[str, str] = {}

    def record_compression_skew_history(self, session_id, skew_history):
        self.rows[session_id] = json.dumps(skew_history)

    def get_compression_skew_history(self, session_id):
        raw = self.rows.get(session_id)
        if not raw:
            return []
        return [float(v) for v in json.loads(raw) if 0.0 < float(v) <= 1.0]

    def clear_compression_skew_history(self, session_id):
        self.rows.pop(session_id, None)

    # bind_session_state calls this via get_active_compression_failure_cooldown
    def get_compression_failure_cooldown(self, session_id):
        return None


def make_compressor(session_db=None, session_id=""):
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=1_000_000
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.75,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    if session_db is not None:
        c.bind_session_state(session_db=session_db, session_id=session_id)
    return c


def record_pair(c, rough, real):
    c.note_rough_sent(rough)
    c.record_skew_from_real(real)


class TestIncidentRegression:
    def test_restart_seeds_skew_and_does_not_false_fire(self):
        """THE incident: 767k raw / 476k real / 750k threshold.

        Pre-fix: fresh process → skew=1.0 → calibrated == raw 767k ≥ 750k
        → false compression at 48% real usage. Post-fix: the persisted
        history seeds the new process; calibrated ≈ 476k < 750k → no fire.
        """
        db = FakeSessionDB()
        sid = "sess-incident"

        # Process 1: session learns its real skew (real/rough ≈ 0.62).
        c1 = make_compressor(db, sid)
        record_pair(c1, 767_000, 476_000)
        assert db.rows.get(sid), "history must persist on the fresh pair"

        # Process 2 (simulated restart: brand-new compressor object).
        c2 = make_compressor(db, sid)
        preflight_raw = 767_000
        # Seeded skew must calibrate the raw estimate below threshold.
        assert c2._current_skew() < 1.0
        assert c2.calibrated_tokens(preflight_raw) < c2.threshold_tokens
        assert c2.should_compress_calibrated(preflight_raw) is False

    def test_prefix_behavior_reproduced_without_seed(self):
        """Sanity: WITHOUT the persisted seed the false-fire happens (documents
        the pre-fix behavior the seed prevents)."""
        c = make_compressor()  # no session binding, empty history
        assert c._current_skew() == 1.0
        assert c.should_compress_calibrated(767_000) is True


class TestPersistRoundtrip:
    def test_each_fresh_pair_persists(self):
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 75_000)
        record_pair(c, 200_000, 160_000)
        stored = json.loads(db.rows["s1"])
        assert len(stored) == 2
        assert stored[0] == pytest.approx(0.75)
        assert stored[1] == pytest.approx(0.8)

    def test_persist_capped_at_history_window(self):
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        for i in range(8):
            record_pair(c, 100_000, 60_000 + i * 1000)
        stored = json.loads(db.rows["s1"])
        assert len(stored) == c._SKEW_HISTORY

    def test_no_binding_no_crash(self):
        c = make_compressor()  # unbound
        record_pair(c, 100_000, 75_000)  # must not raise
        assert c._current_skew() == pytest.approx(0.75)

    def test_persist_failure_never_raises(self):
        db = FakeSessionDB()
        db.record_compression_skew_history = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("db gone")
        )
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 75_000)  # must not raise
        assert c._current_skew() == pytest.approx(0.75)


class TestSeedRules:
    def test_seed_only_fills_empty_history(self):
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 75_000)  # live history now non-empty
        c.seed_skew_calibration([0.9, 0.95])  # must be ignored
        assert c._current_skew() == pytest.approx(0.75)

    def test_seed_rejects_garbage(self):
        c = make_compressor()
        c.seed_skew_calibration([0.0, -1, 1.5, "nan-ish", None, 2.0])
        assert c._current_skew() == 1.0  # nothing sane to seed

    def test_seed_accepts_partial_garbage(self):
        c = make_compressor()
        c.seed_skew_calibration([1.5, 0.8, None, 0.9])
        assert c._current_skew() == pytest.approx(0.85)

    def test_corrupt_persisted_json_ignored(self):
        db = FakeSessionDB()
        db.rows["s1"] = "not json"

        def strict_get(session_id):
            raw = db.rows.get(session_id)
            try:
                return [float(v) for v in json.loads(raw)]
            except (TypeError, ValueError):
                return []

        db.get_compression_skew_history = strict_get
        c = make_compressor(db, "s1")  # must not raise
        assert c._current_skew() == 1.0


class TestBoundaryClears:
    def test_session_reset_clears_persisted_history(self):
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 75_000)
        assert "s1" in db.rows
        c.on_session_reset()
        assert "s1" not in db.rows
        assert c._current_skew() == 1.0

    def test_session_end_keeps_persisted_history(self):
        """GATEWAY SHUTDOWN calls on_session_end on cached agents
        (shutdown_memory_provider) — NOT a real conversation boundary; the
        same session resumes right after. Clearing the persisted row there
        defeats the restart seed entirely (2026-07-10 restart-B live E2E:
        row gone at boot, skew back to 1.0). In-memory resets; row survives."""
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 75_000)
        c.on_session_end("s1", [])
        assert "s1" in db.rows, "persisted row must survive shutdown-path session_end"
        assert c._current_skew() == 1.0  # in-memory still resets

    def test_shutdown_then_restart_seeds_again(self):
        """Full incident sequence: pair recorded -> gateway shutdown fires
        on_session_end -> new process binds same session -> seed works."""
        db = FakeSessionDB()
        c1 = make_compressor(db, "s1")
        record_pair(c1, 767_000, 476_000)
        c1.on_session_end("s1", [])  # gateway shutdown path
        c2 = make_compressor(db, "s1")  # restart
        assert c2._current_skew() < 1.0
        assert c2.should_compress_calibrated(767_000) is False

    def test_greptile_111_cross_session_isolation_preserved(self):
        """A fresh session (reset) must NOT inherit the prior session's skew —
        the persisted seed must not defeat the Greptile #111 boundary reset."""
        db = FakeSessionDB()
        c = make_compressor(db, "s1")
        record_pair(c, 100_000, 55_000)
        c.on_session_reset()
        # rebind to a NEW session id — nothing persisted for it
        c.bind_session_state(session_db=db, session_id="s2")
        assert c._current_skew() == 1.0


class TestRealHermesState:
    """E2E against the REAL HermesState schema + round-trip (no fakes)."""

    def test_real_state_roundtrip_and_restart_seed(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-real"
        db.create_session(sid, source="test")

        c1 = make_compressor(db, sid)
        record_pair(c1, 767_000, 476_000)
        assert db.get_compression_skew_history(sid), "real DB row must persist"

        # restart: brand-new compressor seeded from the real DB
        c2 = make_compressor(db, sid)
        assert c2._current_skew() < 1.0
        assert c2.should_compress_calibrated(767_000) is False

        # boundary clear round-trips on the real DB too
        c2.on_session_reset()
        assert db.get_compression_skew_history(sid) == []
