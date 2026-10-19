"""Private-database VACUUM policy, retry accounting and retention tests."""

import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.usefixtures("private_db_no_foreign_holders")


class TestVacuum:

    def test_auto_maintenance_records_successful_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]
        assert db.get_meta("last_vacuum") is not None

    def test_auto_maintenance_skips_recent_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
        db.set_meta("last_vacuum", str(time.time()))
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_retries_after_vacuum_interval(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
        db.set_meta("last_vacuum", str(time.time() - 31 * 86400))
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_auto_maintenance_retries_after_failed_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
        vacuum_calls = []

        def fail_first_vacuum():
            vacuum_calls.append(True)
            if len(vacuum_calls) == 1:
                raise RuntimeError("vacuum failed")

        monkeypatch.setattr(db, "vacuum", fail_first_vacuum)

        first = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert first["vacuumed"] is False
        assert db.get_meta("last_vacuum") is None

        second = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert second["vacuumed"] is True
        assert vacuum_calls == [True, True]
        assert db.get_meta("last_vacuum") is not None

    # ── freelist-ratio gate (#54189) ─────────────────────────────────────
    def test_auto_maintenance_skips_vacuum_below_freelist_ratio(self, db, monkeypatch):
        """A prune that frees few pages on a dense DB must NOT trigger VACUUM."""
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.05)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result["freelist_ratio"] == 0.05
        assert vacuum_calls == []
        assert db.get_meta("last_vacuum") is None
        # The prune itself still counts as a maintenance run.
        assert db.get_meta("last_auto_prune") is not None

    def test_auto_maintenance_vacuums_above_freelist_ratio(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.40)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] == 0.40
        assert vacuum_calls == [True]

    def test_auto_maintenance_freelist_ratio_exactly_at_threshold_skips(self, db, monkeypatch):
        """Gate is strictly greater-than: 25.0% reclaimable does not VACUUM."""
        from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: AUTO_VACUUM_MIN_FREELIST_RATIO)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_unknown_freelist_ratio_falls_back_to_time_throttle(self, db, monkeypatch):
        """If the pragmas cannot be read, don't silently disable VACUUM forever."""
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: None)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] is None
        assert vacuum_calls == [True]

    def test_auto_maintenance_ratio_gate_threshold_is_overridable(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.10)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0, min_vacuum_freelist_ratio=0.05
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_freelist_ratio_reads_real_pragmas(self, db):
        """Real-DB check: freeing most of the file pushes the ratio past the gate."""
        from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

        db.create_session(session_id="keep", source="cli")
        db.append_message(session_id="keep", role="user", content="hi")
        for i in range(6):
            sid = f"bulk{i}"
            db.create_session(session_id=sid, source="cli")
            for _ in range(20):
                db.append_message(session_id=sid, role="assistant", content="z" * 4000)
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dense = db._freelist_ratio()
        assert dense is not None and dense < AUTO_VACUUM_MIN_FREELIST_RATIO

        for i in range(6):
            db.delete_session(f"bulk{i}")
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        sparse = db._freelist_ratio()
        assert sparse is not None and sparse > AUTO_VACUUM_MIN_FREELIST_RATIO

    def test_wal_size_limit_is_bounded(self, db):
        """journal_size_limit must be a finite bound, not SQLite's -1 default.

        Contract, not a snapshot: assert the limit is positive (so the WAL is
        truncated back at checkpoints) rather than pinning the exact byte
        count, which is a tunable.
        """
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            pytest.skip("WAL unavailable on this filesystem")
        limit = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        assert limit > 0, "unbounded WAL: state.db-wal never returns disk to the OS"

    def test_vacuum_leaves_wal_truncated(self, db, tmp_path):
        """VACUUM must not strand a giant WAL beside the database.

        VACUUM rewrites every page through the write-ahead log. Without a
        checkpoint *after* it, a 3 GB database leaves a 3 GB state.db-wal
        behind — `sessions optimize` then consumes far more disk than it
        frees, which is the opposite of its purpose.
        """
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            pytest.skip("WAL unavailable on this filesystem")

        db.create_session(session_id="s1", source="cli")
        for i in range(500):
            db.append_message(
                session_id="s1", role="user", content=f"padding message {i} " * 20
            )
        db.vacuum()

        wal = Path(str(db.db_path) + "-wal")
        if wal.exists():
            limit = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
            assert wal.stat().st_size <= max(limit, 0) or wal.stat().st_size == 0, (
                f"WAL left at {wal.stat().st_size} bytes after VACUUM"
            )



class TestAutoMaintenance:
    def _make_old_ended(self, db, sid: str, days_old: int = 100):
        """Create a session that is ended and was started `days_old` days ago."""
        db.create_session(session_id=sid, source="cli")
        db.end_session(sid, end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - days_old * 86400, sid),
        )
        db._conn.commit()

    def test_first_run_prunes_and_skips_vacuum_when_little_reclaimable(self, db):
        """Pruning two empty sessions frees almost nothing → prune yes, VACUUM no."""
        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        db.create_session(session_id="new", source="cli")  # active, must survive

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 2
        assert result["vacuumed"] is False  # freelist ratio gate (#54189)
        assert result["freelist_ratio"] is not None
        assert result["freelist_ratio"] <= 0.25
        assert result.get("error") is None
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None
        assert db.get_session("new") is not None

    def test_first_run_prunes_and_vacuums_when_mostly_reclaimable(self, db):
        """Pruning the bulk of the file's pages crosses the 25% gate → VACUUM runs."""
        db.create_session(session_id="new", source="cli")  # active, must survive
        db.append_message(session_id="new", role="user", content="hi")
        for i in range(6):
            sid = f"old{i}"
            self._make_old_ended(db, sid, days_old=100)
            for _ in range(20):
                db.append_message(session_id=sid, role="assistant", content="z" * 4000)
            # Keep the row aged: append_message bumps activity, prune ages by
            # latest message, so push the message timestamps back too.
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (time.time() - 100 * 86400, sid),
            )
        db._conn.commit()

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 6
        assert result["freelist_ratio"] > 0.25
        assert result["vacuumed"] is True
        assert result.get("error") is None
        assert db.get_session("new") is not None
        assert db.get_meta("last_vacuum") is not None

    def test_second_call_within_interval_skips(self, db):
        self._make_old_ended(db, "old", days_old=100)
        first = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=24
        )
        assert first["skipped"] is False
        assert first["pruned"] == 1

        # Create another prunable session; a second call within
        # min_interval_hours should still skip without touching it.
        self._make_old_ended(db, "old2", days_old=100)
        second = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=24
        )
        assert second["skipped"] is True
        assert second["pruned"] == 0
        assert db.get_session("old2") is not None  # untouched

    def test_auto_prune_deletes_transcript_files(self, db, tmp_path):
        """Issue #3015: auto-prune must also delete on-disk transcript files."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        db.create_session(session_id="new", source="cli")  # active

        # Transcript files mimicking real gateway/CLI layout
        (sessions_dir / "old1.json").write_text("{}")
        (sessions_dir / "old1.jsonl").write_text("{}\n")
        (sessions_dir / "old2.jsonl").write_text("{}\n")
        (sessions_dir / "request_dump_old1_001.json").write_text("{}")
        (sessions_dir / "new.jsonl").write_text("{}\n")  # active, must survive

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90, sessions_dir=sessions_dir
        )
        assert result["pruned"] == 2

        # Pruned transcript files are gone
        assert not (sessions_dir / "old1.json").exists()
        assert not (sessions_dir / "old1.jsonl").exists()
        assert not (sessions_dir / "old2.jsonl").exists()
        assert not (sessions_dir / "request_dump_old1_001.json").exists()
        # Active session's transcript is untouched
        assert (sessions_dir / "new.jsonl").exists()

