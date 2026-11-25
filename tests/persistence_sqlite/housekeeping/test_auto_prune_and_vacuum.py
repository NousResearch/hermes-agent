"""Automatic prune and VACUUM on a SQLite backend store: ``session_retention.run_auto_prune_and_vacuum`` over the store's maintenance repository and its writable admin handle (the time throttles, the freelist-ratio gate, a failed VACUUM retried, transcript files of pruned sessions removed), the admin's real freelist ratio, a bounded WAL, and VACUUM leaving the WAL truncated."""

import time
from pathlib import Path

import pytest

from persistence.housekeeping.maintenance import PruneResult
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, delete_session, end_session, get_session
from session_retention import run_auto_prune_and_vacuum
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


@pytest.fixture
def admin(backend, scope_a, store):
    return backend.open_store_admin(scope_a, read_only=False)


def _maintain(store, admin, **kwargs):
    """One automatic prune and VACUUM pass over *store* with its writable admin handle."""
    return run_auto_prune_and_vacuum(
        store, admin, store_home=store.holder().db_path.parent, media_root=MEDIA_ROOT, **kwargs)


class TestVacuum:

    def test_auto_maintenance_records_successful_vacuum(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(3))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.5)  # ratio gate open
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin, min_interval_hours=0)

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]
        assert db.get_meta("last_vacuum") is not None

    def test_auto_maintenance_skips_recent_vacuum(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(3))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.5)  # ratio gate open
        db.set_meta("last_vacuum", str(time.time()))
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin,
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_retries_after_vacuum_interval(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(3))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.5)  # ratio gate open
        db.set_meta("last_vacuum", str(time.time() - 31 * 86400))
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin,
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_auto_maintenance_retries_after_failed_vacuum(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(3))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.5)  # ratio gate open
        vacuum_calls = []

        def fail_first_vacuum():
            vacuum_calls.append(True)
            if len(vacuum_calls) == 1:
                raise RuntimeError("vacuum failed")

        monkeypatch.setattr(admin, "vacuum", fail_first_vacuum)

        first = _maintain(store, admin, min_interval_hours=0)

        assert first["vacuumed"] is False
        assert db.get_meta("last_vacuum") is None

        second = _maintain(store, admin, min_interval_hours=0)

        assert second["vacuumed"] is True
        assert vacuum_calls == [True, True]
        assert db.get_meta("last_vacuum") is not None

    # ── freelist-ratio gate (#54189) ─────────────────────────────────────
    def test_auto_maintenance_skips_vacuum_below_freelist_ratio(self, db, store, admin, monkeypatch):
        """A prune that frees few pages on a dense DB must NOT trigger VACUUM."""
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(1))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.05)
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin, min_interval_hours=0)

        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result["freelist_ratio"] == 0.05
        assert vacuum_calls == []
        assert db.get_meta("last_vacuum") is None
        # The prune itself still counts as a maintenance run.
        assert db.get_meta("last_auto_prune") is not None

    def test_auto_maintenance_vacuums_above_freelist_ratio(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(1))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.40)
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin, min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] == 0.40
        assert vacuum_calls == [True]

    def test_auto_maintenance_freelist_ratio_exactly_at_threshold_skips(self, db, store, admin, monkeypatch):
        """Gate is strictly greater-than: 25.0% reclaimable does not VACUUM."""
        from session_retention import AUTO_VACUUM_MIN_FREELIST_RATIO

        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(1))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: AUTO_VACUUM_MIN_FREELIST_RATIO)
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin, min_interval_hours=0)

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_unknown_freelist_ratio_falls_back_to_time_throttle(self, db, store, admin, monkeypatch):
        """If the pragmas cannot be read, don't silently disable VACUUM forever."""
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(1))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: None)
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin, min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] is None
        assert vacuum_calls == [True]

    def test_auto_maintenance_ratio_gate_threshold_is_overridable(self, db, store, admin, monkeypatch):
        monkeypatch.setattr(store.maintenance, "prune_sessions", lambda **_kwargs: PruneResult(1))
        monkeypatch.setattr(admin, "freelist_ratio", lambda: 0.10)
        vacuum_calls = []
        monkeypatch.setattr(admin, "vacuum", lambda: vacuum_calls.append(True))

        result = _maintain(store, admin,
            min_interval_hours=0, min_vacuum_freelist_ratio=0.05
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_freelist_ratio_reads_real_pragmas(self, db, admin):
        """Real-DB check: freeing most of the file pushes the ratio past the gate."""
        from session_retention import AUTO_VACUUM_MIN_FREELIST_RATIO

        create_session(db, session_id="keep", source="cli")
        append_message(db, session_id="keep", role="user", content="hi")
        for i in range(6):
            sid = f"bulk{i}"
            create_session(db, session_id=sid, source="cli")
            for _ in range(20):
                append_message(db, session_id=sid, role="assistant", content="z" * 4000)
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dense = admin.freelist_ratio()
        assert dense is not None and dense < AUTO_VACUUM_MIN_FREELIST_RATIO

        for i in range(6):
            delete_session(db, f"bulk{i}", media_root=MEDIA_ROOT)
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        sparse = admin.freelist_ratio()
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

    def test_vacuum_leaves_wal_truncated(self, db, admin, tmp_path):
        """VACUUM must not strand a giant WAL beside the database.

        VACUUM rewrites every page through the write-ahead log. Without a
        checkpoint *after* it, a 3 GB database leaves a 3 GB state.db-wal
        behind — `sessions optimize` then consumes far more disk than it
        frees, which is the opposite of its purpose.
        """
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            pytest.skip("WAL unavailable on this filesystem")

        create_session(db, session_id="s1", source="cli")
        for i in range(500):
            append_message(db,
                session_id="s1", role="user", content=f"padding message {i} " * 20
            )
        admin.vacuum()

        wal = Path(str(db.db_path) + "-wal")
        if wal.exists():
            limit = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
            assert wal.stat().st_size <= max(limit, 0) or wal.stat().st_size == 0, (
                f"WAL left at {wal.stat().st_size} bytes after VACUUM"
            )


class TestAutoMaintenance:
    def _make_old_ended(self, db, sid: str, days_old: int = 100):
        """Create a session that is ended and was started `days_old` days ago."""
        create_session(db, session_id=sid, source="cli")
        end_session(db, sid, end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - days_old * 86400, sid),
        )
        db._conn.commit()

    def test_first_run_prunes_and_skips_vacuum_when_little_reclaimable(self, db, store, admin):
        """Pruning two empty sessions frees almost nothing → prune yes, VACUUM no."""
        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        create_session(db, session_id="new", source="cli")  # active, must survive

        result = _maintain(store, admin, retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 2
        assert result["vacuumed"] is False  # freelist ratio gate (#54189)
        assert result["freelist_ratio"] is not None
        assert result["freelist_ratio"] <= 0.25
        assert result.get("error") is None
        assert get_session(db, "old1") is None
        assert get_session(db, "old2") is None
        assert get_session(db, "new") is not None

    def test_first_run_prunes_and_vacuums_when_mostly_reclaimable(self, db, store, admin):
        """Pruning the bulk of the file's pages crosses the 25% gate → VACUUM runs."""
        create_session(db, session_id="new", source="cli")  # active, must survive
        append_message(db, session_id="new", role="user", content="hi")
        for i in range(6):
            sid = f"old{i}"
            self._make_old_ended(db, sid, days_old=100)
            for _ in range(20):
                append_message(db, session_id=sid, role="assistant", content="z" * 4000)
            # Keep the row aged: append_message bumps activity, prune ages by
            # latest message, so push the message timestamps back too.
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (time.time() - 100 * 86400, sid),
            )
        db._conn.commit()

        result = _maintain(store, admin, retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 6
        assert result["freelist_ratio"] > 0.25
        assert result["vacuumed"] is True
        assert result.get("error") is None
        assert get_session(db, "new") is not None
        assert db.get_meta("last_vacuum") is not None

    def test_second_call_within_interval_skips(self, db, store, admin):
        self._make_old_ended(db, "old", days_old=100)
        first = _maintain(store, admin,
            retention_days=90, min_interval_hours=24
        )
        assert first["skipped"] is False
        assert first["pruned"] == 1

        # Create another prunable session; a second call within
        # min_interval_hours should still skip without touching it.
        self._make_old_ended(db, "old2", days_old=100)
        second = _maintain(store, admin,
            retention_days=90, min_interval_hours=24
        )
        assert second["skipped"] is True
        assert second["pruned"] == 0
        assert get_session(db, "old2") is not None  # untouched

    def test_auto_prune_deletes_transcript_files(self, db, store, admin, tmp_path):
        """Issue #3015: auto-prune must also delete on-disk transcript files."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        create_session(db, session_id="new", source="cli")  # active

        # Transcript files mimicking real gateway/CLI layout
        (sessions_dir / "old1.json").write_text("{}")
        (sessions_dir / "old1.jsonl").write_text("{}\n")
        (sessions_dir / "old2.jsonl").write_text("{}\n")
        (sessions_dir / "request_dump_old1_001.json").write_text("{}")
        (sessions_dir / "new.jsonl").write_text("{}\n")  # active, must survive

        result = _maintain(store, admin,
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
