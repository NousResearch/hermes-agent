"""Tests for #65194: startup-time sweep of orphaned UI-stack sessions.

The TUI/desktop gateway reaps disconnected websocket sessions with an
in-process ``threading.Timer`` grace timer.  A gateway restart destroys the
timer, so the session row stays ``ended_at IS NULL`` forever — nothing
re-checks stale rows on the next boot.  ``SessionDB.sweep_orphaned_sessions()``
is the DB-level startup sweep that closes such rows with a distinct
``end_reason='startup_orphan_reap'``.

Staleness requires BOTH ``started_at`` and the newest ``messages.timestamp``
to be older than the cutoff:

* message-recency alone would sweep a freshly created compression/branch
  child that carries old *copied* message timestamps;
* ``started_at`` alone would sweep a long-lived session that is still
  actively producing messages.
"""

import threading
import time

import pytest

from hermes_state import SessionDB

IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _backdate_session(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, session_id)
    )
    db._conn.commit()


def _set_message_timestamps(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id)
    )
    db._conn.commit()


def _set_last_activity(db: SessionDB, session_id: str, ts: float) -> None:
    conn = db._conn
    assert conn is not None
    conn.execute(
        "UPDATE sessions SET last_activity_at = ? WHERE id = ?", (ts, session_id)
    )
    conn.commit()


def _make_session(
    db: SessionDB,
    session_id: str,
    *,
    source: str,
    started_at: float,
    message_at: float = None,
) -> None:
    db.create_session(session_id, source=source)
    if message_at is not None:
        db.append_message(session_id, role="user", content="hello")
        _set_message_timestamps(db, session_id, message_at)
    _backdate_session(db, session_id, started_at)


class TestSweepOrphanedSessions:
    def test_stale_tui_session_swept(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["stale-tui"]

        row = db.get_session("stale-tui")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "startup_orphan_reap"

    def test_stale_desktop_session_swept(self, db):
        """Desktop chat rows use the same gateway and the same Timer path."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-desktop", source="desktop", started_at=stale, message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["stale-desktop"]
        assert db.get_session("stale-desktop")["end_reason"] == "startup_orphan_reap"

    def test_stale_subagent_session_swept(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-sub", source="subagent", started_at=stale, message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["stale-sub"]
        assert db.get_session("stale-sub")["end_reason"] == "startup_orphan_reap"

    def test_recent_message_spares_old_session(self, db):
        """A long-lived session that is still talking is NOT an orphan."""
        stale = time.time() - 48 * 3600
        _make_session(
            db, "active", source="tui", started_at=stale, message_at=time.time()
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("active")["ended_at"] is None

    def test_recent_heartbeat_spares_old_session(self, db):
        """A turn heartbeat is activity even before its next message lands."""
        stale = time.time() - 48 * 3600
        _make_session(
            db, "active-heartbeat", source="tui", started_at=stale, message_at=stale
        )
        _set_last_activity(db, "active-heartbeat", time.time())

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("active-heartbeat")["ended_at"] is None

    def test_fresh_session_with_old_copied_messages_spared(self, db):
        """Compression/branch children copy history — old message timestamps
        on a just-created row must not get it swept."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "fresh-child", source="tui", started_at=time.time(), message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("fresh-child")["ended_at"] is None

    def test_gateway_owned_source_not_swept(self, db):
        """telegram/discord/... rows belong to the messaging gateway (#60609)."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "tg-sess", source="telegram", started_at=stale, message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("tg-sess")["ended_at"] is None

    def test_already_ended_session_untouched(self, db):
        """First end_reason wins — the sweep never rewrites history."""
        stale = time.time() - 8 * 3600
        _make_session(db, "done", source="tui", started_at=stale, message_at=stale)
        db.end_session("done", "user_exit")

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("done")["end_reason"] == "user_exit"

    def test_stale_empty_session_swept_fresh_spared(self, db):
        """Rows without messages fall back to started_at staleness."""
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-empty", source="tui", started_at=stale)
        _make_session(db, "fresh-empty", source="tui", started_at=time.time())

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["stale-empty"]
        assert db.get_session("stale-empty")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("fresh-empty")["ended_at"] is None

    def test_exclude_ids_spares_live_row(self, db):
        """A row this process still holds in memory must not be closed."""
        stale = time.time() - 8 * 3600
        _make_session(db, "live-tui", source="tui", started_at=stale, message_at=stale)
        _make_session(db, "dead-tui", source="tui", started_at=stale, message_at=stale)

        swept = db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, exclude_ids=("live-tui",)
        )
        assert swept == ["dead-tui"]
        assert db.get_session("live-tui")["ended_at"] is None
        assert db.get_session("dead-tui")["end_reason"] == "startup_orphan_reap"

    def test_custom_sources_respected(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-cli", source="cli", started_at=stale, message_at=stale)
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["stale-cli"]
        assert db.get_session("stale-cli")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("stale-tui")["ended_at"] is None

    def test_explicit_source_scope_spares_gateway_sessions(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-cron", source="cron", started_at=stale, message_at=stale
        )
        for sid, session_key in (
            ("keyed-telegram", "telegram:chat:1"),
            ("unkeyed-telegram", None),
        ):
            db.create_session(sid, source="telegram", session_key=session_key)
            db.append_message(sid, role="user", content="hello")
            _set_message_timestamps(db, sid, stale)
            _backdate_session(db, sid, stale)

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cron",)
        ) == ["stale-cron"]
        assert db.get_session("stale-cron")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("keyed-telegram")["ended_at"] is None
        assert db.get_session("unkeyed-telegram")["ended_at"] is None

    def test_automatic_source_scope_spares_pinned_session(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "pinned", source="cli", started_at=stale, message_at=stale
        )
        db.set_session_pinned("pinned", True)

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S,
            sources=("cli",),
            exclude_pinned=True,
        ) == []
        assert db.get_session("pinned")["ended_at"] is None

    def test_live_turn_lease_on_compression_lineage_spares_session(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "root", source="cli", started_at=stale, message_at=stale)
        db.end_session("root", "compression")
        db.create_session("tip", source="cli", parent_session_id="root")
        db.append_message("tip", role="user", content="continued")
        _set_message_timestamps(db, "tip", stale)
        _backdate_session(db, "tip", stale)
        assert db.try_acquire_session_turn_lease(
            "tip", "external-turn", ttl_seconds=300
        )

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cli",)
        ) == []
        assert db.get_session("tip")["ended_at"] is None

    def test_active_compression_lock_spares_and_expiry_fences_owner(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "compressing", source="cli", started_at=stale, message_at=stale
        )
        assert db.try_acquire_compression_lock(
            "compressing", "compressor", ttl_seconds=300
        )

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cli",)
        ) == []

        conn = db._conn
        assert conn is not None
        conn.execute(
            "UPDATE compression_locks SET expires_at = ? WHERE session_id = ?",
            (time.time() - 1, "compressing"),
        )
        conn.commit()

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["compressing"]
        assert db.get_compression_lock_holder("compressing") is None
        assert db.refresh_compression_lock("compressing", "compressor") is False

    def test_expired_turn_lease_does_not_block_sweep(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "expired", source="cli", started_at=stale, message_at=stale
        )
        assert db.try_acquire_session_turn_lease(
            "expired", "expired-turn", ttl_seconds=300
        )
        db._conn.execute(
            "UPDATE session_turn_leases SET expires_at = ? WHERE conversation_id = ?",
            (time.time() - 1, "expired"),
        )
        db._conn.commit()

        assert db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["expired"]
        assert db.get_session("expired")["end_reason"] == "startup_orphan_reap"
        assert db.refresh_session_turn_lease("expired", "expired-turn") is False

    def test_auto_prune_closes_stale_state_owned_rows_but_spares_live_turns(self, db):
        stale = time.time() - 100 * 86400
        recent = time.time() - 86400
        for sid, source in (
            ("orphan", "cli"),
            ("live-turn", "cli"),
            ("stale-cron", "cron"),
            ("runtime-owned-ui", "tui"),
        ):
            _make_session(db, sid, source=source, started_at=stale, message_at=stale)
            _set_last_activity(db, sid, stale)
        _make_session(
            db,
            "recent-orphan",
            source="cli",
            started_at=recent,
            message_at=recent,
        )
        _set_last_activity(db, "recent-orphan", recent)
        db.create_session(
            "keyed", source="telegram", session_key="telegram:chat:1"
        )
        _backdate_session(db, "keyed", stale)
        db.create_session("unkeyed-gateway", source="telegram")
        _backdate_session(db, "unkeyed-gateway", stale)
        _set_last_activity(db, "unkeyed-gateway", stale)
        assert db.try_acquire_session_turn_lease(
            "live-turn", "external-turn", ttl_seconds=300
        )
        db.register_backend_heartbeat(
            backend_id="unrelated-dashboard",
            pid=12345,
            started_at=time.time(),
            last_heartbeat=time.time(),
        )

        first = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert first["pruned"] == 0
        assert db.get_session("orphan")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("stale-cron")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("live-turn")["ended_at"] is None
        assert db.get_session("recent-orphan")["ended_at"] is None
        assert db.get_session("runtime-owned-ui")["ended_at"] is None
        assert db.get_session("keyed")["ended_at"] is None
        assert db.get_session("unkeyed-gateway")["ended_at"] is None

        second = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert second["pruned"] == 0
        assert db.get_session("orphan") is not None
        assert db.get_session("stale-cron") is not None

        db._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id IN (?, ?)",
            (stale, "orphan", "stale-cron"),
        )
        db._conn.commit()
        third = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert third["pruned"] == 2
        assert db.get_session("orphan") is None
        assert db.get_session("stale-cron") is None

    def test_failed_maintenance_marker_keeps_newly_swept_row_recoverable(
        self, db, monkeypatch
    ):
        stale = time.time() - 100 * 86400
        _make_session(
            db,
            "recoverable",
            source="cli",
            started_at=stale,
            message_at=stale,
        )
        _set_last_activity(db, "recoverable", stale)
        set_meta = db.set_meta
        fail_once = True

        def flaky_set_meta(key, value):
            nonlocal fail_once
            if key == "last_auto_prune" and fail_once:
                fail_once = False
                raise RuntimeError("injected marker failure")
            return set_meta(key, value)

        monkeypatch.setattr(db, "set_meta", flaky_set_meta)

        first = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )
        retry = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert first["error"] == "injected marker failure"
        assert retry["pruned"] == 0
        assert db.get_session("recoverable")["end_reason"] == "startup_orphan_reap"

    def test_concurrent_auto_maintenance_preserves_the_recovery_window(
        self, db, monkeypatch
    ):
        stale = time.time() - 100 * 86400
        _make_session(db, "concurrent", source="cli", started_at=stale, message_at=stale)
        _set_last_activity(db, "concurrent", stale)
        peer = SessionDB(db.db_path)
        read_barrier = threading.Barrier(2)
        second_done = threading.Event()
        release_first_prune = threading.Event()
        errors = []
        results = {}

        for instance in (db, peer):
            get_meta = instance.get_meta

            def synchronized_get_meta(key, *, _get_meta=get_meta):
                value = _get_meta(key)
                if key == "last_auto_prune":
                    try:
                        read_barrier.wait(timeout=1)
                    except threading.BrokenBarrierError:
                        pass
                return value

            monkeypatch.setattr(instance, "get_meta", synchronized_get_meta)

        prune_sessions = db.prune_sessions

        def delayed_prune(*args, **kwargs):
            assert release_first_prune.wait(timeout=5)
            return prune_sessions(*args, **kwargs)

        monkeypatch.setattr(db, "prune_sessions", delayed_prune)

        def run(name, instance, *, done=None):
            try:
                results[name] = instance.maybe_auto_prune_and_vacuum(
                    retention_days=90,
                    min_interval_hours=24,
                    vacuum=False,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        first = threading.Thread(target=run, args=("first", db))
        second = threading.Thread(
            target=run, args=("second", peer), kwargs={"done": second_done}
        )
        try:
            first.start()
            second.start()
            assert second_done.wait(timeout=5)
            release_first_prune.set()
        finally:
            release_first_prune.set()
            first.join(timeout=5)
            second.join(timeout=5)
            peer.close()

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert sum(bool(result["skipped"]) for result in results.values()) == 1
        assert sum(int(result["pruned"]) for result in results.values()) == 0
        assert db.get_session("concurrent")["end_reason"] == "startup_orphan_reap"

    def test_auto_prune_spares_compression_root_of_live_turn(self, db):
        stale = time.time() - 100 * 86400
        _make_session(db, "root", source="cli", started_at=stale, message_at=stale)
        db.end_session("root", "compression")
        db.create_session("tip", source="cli", parent_session_id="root")
        db.append_message("tip", role="user", content="continued")
        _set_message_timestamps(db, "tip", stale)
        _backdate_session(db, "tip", stale)
        _set_last_activity(db, "tip", stale)
        assert db.try_acquire_session_turn_lease(
            "tip", "external-turn", ttl_seconds=300
        )

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert db.get_session("root") is not None
        assert db.get_session("tip")["ended_at"] is None

    def test_auto_prune_spares_prior_sweep_row_with_new_turn_lease(self, db):
        stale = time.time() - 100 * 86400
        _make_session(db, "racy", source="cli", started_at=stale, message_at=stale)
        _set_last_activity(db, "racy", stale)
        db.end_session("racy", "startup_orphan_reap")
        assert db.try_acquire_session_turn_lease(
            "racy", "arriving-turn", ttl_seconds=300
        )

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert db.get_session("racy") is not None

    def test_auto_prune_spares_prior_sweep_row_with_new_compression_lock(self, db):
        stale = time.time() - 100 * 86400
        _make_session(
            db,
            "racy-compression",
            source="cli",
            started_at=stale,
            message_at=stale,
        )
        _set_last_activity(db, "racy-compression", stale)
        db.end_session("racy-compression", "startup_orphan_reap")
        assert db.try_acquire_compression_lock(
            "racy-compression", "arriving-compressor", ttl_seconds=300
        )

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert db.get_session("racy-compression") is not None

    def test_returns_empty_on_empty_db(self, db):
        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []

    def test_auto_prune_reports_closed_count_and_deletes_after_second_window(
        self, db
    ):
        """#54189 end-to-end: leaky producers (cron/kanban/subagent) never set
        ``ended_at``; pass 1 closes them (reported via ``closed``), pass 2 —
        after a further retention window — deletes them, and a messaging row
        is never touched by either pass."""
        stale = time.time() - 200 * 86400
        for sid, source in (
            ("cron-0", "cron"),
            ("kanban-1", "kanban"),
            ("subagent-2", "subagent"),
            ("telegram-3", "telegram"),
        ):
            _make_session(db, sid, source=source, started_at=stale, message_at=stale)
            _set_last_activity(db, sid, stale)

        first = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert first["closed"] == 3
        assert first["pruned"] == 0
        for sid in ("cron-0", "kanban-1", "subagent-2"):
            assert db.get_session(sid)["end_reason"] == "startup_orphan_reap"
        assert db.get_session("telegram-3")["ended_at"] is None

        # Simulate the next maintenance pass after another retention window.
        db._conn.execute(
            "UPDATE sessions SET ended_at = ended_at - 91 * 86400 "
            "WHERE end_reason = 'startup_orphan_reap'"
        )
        db._conn.commit()
        second = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert second["closed"] == 0
        assert second["pruned"] == 3
        remaining = [r["id"] for r in db._conn.execute("SELECT id FROM sessions")]
        assert remaining == ["telegram-3"]

    def test_zero_ttl_is_noop(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)
        assert db.sweep_orphaned_sessions(max_idle_seconds=0) == []
        assert db.get_session("stale-tui")["ended_at"] is None
