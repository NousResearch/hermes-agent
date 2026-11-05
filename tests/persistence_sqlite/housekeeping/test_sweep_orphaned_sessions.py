"""Tests for #65194: startup-time sweep of orphaned UI-stack sessions.

The TUI/desktop gateway reaps disconnected websocket sessions with an
in-process ``threading.Timer`` grace timer.  A gateway restart destroys the
timer, so the session row stays ``ended_at IS NULL`` forever: nothing
re-checks stale rows on the next boot.  ``sweep_orphaned_sessions()``
is the DB-level startup sweep that closes such rows with a distinct
``end_reason='startup_orphan_reap'``.

Staleness requires BOTH ``started_at`` and the newest ``messages.timestamp``
to be older than the cutoff:

* message-recency alone would sweep a freshly created compression/branch
  child that carries old *copied* message timestamps;
* ``started_at`` alone would sweep a long-lived session that is still
  actively producing messages.

The startup auto-maintenance pass (``session_retention.run_auto_prune_and_vacuum``) runs the
store's prune and this sweep under the store's auto-maintenance lock.
"""

import sqlite3
import threading
import time

import pytest

import session_retention
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, set_session_pinned,
)
from tests.persistence_sqlite.conftest import store_for

IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    return store.holder()


@pytest.fixture
def maintain(backend, scope_a, tmp_path):
    """One startup auto-maintenance pass over a store of scope_a, holding the store's maintenance lock."""
    admin = backend.open_store_admin(scope_a, read_only=False)

    def run(store, **kwargs):
        return session_retention.run_auto_prune_and_vacuum(
            store, admin, store_home=scope_a.profile_dir, media_root=str(tmp_path / "media"), **kwargs)
    return run


def _sweep(store, **kwargs):
    return list(store.maintenance.sweep_orphaned_sessions(**kwargs))


def _backdate_session(db, session_id: str, ts: float) -> None:
    db._write_sql(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, session_id)
    )


def _set_message_timestamps(db, session_id: str, ts: float) -> None:
    db._write_sql(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id)
    )


def _set_last_activity(db, session_id: str, ts: float) -> None:
    db._write_sql(
        "UPDATE sessions SET last_activity_at = ? WHERE id = ?", (ts, session_id)
    )


def _make_session(
    db,
    session_id: str,
    *,
    source: str,
    started_at: float,
    message_at: float | None = None,
) -> None:
    create_session(db, session_id, source=source)
    if message_at is not None:
        append_message(db, session_id, role="user", content="hello")
        _set_message_timestamps(db, session_id, message_at)
    _backdate_session(db, session_id, started_at)


class TestSweepOrphanedSessions:
    def test_stale_tui_session_swept(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert _sweep(store, max_idle_seconds=IDLE_S) == ["stale-tui"]

        row = get_session(db, "stale-tui")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "startup_orphan_reap"

    def test_stale_desktop_session_swept(self, store, db):
        """Desktop chat rows use the same gateway and the same Timer path."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-desktop", source="desktop", started_at=stale, message_at=stale
        )

        assert _sweep(store, max_idle_seconds=IDLE_S) == ["stale-desktop"]
        assert get_session(db, "stale-desktop")["end_reason"] == "startup_orphan_reap"

    def test_stale_subagent_session_swept(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-sub", source="subagent", started_at=stale, message_at=stale
        )

        assert _sweep(store, max_idle_seconds=IDLE_S) == ["stale-sub"]
        assert get_session(db, "stale-sub")["end_reason"] == "startup_orphan_reap"

    def test_recent_message_spares_old_session(self, store, db):
        """A long-lived session that is still talking is NOT an orphan."""
        stale = time.time() - 48 * 3600
        _make_session(
            db, "active", source="tui", started_at=stale, message_at=time.time()
        )

        assert _sweep(store, max_idle_seconds=IDLE_S) == []
        assert get_session(db, "active")["ended_at"] is None

    def test_recent_heartbeat_spares_old_session(self, store, db):
        """A turn heartbeat is activity even before its next message lands."""
        stale = time.time() - 48 * 3600
        _make_session(
            db, "active-heartbeat", source="tui", started_at=stale, message_at=stale
        )
        _set_last_activity(db, "active-heartbeat", time.time())

        assert _sweep(store, max_idle_seconds=IDLE_S) == []
        assert get_session(db, "active-heartbeat")["ended_at"] is None

    def test_fresh_session_with_old_copied_messages_spared(self, store, db):
        """Compression/branch children copy history: old message timestamps
        on a just-created row must not get it swept."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "fresh-child", source="tui", started_at=time.time(), message_at=stale
        )

        assert _sweep(store, max_idle_seconds=IDLE_S) == []
        assert get_session(db, "fresh-child")["ended_at"] is None

    def test_gateway_owned_source_not_swept(self, store, db):
        """telegram/discord/... rows belong to the messaging gateway (#60609)."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "tg-sess", source="telegram", started_at=stale, message_at=stale
        )

        assert _sweep(store, max_idle_seconds=IDLE_S) == []
        assert get_session(db, "tg-sess")["ended_at"] is None

    def test_already_ended_session_untouched(self, store, db):
        """First end_reason wins: the sweep never rewrites history."""
        stale = time.time() - 8 * 3600
        _make_session(db, "done", source="tui", started_at=stale, message_at=stale)
        end_session(db, "done", "user_exit")

        assert _sweep(store, max_idle_seconds=IDLE_S) == []
        assert get_session(db, "done")["end_reason"] == "user_exit"

    def test_stale_empty_session_swept_fresh_spared(self, store, db):
        """Rows without messages fall back to started_at staleness."""
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-empty", source="tui", started_at=stale)
        _make_session(db, "fresh-empty", source="tui", started_at=time.time())

        assert _sweep(store, max_idle_seconds=IDLE_S) == ["stale-empty"]
        assert get_session(db, "stale-empty")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "fresh-empty")["ended_at"] is None

    def test_exclude_ids_spares_live_row(self, store, db):
        """A row this process still holds in memory must not be closed."""
        stale = time.time() - 8 * 3600
        _make_session(db, "live-tui", source="tui", started_at=stale, message_at=stale)
        _make_session(db, "dead-tui", source="tui", started_at=stale, message_at=stale)

        swept = _sweep(
            store, max_idle_seconds=IDLE_S, exclude_ids=("live-tui",)
        )
        assert swept == ["dead-tui"]
        assert get_session(db, "live-tui")["ended_at"] is None
        assert get_session(db, "dead-tui")["end_reason"] == "startup_orphan_reap"

    def test_custom_sources_respected(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-cli", source="cli", started_at=stale, message_at=stale)
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["stale-cli"]
        assert get_session(db, "stale-cli")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "stale-tui")["ended_at"] is None

    def test_explicit_source_scope_spares_gateway_sessions(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-cron", source="cron", started_at=stale, message_at=stale
        )
        for sid, session_key in (
            ("keyed-telegram", "telegram:chat:1"),
            ("unkeyed-telegram", None),
        ):
            create_session(db, sid, source="telegram", session_key=session_key)
            append_message(db, sid, role="user", content="hello")
            _set_message_timestamps(db, sid, stale)
            _backdate_session(db, sid, stale)

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cron",)
        ) == ["stale-cron"]
        assert get_session(db, "stale-cron")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "keyed-telegram")["ended_at"] is None
        assert get_session(db, "unkeyed-telegram")["ended_at"] is None

    def test_automatic_source_scope_spares_pinned_session(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "pinned", source="cli", started_at=stale, message_at=stale
        )
        set_session_pinned(db, "pinned", True)

        assert _sweep(
            store,
            max_idle_seconds=IDLE_S,
            sources=("cli",),
            exclude_pinned=True,
        ) == []
        assert get_session(db, "pinned")["ended_at"] is None

    def test_live_turn_lease_on_compression_lineage_spares_session(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "root", source="cli", started_at=stale, message_at=stale)
        end_session(db, "root", "compression")
        create_session(db, "tip", source="cli", parent_session_id="root")
        append_message(db, "tip", role="user", content="continued")
        _set_message_timestamps(db, "tip", stale)
        _backdate_session(db, "tip", stale)
        assert store.coordination.try_acquire_turn_lease(
            session_id="tip", holder="external-turn", ttl_seconds=300
        )

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cli",)
        ) == []
        assert get_session(db, "tip")["ended_at"] is None

    def test_active_compression_lock_spares_and_expiry_fences_owner(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "compressing", source="cli", started_at=stale, message_at=stale
        )
        assert store.coordination.try_acquire_compression_lock(
            session_id="compressing", holder="compressor", ttl_seconds=300
        )

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cli",)
        ) == []

        db._write_sql(
            "UPDATE compression_locks SET expires_at = ? WHERE session_id = ?",
            (time.time() - 1, "compressing"),
        )

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["compressing"]
        assert store.coordination.get_compression_lock_holder(session_id="compressing") is None
        assert store.coordination.refresh_compression_lock(session_id="compressing", holder="compressor") is False

    def test_expired_turn_lease_does_not_block_sweep(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "expired", source="cli", started_at=stale, message_at=stale
        )
        assert store.coordination.try_acquire_turn_lease(
            session_id="expired", holder="expired-turn", ttl_seconds=300
        )
        db._write_sql(
            "UPDATE session_turn_leases SET expires_at = ? WHERE conversation_id = ?",
            (time.time() - 1, "expired"),
        )

        assert _sweep(
            store, max_idle_seconds=IDLE_S, sources=("cli",)
        ) == ["expired"]
        assert get_session(db, "expired")["end_reason"] == "startup_orphan_reap"
        assert store.coordination.refresh_turn_lease(session_id="expired", holder="expired-turn") is False

    def test_auto_prune_closes_stale_state_owned_rows_but_spares_live_turns(self, store, db, maintain):
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
        create_session(
            db, "keyed", source="telegram", session_key="telegram:chat:1"
        )
        _backdate_session(db, "keyed", stale)
        create_session(db, "unkeyed-gateway", source="telegram")
        _backdate_session(db, "unkeyed-gateway", stale)
        _set_last_activity(db, "unkeyed-gateway", stale)
        assert store.coordination.try_acquire_turn_lease(
            session_id="live-turn", holder="external-turn", ttl_seconds=300
        )
        store.maintenance.register_backend_heartbeat(
            backend_id="unrelated-dashboard",
            pid=12345,
            started_at=time.time(),
            last_heartbeat=time.time(),
        )

        first = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert first["pruned"] == 0
        assert get_session(db, "orphan")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "stale-cron")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "live-turn")["ended_at"] is None
        assert get_session(db, "recent-orphan")["ended_at"] is None
        assert get_session(db, "runtime-owned-ui")["ended_at"] is None
        assert get_session(db, "keyed")["ended_at"] is None
        assert get_session(db, "unkeyed-gateway")["ended_at"] is None

        second = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert second["pruned"] == 0
        assert get_session(db, "orphan") is not None
        assert get_session(db, "stale-cron") is not None

        db._write_sql(
            "UPDATE sessions SET ended_at = ? WHERE id IN (?, ?)",
            (stale, "orphan", "stale-cron"),
        )
        third = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert third["pruned"] == 2
        assert get_session(db, "orphan") is None
        assert get_session(db, "stale-cron") is None

    def test_failed_maintenance_marker_keeps_newly_swept_row_recoverable(
        self, store, db, maintain, monkeypatch
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
        set_meta = store.state_meta.set
        fail_once = True

        def flaky_set_meta(key, value):
            nonlocal fail_once
            if key == "last_auto_prune" and fail_once:
                fail_once = False
                raise RuntimeError("injected marker failure")
            return set_meta(key, value)

        monkeypatch.setattr(store.state_meta, "set", flaky_set_meta)

        first = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )
        retry = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert first["error"] == "injected marker failure"
        assert retry["pruned"] == 0
        assert get_session(db, "recoverable")["end_reason"] == "startup_orphan_reap"

    def test_concurrent_auto_maintenance_preserves_the_recovery_window(
        self, backend, scope_a, store, db, maintain, monkeypatch
    ):
        stale = time.time() - 100 * 86400
        _make_session(db, "concurrent", source="cli", started_at=stale, message_at=stale)
        _set_last_activity(db, "concurrent", stale)
        peer = backend.open(scope_a)
        read_barrier = threading.Barrier(2)
        second_done = threading.Event()
        release_first_prune = threading.Event()
        errors = []
        results = {}

        for instance in (store, peer):
            get_meta = instance.state_meta.get

            def synchronized_get_meta(key, *, _get_meta=get_meta):
                value = _get_meta(key)
                if key == "last_auto_prune":
                    try:
                        read_barrier.wait(timeout=1)
                    except threading.BrokenBarrierError:
                        pass
                return value

            monkeypatch.setattr(instance.state_meta, "get", synchronized_get_meta)

        prune_sessions = store.maintenance.prune_sessions

        def delayed_prune(*args, **kwargs):
            assert release_first_prune.wait(timeout=5)
            return prune_sessions(*args, **kwargs)

        monkeypatch.setattr(store.maintenance, "prune_sessions", delayed_prune)

        def run(name, instance, *, done=None):
            try:
                results[name] = maintain(
                    instance,
                    retention_days=90,
                    min_interval_hours=24,
                    vacuum=False,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        first = threading.Thread(target=run, args=("first", store))
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

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert sum(bool(result["skipped"]) for result in results.values()) == 1
        assert sum(int(result["pruned"]) for result in results.values()) == 0
        assert get_session(db, "concurrent")["end_reason"] == "startup_orphan_reap"

    def test_auto_prune_spares_compression_root_of_live_turn(self, store, db, maintain):
        stale = time.time() - 100 * 86400
        _make_session(db, "root", source="cli", started_at=stale, message_at=stale)
        end_session(db, "root", "compression")
        create_session(db, "tip", source="cli", parent_session_id="root")
        append_message(db, "tip", role="user", content="continued")
        _set_message_timestamps(db, "tip", stale)
        _backdate_session(db, "tip", stale)
        _set_last_activity(db, "tip", stale)
        assert store.coordination.try_acquire_turn_lease(
            session_id="tip", holder="external-turn", ttl_seconds=300
        )

        result = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert get_session(db, "root") is not None
        assert get_session(db, "tip")["ended_at"] is None

    def test_auto_prune_spares_prior_sweep_row_with_new_turn_lease(self, store, db, maintain):
        stale = time.time() - 100 * 86400
        _make_session(db, "racy", source="cli", started_at=stale, message_at=stale)
        _set_last_activity(db, "racy", stale)
        end_session(db, "racy", "startup_orphan_reap")
        assert store.coordination.try_acquire_turn_lease(
            session_id="racy", holder="arriving-turn", ttl_seconds=300
        )

        result = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert get_session(db, "racy") is not None

    def test_auto_prune_spares_prior_sweep_row_with_new_compression_lock(self, store, db, maintain):
        stale = time.time() - 100 * 86400
        _make_session(
            db,
            "racy-compression",
            source="cli",
            started_at=stale,
            message_at=stale,
        )
        _set_last_activity(db, "racy-compression", stale)
        end_session(db, "racy-compression", "startup_orphan_reap")
        assert store.coordination.try_acquire_compression_lock(
            session_id="racy-compression", holder="arriving-compressor", ttl_seconds=300
        )

        result = maintain(
            store,
            retention_days=90,
            min_interval_hours=0,
            vacuum=False,
        )

        assert result["pruned"] == 0
        assert get_session(db, "racy-compression") is not None

    def test_returns_empty_on_empty_db(self, store):
        assert _sweep(store, max_idle_seconds=IDLE_S) == []

    def test_auto_prune_reports_closed_count_and_deletes_after_second_window(
        self, store, db, maintain
    ):
        """#54189 end-to-end: leaky producers (cron/kanban/subagent) never set
        ``ended_at``; pass 1 closes them (reported via ``closed``), pass 2:
        after a further retention window: deletes them, and a messaging row
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

        first = maintain(
            store, retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert first["closed"] == 3
        assert first["pruned"] == 0
        for sid in ("cron-0", "kanban-1", "subagent-2"):
            assert get_session(db, sid)["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "telegram-3")["ended_at"] is None

        # Simulate the next maintenance pass after another retention window.
        db._write_sql(
            "UPDATE sessions SET ended_at = ended_at - 91 * 86400 "
            "WHERE end_reason = 'startup_orphan_reap'"
        )
        second = maintain(
            store, retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert second["closed"] == 0
        assert second["pruned"] == 3
        remaining = [r["id"] for r in db._read_all("SELECT id FROM sessions")]
        assert remaining == ["telegram-3"]

    def test_auto_prune_closes_and_later_deletes_recovered_placeholders(self, store, db, maintain):
        """#114730: ``sessions recover`` synthesizes ``source='recovered'`` placeholder
        rows with no ``ended_at``. Without ``'recovered'`` in
        ``AUTO_PRUNE_STALE_OPEN_SOURCES`` these rows are immortal: never closed by
        the sweep above, and therefore never reached by the prune either."""
        stale = time.time() - 200 * 86400
        _make_session(db, "recovered-0", source="recovered", started_at=stale, message_at=stale)
        _set_last_activity(db, "recovered-0", stale)

        first = maintain(
            store, retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert first["closed"] == 1
        assert get_session(db, "recovered-0")["end_reason"] == "startup_orphan_reap"

        db._write_sql(
            "UPDATE sessions SET ended_at = ended_at - 91 * 86400 WHERE id = 'recovered-0'"
        )
        second = maintain(
            store, retention_days=90, min_interval_hours=0, vacuum=False
        )
        assert second["pruned"] == 1
        assert get_session(db, "recovered-0") is None

    def test_real_recovery_placeholders_age_out_but_fresh_one_stays_open(self, store, db, maintain, scope_a):
        """Through the real `_reconstruct_missing_sessions` path: a stale placeholder is closed
        then deleted; a fresh one is left open (source survives as 'recovered')."""
        from persistence.sqlite.admin._recovery import _reconstruct_missing_sessions

        stale = time.time() - 200 * 86400
        for sid in ("lost-old", "lost-fresh"):
            create_session(db, sid, source="cli")
            append_message(db, sid, role="user", content="salvaged")
        raw = sqlite3.connect(store_for(scope_a))
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("DELETE FROM sessions WHERE id IN ('lost-old', 'lost-fresh')")
        assert _reconstruct_missing_sessions(raw)["sessions_reconstructed"] == 2
        raw.execute("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = 'lost-old'", (stale, stale))
        raw.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'lost-old'", (stale,))
        raw.commit()
        raw.close()

        first = maintain(store, retention_days=90, min_interval_hours=0, vacuum=False)
        assert first["closed"] == 1
        assert get_session(db, "lost-old")["end_reason"] == "startup_orphan_reap"
        assert get_session(db, "lost-fresh")["ended_at"] is None

        db._write_sql("UPDATE sessions SET ended_at = ended_at - 91 * 86400 WHERE id = 'lost-old'")
        second = maintain(store, retention_days=90, min_interval_hours=0, vacuum=False)
        assert second["pruned"] == 1
        assert get_session(db, "lost-old") is None
        assert get_session(db, "lost-fresh")["source"] == "recovered"

    def test_zero_ttl_is_noop(self, store, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)
        assert _sweep(store, max_idle_seconds=0) == []
        assert get_session(db, "stale-tui")["ended_at"] is None
