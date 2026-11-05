"""Tests for RC2: pre-publication lease refresh in the backend's publish_compression_child.

When the lease refresher stopped due to transient DB failures, the final
pre-publication refresh inside the same transaction gives one last chance
to extend the lease before the expiry check.
"""
import time
from unittest.mock import patch

import pytest

from persistence.sqlite._holder import StoreHolder
from hermes_state_errors import CompressionSessionBusyError
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.sessions import create_session


def _setup_db(tmp_path):
    db = StoreHolder(tmp_path / "state.db")
    return db


def _seed_lock(conn, session_id, holder, expired=False):
    now = time.time()
    conn.execute(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (session_id, holder, now, (now - 10.0) if expired else (now + 300.0)),
    )


class TestLeaseRefreshBeforePublish:

    def test_refresher_stopped_final_refresh_succeeds(self, tmp_path):
        db = _setup_db(tmp_path)
        create_session(db, "parent-1", source="test")
        _seed_lock(db._conn, "parent-1", "holder-1", expired=True)

        with patch.object(db, "_execute_write", side_effect=lambda fn: fn(db._conn)):
            publish_compression_child(
                db,
                parent_session_id="parent-1",
                child_session_id="child-1",
                source="test",
                messages=[{"role": "user", "content": "hello"}],
                compression_lock_holder="holder-1",
                require_compression_lease=True,
                require_lease_refresh=True,
                lease_ttl_seconds=300.0,
            )

        lock = db._conn.execute(
            "SELECT expires_at FROM compression_locks WHERE session_id = ?",
            ("parent-1",),
        ).fetchone()
        assert lock is not None
        assert lock[0] > time.time()

        parent = db._conn.execute(
            "SELECT ended_at FROM sessions WHERE id = ?",
            ("parent-1",),
        ).fetchone()
        assert parent is not None
        assert parent[0] is not None

    def test_refresher_stopped_final_refresh_fails_wrong_holder(self, tmp_path):
        db = _setup_db(tmp_path)
        _seed_lock(db._conn, "parent-1", "other-holder", expired=True)

        with patch.object(db, "_execute_write", side_effect=lambda fn: fn(db._conn)):
            with pytest.raises(CompressionSessionBusyError, match="lease lost"):
                publish_compression_child(
                    db,
                    parent_session_id="parent-1",
                    child_session_id="child-1",
                    source="test",
                    messages=[{"role": "user", "content": "hello"}],
                    compression_lock_holder="holder-1",
                    require_compression_lease=True,
                    require_lease_refresh=True,
                    lease_ttl_seconds=300.0,
                )

    def test_refresher_healthy_no_duplicate_behavior(self, tmp_path):
        db = _setup_db(tmp_path)
        create_session(db, "parent-1", source="test")
        now = time.time()
        future = now + 300.0
        conn = db._conn
        conn.execute(
            "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
            ("parent-1", "holder-1", now, future),
        )

        with patch.object(db, "_execute_write", side_effect=lambda fn: fn(db._conn)):
            publish_compression_child(
                db,
                parent_session_id="parent-1",
                child_session_id="child-1",
                source="test",
                messages=[{"role": "user", "content": "hello"}],
                compression_lock_holder="holder-1",
                require_compression_lease=True,
                require_lease_refresh=True,
                lease_ttl_seconds=300.0,
            )

        lock = conn.execute(
            "SELECT expires_at FROM compression_locks WHERE session_id = ?",
            ("parent-1",),
        ).fetchone()
        assert lock is not None
        assert lock[0] >= future

    def test_stale_holder_cannot_refresh_and_publish(self, tmp_path):
        db = _setup_db(tmp_path)
        _seed_lock(db._conn, "parent-1", "new-holder", expired=False)

        with patch.object(db, "_execute_write", side_effect=lambda fn: fn(db._conn)):
            with pytest.raises(CompressionSessionBusyError, match="lease lost"):
                publish_compression_child(
                    db,
                    parent_session_id="parent-1",
                    child_session_id="child-1",
                    source="test",
                    messages=[{"role": "user", "content": "hello"}],
                    compression_lock_holder="old-holder",
                    require_compression_lease=True,
                    require_lease_refresh=True,
                    lease_ttl_seconds=300.0,
                )

    def test_no_refresh_when_require_lease_refresh_false(self, tmp_path):
        db = _setup_db(tmp_path)
        _seed_lock(db._conn, "parent-1", "holder-1", expired=True)

        with patch.object(db, "_execute_write", side_effect=lambda fn: fn(db._conn)):
            with pytest.raises(CompressionSessionBusyError, match="lease lost"):
                publish_compression_child(
                    db,
                    parent_session_id="parent-1",
                    child_session_id="child-1",
                    source="test",
                    messages=[{"role": "user", "content": "hello"}],
                    compression_lock_holder="holder-1",
                    require_compression_lease=True,
                    require_lease_refresh=False,
                    lease_ttl_seconds=300.0,
                )

    def test_refresh_and_lease_check_are_atomic(self, tmp_path):
        db = _setup_db(tmp_path)
        create_session(db, "parent-1", source="test")
        _seed_lock(db._conn, "parent-1", "holder-1", expired=True)

        real_execute_write = StoreHolder._execute_write

        def intercepted_execute_write(self, fn, patience_s=None):
            original_fn = fn
            def wrapper(conn):
                result = original_fn(conn)
                lock = conn.execute(
                    "SELECT expires_at FROM compression_locks WHERE session_id = ?",
                    ("parent-1",),
                ).fetchone()
                assert lock is not None
                assert lock[0] > time.time()
                return result
            return real_execute_write(self, wrapper, patience_s)

        with patch.object(StoreHolder, "_execute_write", intercepted_execute_write):
            publish_compression_child(
                db,
                parent_session_id="parent-1",
                child_session_id="child-1",
                source="test",
                messages=[{"role": "user", "content": "hello"}],
                compression_lock_holder="holder-1",
                require_compression_lease=True,
                require_lease_refresh=True,
                lease_ttl_seconds=300.0,
            )
