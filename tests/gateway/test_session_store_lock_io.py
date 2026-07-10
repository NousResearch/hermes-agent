"""Regression: blocking I/O must not run while session_store._lock is held.

``get_or_create_session`` previously held the store lock during SQLite
SELECTs (``_is_session_ended_in_db``), a full routing-index rewrite +
``os.fsync`` (``_save``), and a recovery DB query
(``_recover_session_from_db``) -- all on every inbound message.

These tests assert those three I/O calls are invoked *outside* the lock.
They follow the mock-DB idiom from ``test_session_store_runtime_stale_guard``.
"""
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _TrackedLock:
    """Drop-in replacement for ``threading.Lock`` that tracks hold state.

    Used to assert that blocking I/O runs only when the lock is released.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._held = False

    def acquire(self, *a, **kw):
        r = self._lock.acquire(*a, **kw)
        if r:
            self._held = True
        return r

    def release(self):
        self._held = False
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()

    @property
    def held(self) -> bool:
        return self._held


def _db_with_rows(rows: dict) -> MagicMock:
    """Mock SessionDB where ``get_session`` maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    # Identity compression tip (no child session).
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _make_store(tmp_path, db_mock=None) -> SessionStore:
    """Build a SessionStore with a ``_TrackedLock``, bypassing disk load."""
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    if db_mock is not None:
        store._db = db_mock
    store._loaded = True
    store._lock = _TrackedLock()
    return store


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )


def _seed_entry(store, key, session_id) -> SessionEntry:
    now = datetime.now()
    entry = SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    store._entries[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStaleCheckOutsideLock:
    def test_is_session_ended_not_holding_lock(self, tmp_path):
        """``_is_session_ended_in_db`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({
            "sid_alive": {"end_reason": None, "id": "sid_alive"},
        })
        store = _make_store(tmp_path, db)
        key = store._generate_session_key(source)
        _seed_entry(store, key, "sid_alive")

        lock = store._lock
        calls_under_lock = []

        orig = store._is_session_ended_in_db

        def tracking(sid):
            if lock.held:
                calls_under_lock.append(sid)
            return orig(sid)

        store._is_session_ended_in_db = tracking  # type: ignore[method-assign]

        store.get_or_create_session(source)

        assert not calls_under_lock, (
            f"_is_session_ended_in_db called {len(calls_under_lock)} "
            f"time(s) while lock was held"
        )


class TestSaveOutsideLock:
    def test_save_not_holding_lock(self, tmp_path):
        """``_save`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({})
        store = _make_store(tmp_path, db)

        lock = store._lock
        save_calls_under_lock = []

        orig_save = store._save

        def tracking_save():
            if lock.held:
                save_calls_under_lock.append(True)
            orig_save()

        store._save = tracking_save  # type: ignore[method-assign]

        # force_new bypasses the existing-entry path, goes straight to create.
        store.get_or_create_session(source, force_new=True)

        assert not save_calls_under_lock, (
            f"_save called {len(save_calls_under_lock)} time(s) "
            f"while lock was held"
        )


class TestRecoverOutsideLock:
    def test_recover_not_holding_lock(self, tmp_path):
        """``_recover_session_from_db`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recovered",
            "started_at": datetime.now().timestamp(),
        }
        store = _make_store(tmp_path, db)
        # No entry seeded -- forces the recovery path.

        lock = store._lock
        recover_calls_under_lock = []

        orig = store._recover_session_from_db

        def tracking(**kw):
            if getattr(lock, "held", False):
                recover_calls_under_lock.append(True)
            return orig(**kw)

        store._recover_session_from_db = tracking  # type: ignore[method-assign]

        store.get_or_create_session(source)

        assert not recover_calls_under_lock, (
            f"_recover_session_from_db called "
            f"{len(recover_calls_under_lock)} time(s) while lock was held"
        )
