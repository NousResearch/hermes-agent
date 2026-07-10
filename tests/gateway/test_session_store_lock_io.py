"""Regression: blocking I/O must not run while session_store._lock is held.

``get_or_create_session`` previously held the store lock during SQLite
SELECTs (``_is_session_ended_in_db``), a full routing-index rewrite +
``os.fsync`` (``_save``), and a recovery DB query
(``_recover_session_from_db``) -- all on every inbound message.

These tests assert those three I/O calls are invoked *outside* the lock.
They follow the mock-DB idiom from ``test_session_store_runtime_stale_guard``.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
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

        orig_save = store._save_entries

        def tracking_save():
            if lock.held:
                save_calls_under_lock.append(True)
            orig_save()

        store._save_entries = tracking_save  # type: ignore[method-assign]

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

        orig = store._query_recoverable_session

        def tracking(**kw):
            if getattr(lock, "held", False):
                recover_calls_under_lock.append(True)
            return orig(**kw)

        store._query_recoverable_session = tracking  # type: ignore[method-assign]

        store.get_or_create_session(source)

        assert not recover_calls_under_lock, (
            f"_recover_session_from_db called "
            f"{len(recover_calls_under_lock)} time(s) while lock was held"
        )


def test_concurrent_same_key_returns_one_published_session(tmp_path):
    """Concurrent first messages for one routing key must converge on one ID."""
    source = _source()
    db = _db_with_rows({})
    store = _make_store(tmp_path, db)
    barrier = threading.Barrier(2)
    original_query = store._query_recoverable_session

    def synchronized_query(**kwargs):
        barrier.wait(timeout=2)
        return original_query(**kwargs)

    store._query_recoverable_session = synchronized_query  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(pool.map(lambda _: store.get_or_create_session(source), range(2)))

    key = store._generate_session_key(source)
    assert entries[0] is entries[1]
    assert entries[0].session_id == store._entries[key].session_id
    created_ids = {call.kwargs["session_id"] for call in db.create_session.call_args_list}
    assert created_ids == {entries[0].session_id}


def test_save_serialization_snapshots_latest_routing_index(tmp_path):
    """A delayed earlier writer must snapshot the state visible when it writes."""
    db = _db_with_rows({})
    persisted: dict[str, str] = {}
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    write_count = 0
    count_lock = threading.Lock()

    def replace(entries, *, scope):
        nonlocal write_count, persisted
        with count_lock:
            write_count += 1
            call_number = write_count
        if call_number == 1:
            first_write_started.set()
            assert release_first_write.wait(timeout=2)
        persisted = dict(entries)

    db.replace_gateway_routing_entries.side_effect = replace
    store = _make_store(tmp_path, db)
    source_a = _source()
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="67890",
        chat_type="dm",
        user_id="67890",
    )
    key_a = store._generate_session_key(source_a)
    key_b = store._generate_session_key(source_b)
    entry_a = _seed_entry(store, key_a, "sid-a")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(store._save_entries)
        assert first_write_started.wait(timeout=2)
        entry_b = _seed_entry(store, key_b, "sid-b")
        future_b = pool.submit(store._save_entries)
        release_first_write.set()
        future_a.result(timeout=2)
        future_b.result(timeout=2)

    assert set(store._entries) == {key_a, key_b}
    assert set(persisted) == {key_a, key_b}
    assert json.loads(persisted[key_a])["session_id"] == entry_a.session_id
    assert json.loads(persisted[key_b])["session_id"] == entry_b.session_id


def test_recovery_rejects_other_profile_row(tmp_path, monkeypatch):
    """The lock-free recovery path must retain the canonical profile guard."""
    source = _source()
    db = _db_with_rows({})
    db.find_latest_gateway_session_for_peer.return_value = {
        "id": "foreign-session",
        "session_key": "agent:other:telegram:dm:12345",
        "started_at": datetime.now().timestamp(),
    }
    store = _make_store(tmp_path, db)
    monkeypatch.setattr(store, "_active_profile_name", lambda: "default")

    entry = store.get_or_create_session(source)

    assert entry.session_id != "foreign-session"
    db.reopen_session.assert_not_called()
