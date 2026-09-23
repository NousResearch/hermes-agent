"""Regression: gateway state.db connection lifecycle must never grow (#96027).

The issue: a long-lived gateway process leaked SQLite connections on
``~/.hermes/state.db`` (48 db + 46 wal fds after 22 days, ~2 connections/day)
until every subsystem started failing with ``[Errno 24] Too many open files``.

Root-cause class: read paths that open a fresh ``SessionDB`` per call site and
never close it — the ``session_search`` tool's default handle, the
cross-profile read handle, ``react_to_message``'s handle, and any handle that
loses the gateway's per-path handle-cache race. Each leaked instance pins
itself via the token-writer atexit hook once it starts writing, so GC cannot
reclaim it — the fd lives until process exit.

These tests are the fd-count regression guard the issue asks for:

- open/close pairing via a tracked ``sqlite3.connect`` (the repo's established
  leak-test pattern — see ``test_async_delegation_fd_leak.py``): every
  connection opened by a public read path must be closed by that path,
  deterministically, without relying on interpreter GC;
- live-connection counting via the tracked-connection registry
  (``live_connection_count``), the cross-platform analogue of counting
  ``/proc/self/fd`` entries;
- the housekeeping fd-growth guard itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hermes_state
from gateway.session_db_recovery import RecoverableHandleCache


def _track_connections(monkeypatch):
    """Record every sqlite3 connection opened and closed by the state layer.

    SessionDB opens via ``hermes_state._connect_tracked_db`` →
    ``connect_fn=hermes_state.sqlite3.connect``, so patching that attribute
    records opens. Closes are recorded through two seams:

    - tracked connections (the ``connect_tracked`` factory) all go through
      ``TrackedConnection.close`` — patching that class records them;
    - plain ``sqlite3.connect`` call sites (the schema ``:memory:`` reference
      database in ``hermes_state_schema``, capability probes) close a plain
      ``sqlite3.Connection``, whose C-level ``close`` cannot be patched — so
      those are wrapped in a delegating recorder (safe: they never pass
      through ``connect_tracked``'s tracking retrofit).

    Every opened connection is also PINNED (held in a strong reference for the
    duration of the test) and ``SessionDB.__del__`` is neutralised. Together
    these model production: once a SessionDB's token-writer thread starts, the
    instance pins itself via ``atexit.register`` and interpreter GC can never
    reclaim it — the exact reason the v0.20.0 leak (open-without-close)
    survived to EMFILE instead of being quietly cleaned up at function return.
    A close() in the code under test still runs against a pinned connection,
    so the pairing assertion measures the ownership contract, not GC luck.

    Returns (opened_ids, closed_ids).
    """
    from hermes_cli import sqlite_safe_read as ssr

    opened, closed = [], []
    pinned = []
    real_connect = hermes_state.sqlite3.connect
    tracked_cls = ssr._tracking_factory(sqlite3.Connection)
    real_close = tracked_cls.close

    monkeypatch.setattr(hermes_state.SessionDB, "__del__", lambda self: None)

    class _CloseRecorder:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def close(self):
            closed.append(id(self._real))
            self._real.close()

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._real.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def __setattr__(self, name, value):
            setattr(self._real, name, value)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        pinned.append(conn)  # model the production atexit pin
        if isinstance(conn, ssr._TrackingMixin):
            return conn  # close is recorded via the patched TrackedConnection.close
        return _CloseRecorder(conn)

    def tracking_close(self):
        closed.append(id(self))
        return real_close(self)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracking_connect)
    monkeypatch.setattr(tracked_cls, "close", tracking_close)
    return opened, closed


def _assert_all_closed(opened, closed):
    assert opened, "expected the read path to open at least one connection"
    assert len(opened) == len(closed), (
        f"{len(opened)} connections opened but only {len(closed)} closed: "
        f"leaked {len(opened) - len(closed)} state.db handles (#96027)"
    )
    assert set(opened) == set(closed)


# ── The v0.20.0 leak: tool-level read paths that opened without closing ────

def test_session_search_closes_owned_default_db(monkeypatch):
    """session_search with no caller-supplied db must close the one it opens.

    Red on the v0.20.0 tool code (the default SessionDB() was dropped without
    close() on every return path); green on the fixed code.
    """
    opened, closed = _track_connections(monkeypatch)
    from tools.session_search_tool import session_search

    session_search(query="probe", limit=1)

    _assert_all_closed(opened, closed)


def test_session_search_cross_profile_closes_both_handles(monkeypatch, tmp_path):
    """A cross-profile read must close both the default and the profile handle.

    Red on the v0.20.0 tool code (the profile's read-only SessionDB was never
    closed and the default handle was dropped too).
    """
    # Create a second profile whose state.db exists and is initialised
    # (read-only opens require an existing store). Pin the profiles root to
    # the test tmpdir (established pattern, see test_control_socket.py).
    profiles_root = tmp_path / "profiles"
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
    )
    profile_home = profiles_root / "p2"
    profile_home.mkdir(parents=True)
    opener = hermes_state.SessionDB(db_path=profile_home / "state.db")
    opener.close()

    opened, closed = _track_connections(monkeypatch)
    from tools.session_search_tool import session_search

    session_search(query="probe", limit=1, profile="p2")

    _assert_all_closed(opened, closed)


def test_react_to_message_closes_owned_handle(monkeypatch):
    """react_to_message_tool must close the SessionDB it opens itself.

    Red on the v0.20.0 tool code, which contained no close() at all.
    """
    monkeypatch.setenv("HERMES_SESSION_KEY", "s1")
    opened, closed = _track_connections(monkeypatch)
    from tools.react_to_message_tool import react_to_message_tool

    react_to_message_tool(emoji="👍")

    _assert_all_closed(opened, closed)


# ── Live-connection-count invariants on the fixed code ─────────────────────

def test_gateway_turns_keep_live_connection_count_flat(monkeypatch, tmp_path):
    """The gateway's hot session-store path must not grow connections.

    Warm-up may open the writer + first pooled reader; once warm, the count
    must stay flat turn after turn, and close_all_db_handles() must drain it.
    """
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_cli.sqlite_safe_read import live_connection_count

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    db_path = tmp_path / "hermes_test" / "state.db"
    # SessionStore resolves HERMES_HOME/state.db; find whichever path it used.
    candidates = [
        tmp_path / "hermes_test" / "state.db",
        tmp_path / "state.db",
    ]
    target = next((p for p in candidates if live_connection_count(p) > 0), None)
    assert target is not None, "SessionStore should hold a live state.db connection"

    def one_turn(i):
        from gateway.session import Platform, SessionSource

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=f"chat-{i % 3}",
            user_id="u1",
        )
        entry = store.get_or_create_session(source=source)
        store.get_session_metadata(entry.session_key, "k")
        store.list_sessions(active_minutes=60)

    for i in range(6):
        one_turn(i)
    warm = live_connection_count(target)
    assert warm >= 1
    for i in range(6, 15):
        one_turn(i)
        assert live_connection_count(target) == warm, (
            f"gateway turn {i} grew state.db connections {warm} -> "
            f"{live_connection_count(target)} (#96027)"
        )

    store.close_all_db_handles()
    assert live_connection_count(target) == 0


def test_sessiondb_read_loop_flat_then_close_drains(monkeypatch, tmp_path):
    """SessionDB read paths reuse the pool; close() drains every connection."""
    from hermes_cli.sqlite_safe_read import live_connection_count

    db_path = tmp_path / "state.db"
    db = hermes_state.SessionDB(db_path=db_path)
    db.create_session(session_id="s1", source="telegram", model="m")
    db.append_message(session_id="s1", role="user", content="hello probe")
    db.append_message(session_id="s1", role="assistant", content="hi there")

    for i in range(5):
        db.search_messages(query="probe", limit=3)
        db.list_sessions_rich(limit=5)
        db.get_session("s1")
        db.get_messages("s1")
    warm = live_connection_count(db_path)
    assert warm >= 1

    for i in range(10):
        db.search_messages(query="probe", limit=3)
        db.get_anchored_view(session_id="s1", around_message_id=1, window=3)
        db.session_count()
        assert live_connection_count(db_path) == warm, (
            f"read iteration {i} grew state.db connections (#96027)"
        )

    db.close()
    assert live_connection_count(db_path) == 0


# ── Gateway handle-cache rejection must close, never drop ──────────────────

class _RecordingHandle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_rejected_handle_is_closed_even_without_close_all_callback():
    """A handle that loses the cache race must be closed, not dropped.

    If the open completes after a concurrent close_all() bumped the cache
    generation, the finished handle is rejected. The rejection path must
    release it even when no close_all callback is installed (the pre-shutdown
    window) — an open SessionDB dropped on the floor keeps its state.db fds
    for the life of the process (#96027).
    """
    cache = RecoverableHandleCache()
    path = Path("p/state.db")
    handle = _RecordingHandle()

    def opener():
        # A concurrent close_all() bumps the generation mid-open.
        with cache.lock:
            cache._generation += 1
        return handle

    assert cache.get(path, opener) is None  # rejected as stale
    assert handle.closed, "rejected handle was dropped without close() (#96027)"


def test_rejected_handle_uses_close_all_callback_when_installed():
    """With a close_all callback installed, rejection goes through it."""
    cache = RecoverableHandleCache()
    path = Path("p/state.db")
    handle = _RecordingHandle()
    closed_via_callback = []

    def closer(h):
        closed_via_callback.append(h)
        h.close()

    cache.close_all(closer)

    def opener():
        with cache.lock:
            cache._generation += 1
        return handle

    assert cache.get(path, opener) is None
    assert closed_via_callback == [handle]
    assert handle.closed


def test_rejected_handle_close_failure_is_swallowed():
    """A raising closer must not propagate out of get()."""
    cache = RecoverableHandleCache()
    path = Path("p/state.db")

    class _Boom:
        def close(self):
            raise RuntimeError("boom")

    def opener():
        with cache.lock:
            cache._generation += 1
        return _Boom()

    assert cache.get(path, opener) is None  # must not raise


# ── The fd-count helper and the housekeeping guard ─────────────────────────

def test_live_connection_count_and_growth_guard(monkeypatch, tmp_path):
    """live_connection_count tracks opens/closes; the guard logs growth."""
    from gateway.run import (
        _fd_guard_last,
        _fd_guard_warn_ceiling,
        _housekeeping_state_db_fd_guard,
    )
    from hermes_cli.sqlite_safe_read import live_connection_count

    db_path = tmp_path / "state.db"
    key = str(db_path.resolve())
    assert live_connection_count(db_path) == 0

    db = hermes_state.SessionDB(db_path=db_path)
    assert live_connection_count(db_path) == 1
    db.close()
    assert live_connection_count(db_path) == 0

    # Point the guard at this tmpdb (it normally resolves HERMES_HOME/state.db).
    monkeypatch.setattr("hermes_state._default_db_path", lambda: db_path)

    _fd_guard_last.clear()
    _housekeeping_state_db_fd_guard()  # seeds the baseline
    _housekeeping_state_db_fd_guard()  # no growth -> silent

    db = hermes_state.SessionDB(db_path=db_path)  # simulate a leaked handle
    try:
        _housekeeping_state_db_fd_guard()  # growth -> INFO, never raises
        assert _fd_guard_last[key] == 1
    finally:
        db.close()
    _housekeeping_state_db_fd_guard()
    assert _fd_guard_last[key] == 0

    # Above the ceiling -> WARNING (observed via the log, not an exception).
    monkeypatch.setattr("gateway.run._fd_guard_warn_ceiling", lambda: -1)
    _housekeeping_state_db_fd_guard()
    assert _fd_guard_warn_ceiling() > 0  # the real ceiling is a positive int
