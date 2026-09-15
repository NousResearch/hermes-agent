"""SessionDB work must never run on the dashboard's event-loop thread.

Opening a SessionDB is blocking sqlite I/O. When a route handler did it
inline, one slow query stalled every other dashboard request sharing the
loop. Each handler below therefore offloads the open + work to a thread
and declares an explicit ``read_only`` access mode.

These are behavior tests: they drive the real handlers with a recording
stub in place of ``_open_session_db_for_profile`` and assert on the thread
the open actually happened on. Reading the handlers' source text instead
would pass on a mis-wired call site and fail on a correct refactor (see
"Never read source code in tests" in AGENTS.md).
"""

import asyncio
import threading
from typing import Optional
from unittest.mock import MagicMock

import pytest

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.analytics as _rt_analytics
import hermes_cli.web_routers.sessions as _rt_sessions
import hermes_cli.web_server_sessions as _web_server_sessions


def _drive(handler_call, monkeypatch):
    """Run *handler_call* with a recording DB opener.

    Returns (result, opens) where each open records the thread it ran on
    and the read_only mode it was given.
    """
    opens: list[dict] = []

    def _open_db(profile: Optional[str] = None, *, read_only: bool):
        opens.append({"thread": threading.get_ident(), "read_only": read_only})
        db = MagicMock()
        # Session-id resolution must yield a real string: handlers treat a
        # falsy id as "absent" and short-circuit before touching the DB.
        db.resolve_resume_session_id.return_value = "sid-1"
        db.count_empty_sessions.return_value = 0
        db.delete_empty_sessions.return_value = 0
        db.delete_sessions.return_value = 0
        return db

    monkeypatch.setattr(
        _web_server_sessions, "_open_session_db_for_profile", _open_db
    )

    loop_thread = threading.get_ident()
    try:
        result = asyncio.run(handler_call())
    except Exception as exc:  # noqa: BLE001 - 404s etc. are fine; we assert on opens
        result = exc
    return result, opens, loop_thread


# Every handler that touches SessionDB, with a minimal real invocation.
SESSION_HANDLERS = {
    "bulk_delete_sessions_endpoint": lambda: _rt_sessions.bulk_delete_sessions_endpoint(
        _web_models.BulkDeleteSessions(ids=["one", "two"])
    ),
    "count_empty_sessions_endpoint": lambda: _rt_sessions.count_empty_sessions_endpoint(),
    "delete_empty_sessions_endpoint": lambda: _rt_sessions.delete_empty_sessions_endpoint(),
    "get_session_latest_descendant": lambda: _rt_sessions.get_session_latest_descendant("sid-1"),
    # FastAPI Query(...) defaults are only resolved by the framework, so
    # direct calls must pass real values or the handler short-circuits on
    # validation before it ever reaches the DB.
    "get_session_messages": lambda: _rt_sessions.get_session_messages(
        "sid-1", None, None, 0, None, False
    ),
    "delete_session_endpoint": lambda: _rt_sessions.delete_session_endpoint("sid-1"),
    "export_session_endpoint": lambda: _rt_sessions.export_session_endpoint("sid-1"),
    "get_usage_analytics": lambda: _rt_analytics.get_usage_analytics(),
    "get_models_analytics": lambda: _rt_analytics.get_models_analytics(),
}


@pytest.mark.parametrize("name", sorted(SESSION_HANDLERS))
def test_sessiondb_open_runs_off_the_event_loop(name, monkeypatch):
    """The SessionDB open must happen on a worker thread, not the loop."""
    _result, opens, loop_thread = _drive(SESSION_HANDLERS[name], monkeypatch)

    assert opens, f"{name} never opened a SessionDB — stub not reached"
    off_loop = [o for o in opens if o["thread"] != loop_thread]
    assert len(off_loop) == len(opens), (
        f"{name} opened a SessionDB on the event-loop thread; one slow query "
        f"there stalls every other dashboard request."
    )


@pytest.mark.parametrize("name", sorted(SESSION_HANDLERS))
def test_sessiondb_opens_declare_access_mode(name, monkeypatch):
    """Every open must state read_only explicitly, so a read path can never
    silently take a write lock on the shared session store."""
    _result, opens, _loop = _drive(SESSION_HANDLERS[name], monkeypatch)

    assert opens, f"{name} never opened a SessionDB — stub not reached"
    for open_call in opens:
        assert isinstance(open_call["read_only"], bool), (
            f"{name} opened a SessionDB without an explicit read_only mode"
        )


def test_mutating_handlers_open_for_write():
    """Guards the pairing itself: a delete path taking a read-only handle
    would fail at runtime, so the mode is part of the contract."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        _result, opens, _loop = _drive(
            SESSION_HANDLERS["bulk_delete_sessions_endpoint"], monkeypatch
        )
    finally:
        monkeypatch.undo()

    assert opens
    assert any(o["read_only"] is False for o in opens), (
        "bulk delete must open the SessionDB for write"
    )


def test_bulk_delete_sessiondb_work_runs_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []
    db_modes: list[bool] = []

    class _DB:
        def delete_sessions(self, ids):
            db_threads.append(threading.get_ident())
            assert ids == ["one", "two"]
            return 2

        def close(self):
            db_threads.append(threading.get_ident())

    def _open_db(profile=None, *, read_only):
        assert profile is None
        db_modes.append(read_only)
        return _DB()

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)

    result = asyncio.run(
        _rt_sessions.bulk_delete_sessions_endpoint(
            _web_models.BulkDeleteSessions(ids=["one", "two"])
        )
    )

    assert result == {"ok": True, "deleted": 2}
    assert db_modes == [False]
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)
