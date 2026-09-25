"""Dashboard session routes must run SessionDB open + work off the event loop.

Regression for #60747: /search, /stats, /{id} (and bulk delete) ran FTS and
SQLite inline on the loop thread, so a large state.db froze every other
dashboard request (rename too). These drive the real route handlers and record which
thread every SessionDB call lands on.
"""

import asyncio
import threading

import pytest

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.sessions as _rt_sessions
import hermes_cli.web_server_sessions as _web_server_sessions


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


class _ReadDB:
    """Read-only stand-in that records which thread every SessionDB call runs on."""

    def __init__(self, threads: list[int]):
        self._threads = threads

    def _hit(self, value=None):
        self._threads.append(threading.get_ident())
        return value

    def session_count(self, **_kw):
        return self._hit(3)

    def message_count(self):
        return self._hit(7)

    def session_count_by_source(self, **_kw):
        return self._hit({"cli": 3})

    def get_session(self, sid):
        return self._hit({"id": sid, "title": "t"})

    def get_session_rich_row(self, sid):
        return self._hit(None)

    def get_compression_tip(self, root_id):
        return self._hit(None)

    def search_sessions_by_id(self, q, **_kw):
        return self._hit([{"id": "sess-1", "preview": "hello", "started_at": 1.0}])

    def search_messages(self, **_kw):
        return self._hit([])

    def close(self):
        self._hit()


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: _rt_sessions.search_sessions(q="hello"), id="search"),
        pytest.param(lambda: _rt_sessions.get_session_stats(), id="stats"),
        pytest.param(lambda: _rt_sessions.get_session_detail("sess-1"), id="detail"),
    ],
)
def test_session_read_handlers_run_sessiondb_work_off_event_loop(monkeypatch, call):
    """Regression for #60747: /search, /stats and /{id} ran FTS + SQLite inline on the
    loop thread, so a large state.db froze every other dashboard request."""
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    def _open_db(profile=None, *, read_only):
        assert read_only is True
        return _ReadDB(db_threads)

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_rt_sessions, "_resolve_session_id", lambda db, sid: sid)

    asyncio.run(call())

    assert db_threads, "handler never touched the SessionDB"
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_session_rename_routes_through_authority_without_second_writer(monkeypatch):
    """Main's rename fix offloaded the ad-hoc ``_with_db`` writer; on the unified runtime the
    PATCH route never opens a SessionDB at all — the owning session authority applies the
    ``sidebar`` mutation, so a second writer (on or off the loop) would be the regression."""
    seen = {}

    def _open_db(*_args, **_kwargs):
        raise AssertionError("rename must not open its own SessionDB writer")

    async def _mutate(request, profile, session_id, *, request_id, expected_revision, operation,
                      payload, expected_generation=None):
        seen.update(profile=profile, session_id=session_id, request_id=request_id,
                    expected_revision=expected_revision, operation=operation, payload=payload,
                    expected_generation=expected_generation)
        return {"ok": True, "title": payload.get("title")}

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_web_server_sessions, "_mutate_session_request", _mutate)

    body = _web_models.SessionRename(title="renamed", pinned=True, request_id="req-1",
                                     expected_revision=3, expected_generation=1)
    result = asyncio.run(_rt_sessions.rename_session_endpoint("sess-1", body, request=object()))

    assert result == {"ok": True, "title": "renamed"}
    assert seen == {
        "profile": None, "session_id": "sess-1", "request_id": "req-1", "expected_revision": 3,
        "expected_generation": 1, "operation": "sidebar", "payload": {"title": "renamed", "pinned": True},
    }
