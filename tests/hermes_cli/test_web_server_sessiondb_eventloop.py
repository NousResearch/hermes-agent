"""Dashboard session routes must run store open + work off the event loop.

Regression for #60747: /search, /stats, /{id} (and bulk delete) ran FTS and
SQLite inline on the loop thread, so a large state.db froze every other
dashboard request (rename too). These drive the real route handlers and record which
thread every store call lands on.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.sessions as _rt_sessions
import hermes_cli.web_server_sessions as _web_server_sessions


def test_bulk_delete_sessiondb_work_runs_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []
    db_modes: list[bool] = []

    def _open_db(profile=None, *, read_only):
        assert profile is None
        db_modes.append(read_only)
        return store

    def _delete_many(ids, **kwargs):
        from persistence.conversation.sessions import DeleteManyOutcome

        db_threads.append(threading.get_ident())
        assert ids == ["one", "two"]
        return DeleteManyOutcome(deleted=2, skipped_guarded=(), removed_ids=("one", "two"))

    def _after_removed(store, store_home, removed_ids, **kwargs):
        db_threads.append(threading.get_ident())

    import session_retention

    store = SimpleNamespace(sessions=SimpleNamespace(delete_many=_delete_many), scope=SimpleNamespace(profile_dir=None),
                            close=lambda: db_threads.append(threading.get_ident()))
    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(session_retention, "after_sessions_removed", _after_removed)

    result = asyncio.run(
        _rt_sessions.bulk_delete_sessions_endpoint(
            _web_models.BulkDeleteSessions(ids=["one", "two"])
        )
    )

    assert result == {"ok": True, "deleted": 2, "skipped_active": []}
    assert db_modes == [False]
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


class _ReadDB:
    """Read-only stand-in that records which thread every store call runs on."""

    def __init__(self, threads: list[int]):
        self._threads = threads

    def _hit(self, value=None):
        self._threads.append(threading.get_ident())
        return value

    def get_session(self, sid):
        return self._hit({"id": sid, "title": "t"})

    def close(self):
        self._hit()


class _ReadStore(_ReadDB):
    """Storage interface stand-in that records which thread every verb runs on."""

    def __init__(self, threads: list[int]):
        super().__init__(threads)
        hit = self._hit
        self.sessions = SimpleNamespace(
            count_sessions=lambda *_a: hit(3), count_by_source=lambda **_kw: hit({"cli": 3}),
            get=lambda sid: hit(_record(session_id=sid, title="t", expiry_finalized=False, archived=False,
                                        auto_archived=False, pinned=False, hidden=False)),
            summary=lambda sid: hit(None),
            search_by_id=lambda q, **_kw: hit((_entry(preview="hello", session_id="sess-1", started_at=1.0),)))
        self.messages = SimpleNamespace(count_all=lambda: hit(7))
        self.lineage = SimpleNamespace(tip=lambda sid: hit(sid))
        self.search = SimpleNamespace(search_messages=lambda *_a, **_kw: hit(()))


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
        return _ReadStore(db_threads)

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_rt_sessions, "_resolve_session_id", lambda store, sid: sid)

    asyncio.run(call())

    assert db_threads, "handler never touched the store"
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_session_rename_runs_writer_open_and_update_off_event_loop(monkeypatch, tmp_path):
    """Main's rename fix (1769024ca3b) ported to the unified runtime: the PATCH still opens no
    second store writer (the owning authority applies the ``sidebar`` mutation), and that
    authority's SQLite transaction runs off the event loop thread."""
    from types import SimpleNamespace

    from persistence.sqlite.authority import _mutation as _mutations
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor

    def _open_db(*_args, **_kwargs):
        raise AssertionError("rename must not open its own store writer")

    write_threads: list[int] = []
    commit = _mutations.mutate_runtime_session

    def _recording_commit(*args, **kwargs):
        write_threads.append(threading.get_ident())
        return commit(*args, **kwargs)

    persistence = storage_handle(tmp_path)
    seed_session(persistence, "sess-1", "cli")
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="p", instance_id="owner",
                                 epoch=persistence.runtime.begin_epoch(instance_id="owner"),
                                 persistence=persistence)
    authority.sessions["sess-1"] = LiveSession(None, "route")
    actor = SessionActor("owner", "p", frozenset({"session:read", "session:control"}), "http")
    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_web_server_sessions, "_session_mutation_context", lambda request, profile: (authority, actor))
    monkeypatch.setattr(_mutations, "mutate_runtime_session", _recording_commit)

    async def _rename():
        body = _web_models.SessionRename(title="renamed", request_id="req-1", expected_revision=0)
        return threading.get_ident(), await _rt_sessions.rename_session_endpoint("sess-1", body, request=object())

    loop_thread, result = asyncio.run(_rename())
    assert result["ok"] is True and result["title"] == "renamed"
    assert persistence.sessions.get("sess-1").title == "renamed"
    assert write_threads and all(thread_id != loop_thread for thread_id in write_threads)


def _record(**fields):
    """A real ``SessionRecord`` with ``fields`` set and every other field unset."""
    from dataclasses import fields as _fields

    from persistence.conversation.sessions import SessionRecord

    return SessionRecord(**{**{f.name: None for f in _fields(SessionRecord)}, **fields})


def _entry(preview="", last_active=0.0, **fields):
    """A real ``SessionListEntry`` over ``_record(**fields)``."""
    from persistence.conversation.sessions import SessionListEntry

    return SessionListEntry(session=_record(**fields), preview=preview, unread=False, last_active=last_active)
