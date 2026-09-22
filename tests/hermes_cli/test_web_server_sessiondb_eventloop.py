import ast
import asyncio
import threading
from pathlib import Path

import pytest

from hermes_cli import web_server
import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.sessions as _rt_sessions
import hermes_cli.web_server_sessions as _web_server_sessions
from hermes_cli import web_server_sessions
from hermes_cli.web_routers import analytics as web_analytics
from hermes_cli.web_routers import sessions as web_sessions


TARGET_HANDLERS = {
    "bulk_delete_sessions_endpoint",
    "count_empty_sessions_endpoint",
    "delete_empty_sessions_endpoint",
    "get_session_latest_descendant",
    "get_session_messages",
    "delete_session_endpoint",
    "export_session_endpoint",
    "prune_sessions_endpoint",
    "get_usage_analytics",
    "get_models_analytics",
    "search_sessions",
    "get_session_stats",
    "get_session_detail",
}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_sessiondb_handlers_open_connections_inside_executor_helpers():
    # The session and analytics route handlers were extracted to
    # web_routers/{sessions,analytics}.py; the executor helpers live in
    # web_server_sessions.py (and any left in web_server.py) — scan all
    # four modules' top-level bodies.
    handlers: dict[str, ast.AsyncFunctionDef] = {}
    top_level_helpers: dict[str, ast.FunctionDef] = {}
    for mod in (web_server, web_server_sessions, web_sessions, web_analytics):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name in TARGET_HANDLERS:
                handlers[node.name] = node
            elif isinstance(node, ast.FunctionDef):
                top_level_helpers[node.name] = node
    assert handlers.keys() == TARGET_HANDLERS

    for name, handler in handlers.items():
        helpers = {
            **top_level_helpers,
            **{
                node.name: node
                for node in handler.body
                if isinstance(node, ast.FunctionDef)
            },
        }
        offloaded = {
            arg.id
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and _call_name(node) == "to_thread"
            for arg in node.args[:1]
            if isinstance(arg, ast.Name)
        }
        db_open_owners = {
            helper_name
            for helper_name, helper in helpers.items()
            if helper_name in offloaded
            and any(
                isinstance(node, ast.Call)
                and _call_name(node) == "_open_session_db_for_profile"
                for node in ast.walk(helper)
            )
        }
        assert db_open_owners, f"{name} does not offload SessionDB open + work"


def test_sessiondb_opens_declare_access_mode():
    for mod in (web_server_sessions, web_sessions):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _call_name(node) == "_open_session_db_for_profile"
        ]
        assert calls
        for call in calls:
            assert any(keyword.arg == "read_only" for keyword in call.keywords)


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


def test_session_rename_runs_writer_open_and_update_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def set_session_title(self, sid, title):
            db_threads.append(threading.get_ident())
            assert (sid, title) == ("sess-1", "renamed")

        def get_session_title(self, sid):
            db_threads.append(threading.get_ident())
            assert sid == "sess-1"
            return "renamed"

        def close(self):
            db_threads.append(threading.get_ident())

    def _open_db(profile=None, *, read_only):
        db_threads.append(threading.get_ident())
        assert profile is None
        assert read_only is False
        return _DB()

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_rt_sessions, "_resolve_session_id", lambda db, sid: sid)

    result = asyncio.run(
        _rt_sessions.rename_session_endpoint(
            "sess-1", _web_models.SessionRename(title="renamed")
        )
    )

    assert result == {"ok": True, "title": "renamed"}
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
