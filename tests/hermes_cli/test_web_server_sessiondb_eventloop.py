import ast
import asyncio
import threading
from pathlib import Path

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
