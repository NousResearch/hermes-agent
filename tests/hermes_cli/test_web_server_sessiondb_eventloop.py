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
