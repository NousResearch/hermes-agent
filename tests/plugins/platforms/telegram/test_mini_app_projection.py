from __future__ import annotations

import asyncio
import importlib
import inspect
import sqlite3

from plugins.platforms.telegram.mini_app.projection import (
    normalize_swarm_board,
    project_catalog,
    project_sessions,
    project_status,
)

runtime = importlib.import_module("plugins.platforms.telegram.mini_app.app")
projection_module = importlib.import_module(
    "plugins.platforms.telegram.mini_app.projection"
)


def test_runtime_has_no_dashboard_or_network_adapter():
    source = inspect.getsource(runtime) + inspect.getsource(projection_module)
    assert "DashboardProjection" not in source
    assert "httpx" not in source
    assert "fetch_account_usage" not in source
    assert "127.0.0.1:9119" not in source


def test_session_projection_opens_database_read_only(tmp_path, monkeypatch):
    state_db = tmp_path / "state.db"
    state_db.touch()
    observed = {}

    class FakeSessionDB:
        def __init__(self, db_path, read_only=False):
            observed.update(db_path=db_path, read_only=read_only)

        def list_sessions_rich(self, **kwargs):
            observed["list_kwargs"] = kwargs
            return []

        def session_count(self, **kwargs):
            observed["count_kwargs"] = kwargs
            return 0

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(runtime, "HERMES_HOME", tmp_path)
    monkeypatch.setattr("hermes_state.SessionDB", FakeSessionDB)
    assert runtime._session_page(20, 0)["sessions"] == []
    assert observed["db_path"] == state_db
    assert observed["read_only"] is True
    assert observed["list_kwargs"]["compact_rows"] is True
    assert observed["closed"] is True


def test_kanban_projection_uses_read_only_sqlite_uri(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    connection = kb.connect(db_path=db_path)
    connection.close()
    monkeypatch.setattr(kb, "kanban_db_path", lambda: db_path)
    real_connect = sqlite3.connect
    observed = {}

    def recording_connect(database, *args, **kwargs):
        observed.update(database=database, kwargs=kwargs)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(runtime.sqlite3, "connect", recording_connect)
    board = runtime._kanban_board(False, "")
    assert "mode=ro" in observed["database"]
    assert observed["kwargs"]["uri"] is True
    assert board["columns"]


def test_live_usage_never_fetches_account_data(monkeypatch):
    monkeypatch.setattr(
        runtime, "_latest_session", lambda: {"id": "s1", "input_tokens": 4}
    )
    without = asyncio.run(runtime.live_usage(include_accounts=False))
    requested = asyncio.run(runtime.live_usage(include_accounts=True))
    assert without["accounts"] == requested["accounts"] == {}
    assert without["account"] is requested["account"] is None
    assert without["accounts_requested"] is False
    assert requested["accounts_requested"] is True
    assert requested["accounts_available"] is False


def test_profile_projection_never_enumerates_sibling_profile_homes(
    tmp_path, monkeypatch
):
    default_root = tmp_path / "root"
    active = default_root / "profiles" / "work"
    sibling = default_root / "profiles" / "secret-profile"
    sibling.mkdir(parents=True)
    (sibling / ".env").write_text("API_KEY=must-not-read\n")
    monkeypatch.setattr(runtime, "HERMES_HOME", active)
    monkeypatch.setattr(runtime, "get_default_hermes_root", lambda: default_root)
    monkeypatch.setattr(
        runtime,
        "_gateway_status_payload",
        lambda home: {"gateway_running": False},
    )

    assert runtime._profile_list() == [
        {
            "name": "work",
            "profile": "work",
            "active": True,
            "gateway": "stopped",
        }
    ]


def test_tool_and_skill_catalogs_do_not_load_credential_config(tmp_path, monkeypatch):
    skills = tmp_path / "skills" / "safe-skill"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: safe-skill\ndescription: Safe metadata\ncategory: local\n---\nBody\n"
    )
    (tmp_path / "config.yaml").write_text("api_key: must-not-leak\n")
    monkeypatch.setattr(runtime, "HERMES_HOME", tmp_path)
    skill_catalog = runtime._skill_catalog()
    tool_catalog = runtime._toolset_catalog()
    assert skill_catalog == [
        {"name": "safe-skill", "description": "Safe metadata", "category": "local"}
    ]
    assert tool_catalog
    assert all(
        "enabled" not in item and "configured" not in item for item in tool_catalog
    )
    assert "must-not-leak" not in repr(skill_catalog) + repr(tool_catalog)


def test_status_projection_drops_tokens_and_process_details():
    result = project_status({
        "version": "1.2.3",
        "api_key": "secret",
        "gateway_platforms": {
            "telegram": {
                "state": "connected",
                "substate": "running",
                "MainPID": "123",
                "token": "secret",
            }
        },
    })
    assert result == {
        "version": "1.2.3",
        "gateway_platforms": {
            "telegram": {"state": "connected", "substate": "running"}
        },
    }


def test_session_and_catalog_projections_are_field_allowlists():
    sessions = project_sessions({
        "sessions": [
            {
                "id": "s1",
                "title": "Session",
                "preview": "Hello",
                "message_count": 2,
                "api_key": "secret",
                "raw_messages": [{"role": "system"}],
            }
        ],
        "total": 1,
        "database_path": "/secret/state.db",
    })
    assert sessions == {
        "sessions": [
            {"id": "s1", "title": "Session", "preview": "Hello", "message_count": 2}
        ],
        "total": 1,
    }
    assert project_catalog(
        [{"name": "web", "enabled": True, "credential": "secret"}], "toolsets"
    ) == [{"name": "web", "enabled": True}]


def test_swarm_projection_counts_agents_and_drops_board_secrets():
    board = {
        "columns": [
            {
                "name": "running",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Run",
                        "assignee": "coder",
                        "credential": "secret",
                    }
                ],
            },
            {
                "name": "blocked",
                "tasks": [{"id": "t2", "title": "Blocked", "assignee": "coder"}],
            },
        ],
        "boards": [{"db_path": "/secret/kanban.db"}],
        "latest_event_id": 7,
    }
    result = normalize_swarm_board(board, [{"name": "coder"}, {"name": "idle"}])
    coder = next(item for item in result["agents"] if item["name"] == "coder")
    assert coder["assigned_count"] == 2
    assert coder["running_count"] == 1
    assert coder["blocked_count"] == 1
    assert result["summary"]["tasks"] == 2
    assert "secret" not in repr(result)
    assert "/secret/kanban.db" not in repr(result)
