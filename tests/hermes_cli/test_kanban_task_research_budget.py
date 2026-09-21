"""Task-scoped research-budget persistence and worker-envelope contracts.

The policy is a typed task field, not task-body metadata: native DB/tool/CLI
creation validates it, readback exposes it, and the dispatcher sends it only to
the worker process for that card.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    return home


_POLICY = {
    "web_search_max": 8,
    "browser_extract_max": 4,
    "collection_deadline_seconds": 360,
    "synthesis_reserve_seconds": 180,
    "collection_tools": ["web_search", "web_extract"],
}


def test_task_policy_round_trips_through_db_and_native_show(kanban_home):
    from tools import kanban_tools

    created = json.loads(kanban_tools._handle_create({
        "title": "native bounded research", "assignee": "researcher",
        "research_budget": _POLICY,
    }))
    assert created["research_budget"] == _POLICY

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="bounded research", assignee="researcher", research_budget=_POLICY,
        )
        task = kb.get_task(conn, task_id)
        raw = conn.execute(
            "SELECT research_budget FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()["research_budget"]
        shown = json.loads(kanban_tools._handle_show({"task_id": task_id}))

    assert task is not None
    assert task.research_budget == _POLICY
    assert json.loads(raw) == _POLICY
    assert shown["task"]["research_budget"] == _POLICY


def test_task_policy_is_validated_and_malformed_legacy_rows_fail_closed(kanban_home):
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="positive integer"):
            kb.create_task(
                conn, title="invalid", assignee="researcher",
                research_budget={"web_search_max": 0},
            )
        task_id = kb.create_task(conn, title="legacy", assignee="researcher")
        conn.execute(
            "UPDATE tasks SET research_budget = ? WHERE id = ?",
            (json.dumps({"unexpected": 1}), task_id),
        )
        conn.commit()
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.research_budget is None


def test_legacy_migration_adds_task_research_budget_column(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES ('legacy', 'old', 'ready', 1)"
    )
    conn.commit()

    kbc._migrate_add_optional_columns(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "research_budget" in columns
    row = conn.execute("SELECT research_budget FROM tasks WHERE id = 'legacy'").fetchone()
    assert row["research_budget"] is None
    conn.close()


def test_cli_create_parses_and_persists_research_budget(kanban_home, capsys):
    args = argparse.Namespace(
        title="cli bounded research", body=None, assignee="researcher", created_by="test",
        workspace=None, branch=None, project=None, tenant=None, priority=0, parent=[],
        triage=False, idempotency_key=None, max_runtime=None, research_budget=json.dumps(_POLICY),
        skills=[], max_retries=None, model_override=None, provider_override=None,
        goal_mode=False, goal_max_turns=None, completion_contract=None,
        initial_status="running", json=True,
    )

    assert kanban_cli._cmd_create(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["research_budget"] == _POLICY


def test_dashboard_create_and_readback_preserves_research_budget(kanban_home):
    import importlib.util
    import sys

    plugin_path = Path(__file__).parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_budget_test", plugin_path,
    )
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)

    created = plugin.create_task(
        plugin.CreateTaskBody(
            title="dashboard bounded research", assignee="researcher", research_budget=_POLICY,
        ),
        board=None,
    )
    task_id = created["task"]["id"]
    shown = plugin.get_task(task_id, board=None, run_state_type=None, run_state_name=None)

    assert created["task"]["research_budget"] == _POLICY
    assert shown["task"]["research_budget"] == _POLICY


def test_dispatcher_passes_only_the_task_policy_to_worker(kanban_home, monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        pid = 41234

    def fake_popen(_cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setenv("HERMES_KANBAN_RESEARCH_BUDGET", "stale-dispatcher-value")

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="spawn bounded", assignee="worker", research_budget=_POLICY,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace = tmp_path / "worker"
        workspace.mkdir()
        assert dispatch._default_spawn(task, str(workspace)) == 41234

    assert captured["env"]["HERMES_KANBAN_RESEARCH_BUDGET"] == json.dumps(
        _POLICY, separators=(",", ":")
    )

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="spawn profile default", assignee="worker")
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace = tmp_path / "worker-default"
        workspace.mkdir()
        assert dispatch._default_spawn(task, str(workspace)) == 41234

    assert "HERMES_KANBAN_RESEARCH_BUDGET" not in captured["env"]
