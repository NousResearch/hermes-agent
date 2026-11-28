"""Task-scoped Kanban execution budgets and compact-read regressions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("goal_mode", [False, True])
def test_task_iteration_budget_is_snapshotted_for_profile_owner(kanban_home, goal_mode):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="bounded", assignee="worker", max_retries=1,
                                 max_iterations=3, goal_mode=goal_mode)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run = kb.get_run(conn, claimed.current_run_id)
        assert run is not None and run.max_iterations == 3
        assert claimed.goal_mode is goal_mode
        argv = kbd._worker_argv(claimed, "worker", None)
    assert argv == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]


def test_unset_iteration_budget_keeps_worker_argv_unchanged(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="neighbour", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.get_run(conn, claimed.current_run_id).max_iterations is None
        argv = kbd._worker_argv(claimed, "worker", None)
    assert argv == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0}, {"max_retries": 0}, {"max_iterations": -1},
    {"max_iterations": True}, {"max_iterations": 1.5}, {"max_retries": "2"},
])
def test_task_limits_reject_non_integer_values_before_dispatch(kanban_home, kwargs):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="positive integer"):
            kb.create_task(conn, title="bad", assignee="worker", **kwargs)


def test_retry_limit_blocks_after_first_failure(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="one try", assignee="worker", max_retries=1, max_iterations=2)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kbd._record_task_failure(conn, task_id, "synthetic failure", outcome="timed_out", failure_limit=9, release_claim=True, end_run=True) is True
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1


def test_set_task_limits_is_additive_and_rejects_zero(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="editable", assignee="worker")
        assert kb.set_task_limits(conn, task_id, max_retries=2, max_iterations=4)
        task = kb.get_task(conn, task_id)
        assert (task.max_retries, task.max_iterations) == (2, 4)
        with pytest.raises(ValueError, match="max_iterations"):
            kb.set_task_limits(conn, task_id, max_retries=2, max_iterations=0)


@pytest.mark.parametrize("value", [True, 1.5, "2", 0])
def test_set_limits_tool_rejects_non_integer_budget_before_db_write(kanban_home, monkeypatch, value):
    from tools import kanban_tools
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="strict-tool", assignee="worker")
    monkeypatch.setattr(kanban_tools, "_check_kanban_orchestrator_mode", lambda: True)
    out = json.loads(kanban_tools._handle_set_limits({"task_id": task_id, "max_iterations": value}))
    assert "error" in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).max_iterations is None


def test_create_tool_carries_budgets_atomically_into_claim_and_argv(kanban_home):
    from tools import kanban_tools
    created = json.loads(kanban_tools._handle_create({
        "title": "atomic limits", "assignee": "worker", "max_retries": 1, "max_iterations": 3,
    }))
    with kbc.connect() as conn:
        task = kb.get_task(conn, created["task_id"])
        assert (task.max_retries, task.max_iterations) == (1, 3)
        claimed = kb.claim_task(conn, task.id)
        assert claimed is not None
        argv = kbd._worker_argv(claimed, "worker", None)
    assert argv == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]


@pytest.mark.parametrize("value", [True, 1.5, "2", 0])
def test_create_tool_rejects_non_integer_budget_before_task_creation(kanban_home, value):
    from tools import kanban_tools
    rejected = json.loads(kanban_tools._handle_create({
        "title": "invalid atomic limits", "assignee": "worker", "max_iterations": value,
    }))
    assert "error" in rejected
    with kbc.connect() as conn:
        assert kb.list_tasks(conn) == []


def test_set_limits_tool_edits_existing_task_without_direct_sql(kanban_home, monkeypatch):
    from tools import kanban_tools
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="tool-edit", assignee="worker")
    monkeypatch.setattr(kanban_tools, "_check_kanban_orchestrator_mode", lambda: True)
    out = json.loads(kanban_tools._handle_set_limits({"task_id": task_id, "max_retries": 1, "max_iterations": 2}))
    assert out["max_retries"] == 1 and out["max_iterations"] == 2
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert (task.max_retries, task.max_iterations) == (1, 2)


def test_late_run_cannot_complete_a_newer_claim(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="fenced", assignee="worker", max_iterations=2)
        first = kb.claim_task(conn, task_id)
        assert first is not None
        assert kb.reclaim_task(conn, task_id, reason="synthetic retry")
        second = kb.claim_task(conn, task_id)
        assert second is not None and second.current_run_id != first.current_run_id
        assert kb.complete_task(conn, task_id, summary="late", expected_run_id=first.current_run_id) is False
        current = kb.get_task(conn, task_id)
        assert current.status == "running"
        assert current.current_run_id == second.current_run_id


def test_legacy_schema_adds_iteration_columns(tmp_path):
    path = tmp_path / "legacy.db"
    conn = kbc.sqlite3.connect(path)
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT, status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER, workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT, claim_lock TEXT, claim_expires INTEGER)")
    conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)")
    conn.execute("CREATE TABLE task_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, status TEXT NOT NULL, started_at INTEGER NOT NULL)")
    conn.commit(); conn.close()
    with kbc.connect(path) as migrated:
        assert "max_iterations" in {r["name"] for r in migrated.execute("PRAGMA table_info(tasks)")}
        assert "max_iterations" in {r["name"] for r in migrated.execute("PRAGMA table_info(task_runs)")}


def test_compact_show_keeps_latest_comment_and_pages_older_history(kanban_home):
    from tools import kanban_tools
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="history", body="complete original instructions", assignee="worker")
        for index in range(8):
            kb.add_comment(conn, task_id, "operator", f"correction {index}")
        for _ in range(3):
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            assert kb.reclaim_task(conn, task_id, reason="synthetic")
        current = kb.claim_task(conn, task_id)
        assert current is not None
    compact = json.loads(kanban_tools._handle_show({"task_id": task_id, "comment_limit": 2}))
    assert compact["task"]["body"] == "complete original instructions"
    assert [c["body"] for c in compact["comments"]] == ["correction 6", "correction 7"]
    assert "worker_context" not in compact
    cursor = compact["history"]["comments"]["next_before_id"]
    with kbc.connect() as conn:
        kb.add_comment(conn, task_id, "operator", "correction 8")
    older = json.loads(kanban_tools._handle_show({"task_id": task_id, "comment_limit": 3, "before_comment_id": cursor}))
    assert [c["body"] for c in older["comments"]] == ["correction 3", "correction 4", "correction 5"]
    oldest = json.loads(kanban_tools._handle_show({"task_id": task_id, "comment_limit": 3, "before_comment_id": older["history"]["comments"]["next_before_id"]}))
    assert [c["body"] for c in oldest["comments"]] == ["correction 0", "correction 1", "correction 2"]
    fresh = json.loads(kanban_tools._handle_show({"task_id": task_id, "comment_limit": 2}))
    assert [c["body"] for c in fresh["comments"]] == ["correction 7", "correction 8"]
    full = json.loads(kanban_tools._handle_show({"task_id": task_id, "detail": "full"}))
    assert len(full["comments"]) == 9
    assert full["runs"][-1]["id"] == current.current_run_id


def test_goal_mode_show_reports_the_actual_bounded_product(kanban_home):
    from tools import kanban_tools
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="goal", assignee="worker", goal_mode=True, goal_max_turns=4, max_iterations=3)
    shown = json.loads(kanban_tools._handle_show({"task_id": task_id}))
    assert shown["execution_budget"] == {
        "max_iterations_per_turn": 3, "goal_turns": 4, "max_api_calls_per_run": 12,
    }
