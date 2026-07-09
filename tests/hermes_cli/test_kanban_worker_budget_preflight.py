from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_goal_mode_zero_budget_blocks_before_spawn_without_retry_loss(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    def zero_budget_config():
        return {
            "goals": {"max_turns": 0},
            "agent": {"max_turns": 90},
            "delegation": {"max_iterations": 90},
        }

    monkeypatch.setattr("hermes_cli.config.load_config", zero_budget_config)

    spawned = []

    def spawn_fn(task, workspace):
        spawned.append(task.id)
        return 12345

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="zero budget worker",
            assignee="worker",
            goal_mode=True,
        )

        res = kb.dispatch_once(conn, spawn_fn=spawn_fn)

        assert spawned == []
        assert tid in res.spawn_blocked_zero_budget

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 0
        assert task.last_failure_error and "goals.max_turns" in task.last_failure_error

        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "spawn_blocked_zero_budget"
        metadata = run.metadata or {}
        assert metadata["budget_key"] == "goals.max_turns"

        events = kb.list_events(conn, tid)
        assert any(ev.kind == "spawn_blocked_zero_budget" for ev in events)
    finally:
        conn.close()


def test_zero_budget_block_has_operator_diagnostic(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import kanban_diagnostics as diag

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"max_turns": 0}, "agent": {"max_turns": 90}},
    )
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="zero budget diagnostic", assignee="worker", goal_mode=True
        )
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 999)
        task = kb.get_task(conn, tid)
        diagnostics = diag.compute_task_diagnostics(
            task, kb.list_events(conn, tid), kb.list_runs(conn, task_id=tid), now=task.created_at + 1
        )
        zero_budget = [d for d in diagnostics if d.kind == "zero_budget_worker"]
        assert zero_budget
        assert zero_budget[0].severity == "critical"
        assert "goals.max_turns" in zero_budget[0].detail
        commands = [a.payload.get("command", "") for a in zero_budget[0].actions]
        assert any("kanban repair zero-budget-failures --dry-run" in c for c in commands)
    finally:
        conn.close()
