"""Explicit review children consume PRs without weakening implementation guards."""
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "workspaces"))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


@pytest.mark.parametrize("step", ["review", "release"])
@pytest.mark.parametrize("gate", [None, "parents", "blocker_auth", "rate_limit_cooldown", "recent_success", "unlinked", "unknown_template", "unknown_step"])
def test_classified_child_dispatch_preserves_other_gates(board, monkeypatch, step, gate):
    parent = kb.create_task(board, title="upstream", assignee="builder")
    child = kb.create_task(board, title="downstream", assignee="arbitrary-profile",
                           parents=[parent], review_child_step=step)
    ordinary = kb.create_task(board, title="review release exact-SHA", assignee="reviewer",
                              parents=[parent], skills=["sdlc-review"])
    standalone = kb.create_task(board, title="implementation", assignee="builder")
    for tid in (child, ordinary, standalone):
        kb.add_comment(board, tid, author="builder", body="https://github.com/example/repo/pull/123")
    task = kb.get_task(board, child)
    assert (task.workflow_template_id, task.current_step_key) == ("hermes:review_child_v1", step)
    assert task.status == "todo"
    if gate != "parents":
        assert kb.complete_task(board, parent, summary="source ready")
    now = int(time.time())
    with kb.write_txn(board):
        if gate == "blocker_auth":
            board.execute("UPDATE tasks SET last_failure_error = 'invalid api key' WHERE id = ?", (child,))
        if gate in ("rate_limit_cooldown", "recent_success"):
            outcome = "rate_limited" if gate == "rate_limit_cooldown" else "completed"
            board.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'arbitrary-profile', ?, ?, ?, ?)",
                (child, outcome, outcome, now + 5, now + 5),
            )
        if gate == "unlinked":
            board.execute("DELETE FROM task_links WHERE child_id = ?", (child,))
        if gate == "unknown_template":
            board.execute("UPDATE tasks SET workflow_template_id = 'other' WHERE id = ?", (child,))
        if gate == "unknown_step":
            board.execute("UPDATE tasks SET current_step_key = 'implementation' WHERE id = ?", (child,))
    monkeypatch.setattr(kb, "_resolve_rate_limit_cooldown_seconds", lambda: 600)
    result = kbd.dispatch_once(board, dry_run=True)
    spawned = {entry[0] for entry in result.spawned}
    guarded = dict(result.respawn_guarded)
    assert (child in spawned) == (gate is None)
    assert ordinary not in spawned
    assert standalone not in spawned
    assert guarded[standalone] == "active_pr"
    if gate != "parents":
        assert guarded[ordinary] == "active_pr"
    if gate in ("blocker_auth", "rate_limit_cooldown", "recent_success"):
        assert guarded[child] == gate
    if gate in ("unlinked", "unknown_template", "unknown_step"):
        assert guarded[child] == "active_pr"
    if gate == "parents":
        assert kb.get_task(board, child).status == "todo"


@pytest.mark.parametrize("step,parents,error", [
    ("review", [], "requires at least one parent"),
    ("release", ["missing"], "unknown parent"),
    ("implement", ["parent"], "must be review or release"),
    ("", ["parent"], "must be review or release"),
    (True, ["parent"], "must be review or release"),
])
def test_invalid_classification_is_atomic(board, step, parents, error):
    parent = kb.create_task(board, title="upstream", assignee="builder")
    parents = [parent if p == "parent" else p for p in parents]
    before = board.execute("SELECT id FROM tasks").fetchall()
    with pytest.raises(ValueError, match=error):
        kb.create_task(board, title="downstream", assignee="worker", parents=parents, review_child_step=step)
    assert board.execute("SELECT id FROM tasks").fetchall() == before
