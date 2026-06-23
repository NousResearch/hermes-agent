"""create_task workspace_path absolute-path guard + set_workspace_path opt-in reset.

Pins the §4.1 + §4.2 + §4.3 fix for incident crash-rate-totum-operator
2026-06-23 (kanban t_da44022e). The dispatcher already validated
``workspace_path`` at spawn via ``resolve_workspace``, but the bad row
was being created in the first place — the worker crashed BEFORE the
spawn-time guard could catch it, so the row sat in ``ready``
indefinitely. This test module pins the create-time guard.

Also pins the §4.2 deviation: the spec said "unconditional reset" of
``consecutive_failures`` inside ``set_workspace_path``, but the
dispatcher calls that function on every spawn, which would defeat the
circuit breaker. The shipped shape is opt-in via keyword-only
``reset_failure_counter=False``. The operator's ``_cmd_claim`` flow
opts in explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_SKIP_ASSIGNEE_VALIDATION", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _get_failure_counter(conn, task_id):
    row = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None, f"task {task_id} not found"
    return int(row[0])


# ---------------------------------------------------------------------------
# §4.1 -- create_task absolute-path guard (4 spec tests)
# ---------------------------------------------------------------------------

def test_create_task_rejects_relative_dir_workspace_path(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="is not absolute"):
            kb.create_task(
                conn,
                title="relative dir path should be rejected",
                body="Test that create_task guards against non-absolute paths.",
                assignee="r",
                created_by="test",
                workspace_kind="dir",
                workspace_path="relative/path",
            )
    finally:
        conn.close()


def test_create_task_accepts_absolute_dir_workspace_path(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="absolute dir path is accepted",
            body="Test that create_task accepts absolute paths.",
            assignee="r",
            created_by="test",
            workspace_kind="dir",
            workspace_path="/tmp/kanban_test_absolute_dir",
        )
        task = kb.get_task(conn, tid)
        assert task.workspace_path == "/tmp/kanban_test_absolute_dir"
        assert task.workspace_kind == "dir"
    finally:
        conn.close()


def test_create_task_rejects_relative_worktree_workspace_path(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="is not absolute"):
            kb.create_task(
                conn,
                title="relative worktree path should be rejected",
                body="Test that create_task guards worktree paths too.",
                assignee="r",
                created_by="test",
                workspace_kind="worktree",
                workspace_path="../relative/worktree",
            )
    finally:
        conn.close()


def test_create_task_accepts_absolute_scratch_workspace_path(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="absolute scratch path is accepted",
            body="Test that create_task accepts absolute scratch paths.",
            assignee="r",
            created_by="test",
            workspace_kind="scratch",
            workspace_path="/tmp/kanban_test_absolute_scratch",
        )
        task = kb.get_task(conn, tid)
        assert task.workspace_path == "/tmp/kanban_test_absolute_scratch"
        assert task.workspace_kind == "scratch"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# §4.2 -- set_workspace_path opt-in reset (1 deviation-pin test)
# ---------------------------------------------------------------------------

def test_set_workspace_path_resets_failure_counter_only_when_opted_in(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="reset_failure_counter opt-in test",
            body="Pin the §4.2 deviation from the t_da44022e handoff.",
            assignee="r",
            created_by="test",
        )
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 5 WHERE id = ?",
            (tid,),
        )
        assert _get_failure_counter(conn, tid) == 5

        kb.set_workspace_path(conn, tid, "/tmp/dispatcher_path")
        assert _get_failure_counter(conn, tid) == 5, (
            "Default set_workspace_path call must not reset the failure "
            "counter -- the dispatcher invokes this on every spawn."
        )
        assert kb.get_task(conn, tid).workspace_path == "/tmp/dispatcher_path"

        kb.set_workspace_path(
            conn, tid, "/tmp/operator_path", reset_failure_counter=True
        )
        assert _get_failure_counter(conn, tid) == 0, (
            "Opt-in set_workspace_path call must reset the failure "
            "counter -- that's the operator's manual claim flow."
        )
        assert kb.get_task(conn, tid).workspace_path == "/tmp/operator_path"
    finally:
        conn.close()
