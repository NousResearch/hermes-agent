"""Execution-control edits: raw validation, identity denial and isolated persistence.

The pure/refusal subset can run in a genuine worker context. Positive tests MUST
run from a genuine operator context; never clear inherited identity to run them.
"""
import argparse
import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def controls_module():
    from hermes_cli import kanban_db_controls
    return kanban_db_controls


@pytest.mark.parametrize("field,value", [
    ("goal_mode", None), ("goal_mode", 1), ("goal_mode", "false"),
    ("goal_max_turns", True), ("goal_max_turns", 0), ("goal_max_turns", -1),
    ("max_retries", False), ("max_retries", 0), ("max_retries", "2"),
    ("max_runtime_seconds", True), ("max_runtime_seconds", -1),
    ("max_runtime_seconds", 1.5), ("max_runtime_seconds", 2**63),
    ("unknown", 1),
])
def test_pure_invalid_controls(field, value):
    with pytest.raises(ValueError):
        controls_module().validate_controls({field: value})


def test_pure_values_and_explicit_clear():
    values = dict(goal_mode=False, goal_max_turns=None, max_retries=1,
                  max_runtime_seconds=None)
    assert controls_module().validate_controls(values) == values
    assert controls_module().validate_controls({}) == {}


def test_pure_dashboard_strict_values_and_presence():
    from plugins.kanban.dashboard.plugin_api import UpdateTaskBody, CreateTaskBody
    from pydantic import ValidationError
    assert UpdateTaskBody().model_fields_set.isdisjoint({"max_retries"})
    assert UpdateTaskBody(max_retries=None).model_fields_set == {"max_retries"}
    assert CreateTaskBody(title="isolated", max_retries=1).max_retries == 1
    for cls in (UpdateTaskBody, CreateTaskBody):
        for values in ({"goal_mode": 1}, {"max_retries": True}, {"goal_max_turns": "2"},
                       {"max_runtime_seconds": 0}, {"goal_mode": None}):
            with pytest.raises(ValidationError):
                cls(title="isolated", **values)


def test_pure_cli_control_values():
    from hermes_cli.kanban_parser import build_parser
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    args = parser.parse_args(["kanban", "edit", "t_example", "--goal-mode", "false",
                             "--goal-max-turns", "clear", "--max-retries", "1",
                             "--max-runtime-seconds", "1800"])
    assert args.goal_mode is False
    assert args.goal_max_turns is None
    assert args.max_retries == 1
    assert args.max_runtime_seconds == 1800


def test_refusal_delegated_before_connection_access():
    from agent.delegation_context import delegated_child_context
    with delegated_child_context(), pytest.raises(PermissionError):
        kb.edit_task(None, "t_example", goal_mode=True)


def test_refusal_worker_before_connection_access(monkeypatch):
    # Only add a denial marker; never remove or relax inherited authority.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    with pytest.raises(PermissionError):
        kb.edit_task(None, "t_example", max_retries=1)


@pytest.fixture
def operator_db(tmp_path, monkeypatch):
    from agent.delegation_context import is_delegated_child_process_context
    if is_delegated_child_process_context() or os.environ.get("HERMES_KANBAN_TASK"):
        pytest.fail("positive-requiredUNRUN: execute from genuine operator context, do not clear identity")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "controls.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    conn = kbc.connect(tmp_path / "controls.db")
    yield conn
    conn.close()


@pytest.mark.parametrize("status", ["blocked", "todo", "triage"])
def test_positive_atomic_update_preserves_history(operator_db, status, monkeypatch):
    conn = operator_db
    parent = kb.create_task(conn, title="parent", workspace_kind="scratch", initial_status="blocked")
    tid = kb.create_task(conn, title="task", workspace_kind="scratch",
                         initial_status="blocked" if status == "blocked" else "running",
                         triage=status == "triage", parents=[parent] if status == "todo" else [],
                         max_runtime_seconds=90, max_retries=2)
    assert kb.get_task(conn, tid).status == status
    # Historical data fixtures are exclusively in this newly created test DB.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET consecutive_failures=1, block_recurrences=2 WHERE id=?", (tid,))
        conn.execute("INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at, max_runtime_seconds) VALUES (?, 'test', 'failed', 'failed', 1, 2, 90)", (tid,))
    before = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    runs = [tuple(r) for r in conn.execute("SELECT * FROM task_runs")]
    links = [tuple(r) for r in conn.execute("SELECT * FROM task_links")]
    events = [tuple(r) for r in conn.execute("SELECT * FROM task_events")]
    hooks = []
    def notified(c, task_id, fields, **kwargs):
        assert not c.in_transaction
        hooks.append((task_id, fields))
    monkeypatch.setattr(kb, "notify_task_updated", notified)
    values = dict(goal_mode=True, goal_max_turns=2, max_retries=1, max_runtime_seconds=1800)
    assert kb.edit_task(conn, tid, **values)
    task = kb.get_task(conn, tid)
    from hermes_cli.kanban_output import _task_to_dict
    for field, value in values.items():
        assert getattr(task, field) == value
        assert type(getattr(task, field)) is type(value)
        assert _task_to_dict(task)[field] == value
    after = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    assert {k: v for k, v in before.items() if k not in values} == {k: v for k, v in after.items() if k not in values}
    assert runs == [tuple(r) for r in conn.execute("SELECT * FROM task_runs")]
    assert links == [tuple(r) for r in conn.execute("SELECT * FROM task_links")]
    assert events == [tuple(r) for r in conn.execute("SELECT * FROM task_events")][:len(events)]
    event = conn.execute("SELECT payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    audit = json.loads(event[0])["execution_controls"]
    assert audit["old"]["max_runtime_seconds"] == 90
    assert audit["new"] == values
    assert len(hooks) == 1
    assert kb.edit_task(conn, tid, goal_mode=False, goal_max_turns=None,
                        max_retries=None, max_runtime_seconds=None)
    assert kb.get_task(conn, tid).max_runtime_seconds is None


def test_positive_invalid_and_mixed_edit_no_mutation(operator_db):
    conn = operator_db
    tid = kb.create_task(conn, title="unchanged", workspace_kind="scratch", initial_status="blocked")
    before = list(conn.iterdump())
    for values in (dict(goal_mode=True, max_retries=False),
                   dict(goal_mode=True, title="must not change")):
        with pytest.raises(ValueError):
            kb.edit_task(conn, tid, **values)
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize("column,value", [("status", "running"), ("claim_lock", "claim"),
    ("claim_expires", 1), ("worker_pid", 123), ("current_run_id", 123)])
def test_positive_active_refused(operator_db, column, value):
    conn = operator_db
    tid = kb.create_task(conn, title="stopped", workspace_kind="scratch", initial_status="blocked")
    with kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {column}=? WHERE id=?", (value, tid))
    before = list(conn.iterdump())
    with pytest.raises(ValueError, match="stopped"):
        kb.edit_task(conn, tid, goal_mode=True)
    assert list(conn.iterdump()) == before
