"""Superseded Kanban tasks are terminal for dispatcher purposes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from scripts import report_kanban_superseded


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect_closing() as connection:
        yield connection


def test_superseded_ready_task_is_blocked_and_never_dispatched(conn, monkeypatch):
    replacement = kb.create_task(conn, title="clean replacement")
    original = kb.create_task(conn, title="stale rescue", assignee="coder")

    assert kb.supersede_task(conn, original, replacement, actor="operator")
    task = kb.get_task(conn, original)
    assert task.status == "blocked"
    assert task.superseded_by == replacement
    assert kb.claim_task(conn, original) is None

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    result = kb.dispatch_once(conn, dry_run=True)
    assert original not in {task_id for task_id, *_ in result.spawned}


def test_superseded_todo_task_is_not_promoted_when_parent_finishes(conn):
    parent = kb.create_task(conn, title="parent")
    replacement = kb.create_task(conn, title="replacement")
    original = kb.create_task(
        conn,
        title="old child",
        assignee="coder",
        parents=[parent],
    )

    assert kb.supersede_task(conn, original, replacement, actor="operator")
    kb.complete_task(conn, parent, result="done")

    assert kb.recompute_ready(conn) == 0
    task = kb.get_task(conn, original)
    assert task.status == "blocked"
    assert task.superseded_by == replacement


def test_superseded_task_must_be_cleared_before_unblock(conn):
    replacement = kb.create_task(conn, title="replacement")
    original = kb.create_task(conn, title="old card")
    assert kb.supersede_task(conn, original, replacement, actor="operator")

    assert kb.unblock_task(conn, original) is False
    assert kb.get_task(conn, original).status == "blocked"

    assert kb.clear_supersession(conn, original, actor="operator")
    assert kb.unblock_task(conn, original)
    task = kb.get_task(conn, original)
    assert task.status == "ready"
    assert task.superseded_by is None


def test_ordinary_blocked_task_behavior_is_unchanged(conn):
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parents=[parent])
    kb.complete_task(conn, parent, result="done")
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (child,))
    conn.commit()

    assert kb.recompute_ready(conn) == 1
    task = kb.get_task(conn, child)
    assert task.status == "ready"
    assert task.superseded_by is None


def test_legacy_tasks_table_gains_superseded_by_without_data_loss(conn):
    original = kb.create_task(conn, title="legacy card")
    conn.execute("ALTER TABLE tasks DROP COLUMN superseded_by")
    conn.commit()

    kb._migrate_add_optional_columns(conn)

    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
    }
    assert "superseded_by" in columns
    assert kb.get_task(conn, original).title == "legacy card"


def test_cli_listing_labels_superseded_task(conn):
    replacement = kb.create_task(conn, title="replacement")
    original = kb.create_task(conn, title="old card", assignee="coder")
    assert kb.supersede_task(conn, original, replacement, actor="operator")
    task = kb.get_task(conn, original)

    line = kanban_cli._fmt_task_line(task)
    cli_payload = kanban_cli._task_to_dict(task)

    assert "superseded" in line
    assert f"→ {replacement}" in line
    assert cli_payload["superseded_by"] == replacement


def test_supersession_candidates_are_reported_without_mutation(conn):
    replacement = kb.create_task(conn, title="replacement")
    original = kb.create_task(conn, title="old card", assignee="coder")
    claimed = kb.claim_task(conn, original)
    assert claimed is not None
    summary = f"Superseded by {replacement}; keep blocked to avoid retry churn."
    assert kb.block_task(
        conn,
        original,
        reason=summary,
        expected_run_id=claimed.current_run_id,
    )

    candidates = kb.find_supersession_candidates(conn)

    assert candidates == [
        {
            "task_id": original,
            "superseded_by": replacement,
            "summary": summary,
            "already_structured": False,
            "replacement_exists": True,
        }
    ]
    assert kb.get_task(conn, original).superseded_by is None


def test_report_script_is_explicitly_dry_run(conn, capsys):
    replacement = kb.create_task(conn, title="replacement")
    original = kb.create_task(conn, title="old card", assignee="coder")
    claimed = kb.claim_task(conn, original)
    assert claimed is not None
    assert kb.block_task(
        conn,
        original,
        reason=f"Superseded by {replacement}",
        expected_run_id=claimed.current_run_id,
    )

    assert report_kanban_superseded.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["dry_run"] is True
    assert report["candidate_count"] == 1
    assert kb.get_task(conn, original).superseded_by is None
