from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_taskstoissues as tti


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_tasks(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks.md"
    tasks.write_text(
        """
# Tasks

- [ ] T001 Add parser tests in `tests/test_taskstoissues.py`
  for checkbox tasks and stable ids.

- [ ] T002 Implement parser module.
  Depends: T001.
  Keep multiline bodies attached.

- [ ] T003 Implement apply path.
  Depends on: T001, T002.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return tasks


def test_parse_tasks_md_checkboxes_ids_multiline_and_dependencies(tmp_path):
    tasks = tti.parse_tasks_md(_write_tasks(tmp_path))

    assert [task.task_id for task in tasks] == ["T001", "T002", "T003"]
    assert tasks[0].title == (
        "Add parser tests in `tests/test_taskstoissues.py` "
        "for checkbox tasks and stable ids."
    )
    assert "stable ids" in tasks[0].body
    assert tasks[1].depends_on == ("T001",)
    assert tasks[2].depends_on == ("T001", "T002")


def test_build_plan_uses_stable_dedupe_keys_and_detects_existing(kanban_home, tmp_path):
    tasks_file = _write_tasks(tmp_path)
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")
        existing = kb.create_task(
            conn,
            title="T001 existing",
            idempotency_key="tasks-to-issues:067-demo:T001",
        )
        plan = tti.build_plan(
            conn,
            spec_id="067-demo",
            root_ticket=root,
            tasks_file=tasks_file,
        )

    assert plan.items[0].idempotency_key == "tasks-to-issues:067-demo:T001"
    assert plan.items[0].existing_ticket_id == existing
    assert plan.items[1].action == "create"
    assert plan.creates == 2
    assert plan.existing == 1


def test_dry_run_plan_does_not_mutate_db(kanban_home, tmp_path):
    tasks_file = _write_tasks(tmp_path)
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")
        before = len(kb.list_tasks(conn, include_archived=True))
        plan = tti.build_plan(
            conn,
            spec_id="067-demo",
            root_ticket=root,
            tasks_file=tasks_file,
        )
        after = len(kb.list_tasks(conn, include_archived=True))

    assert plan.creates == 3
    assert before == after == 1


def test_apply_creates_linked_tickets_and_rerun_is_idempotent(kanban_home, tmp_path):
    tasks_file = _write_tasks(tmp_path)
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")
        first = tti.apply_tasks_to_issues(
            conn,
            spec_id="067-demo",
            root_ticket=root,
            tasks_file=tasks_file,
            author="tester",
        )
        second = tti.apply_tasks_to_issues(
            conn,
            spec_id="067-demo",
            root_ticket=root,
            tasks_file=tasks_file,
            author="tester",
        )
        rows = conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        ).fetchall()

    assert len(first.created_ticket_ids) == 3
    assert second.created_ticket_ids == ()
    assert set(second.existing_ticket_ids) == set(first.created_ticket_ids)
    t1 = first.task_ticket_ids["T001"]
    t2 = first.task_ticket_ids["T002"]
    t3 = first.task_ticket_ids["T003"]
    links = {(row["parent_id"], row["child_id"]) for row in rows}
    assert (t1, t2) in links
    assert (t1, t3) in links
    assert (t2, t3) in links
    assert (t1, root) in links
    assert (t2, root) in links
    assert (t3, root) in links
    assert len(links) == len(rows)


def test_cli_taskstoissues_json_dry_run(kanban_home, tmp_path, capsys):
    tasks_file = _write_tasks(tmp_path)
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")

    rc = kanban_cli._cmd_taskstoissues(
        argparse.Namespace(
            spec_id="067-demo",
            root_ticket=root,
            spec_root=None,
            tasks_file=str(tasks_file),
            dry_run=True,
            author="tester",
            json=True,
        )
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["creates"] == 3
    assert out["items"][0]["idempotency_key"] == "tasks-to-issues:067-demo:T001"
