from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_cli import kanban_db
from hermes_cli.notion_kanban_sync import (
    NotionKanbanSync,
    NotionTask,
    hermes_status_to_notion,
    make_task_body,
    normalize_assignee,
    normalize_notion_status,
    notion_page_id_from_task,
    notion_properties_for_sync,
    notion_status_to_hermes,
    priority_to_int,
    set_kanban_status,
)


def _conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    kanban_db.init_db(db_path=db)
    return kanban_db.connect(db_path=db)


def test_legacy_notion_statuses_normalize_to_canonical_lifecycle():
    assert normalize_notion_status("Not Started") == "Todo"
    assert normalize_notion_status("To Do") == "Todo"
    assert normalize_notion_status("Ready for Creation") == "Ready"
    assert normalize_notion_status("In Progress") == "Running"
    assert normalize_notion_status("Blocked") == "Triage"
    assert normalize_notion_status("Completed") == "Done"
    assert normalize_notion_status("Cancelled") == "Done"
    assert normalize_notion_status("weird custom status") == "Triage"


def test_canonical_notion_statuses_map_to_hermes_runtime_statuses():
    assert notion_status_to_hermes("Triage") == "triage"
    assert notion_status_to_hermes("Todo") == "todo"
    assert notion_status_to_hermes("Ready") == "ready"
    assert notion_status_to_hermes("Running") == "running"
    assert notion_status_to_hermes("Done") == "done"
    assert hermes_status_to_notion("blocked") == "Triage"
    assert hermes_status_to_notion("archived") == "Done"


def test_assignee_and_priority_normalization_are_safe():
    assert normalize_assignee("Dev", {"dev", "halo"}) == "dev"
    assert normalize_assignee("Head of Engineering", {"dev", "halo"}) == "dev"
    assert normalize_assignee("Unknown Agent", {"dev", "halo"}) is None
    assert priority_to_int("High") > priority_to_int("Medium") > priority_to_int("Low")
    assert priority_to_int(None) == 0


def test_make_task_body_preserves_notion_cross_reference():
    notion = NotionTask(
        page_id="abc123",
        url="https://notion.so/example",
        title="Ship sync",
        status="In Progress",
        canonical_status="Running",
        assigned_agent="Dev",
        priority="High",
        blockers="none",
        notes="keep notes",
        source="Canon",
        last_edited_time="2026-05-14T13:00:00Z",
        hermes_task_id=None,
    )

    body = make_task_body(notion)

    assert "Notion Page ID: abc123" in body
    assert "Notion URL: https://notion.so/example" in body
    assert "Blockers: none" in body
    assert "Notes: keep notes" in body


def test_notion_page_id_from_task_prefers_idempotency_key(tmp_path):
    with _conn(tmp_path) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="Imported",
            assignee="dev",
            idempotency_key="notion:page-123",
        )
        task = kanban_db.get_task(conn, task_id)

    assert task is not None
    assert notion_page_id_from_task(task) == "page-123"


def test_set_kanban_status_appends_a_sync_event(tmp_path):
    with _conn(tmp_path) as conn:
        task_id = kanban_db.create_task(conn, title="Imported", assignee="dev")

        changed = set_kanban_status(conn, task_id, "done", reason="Notion says complete")

        task = kanban_db.get_task(conn, task_id)
        events = kanban_db.list_events(conn, task_id)

    assert changed is True
    assert task is not None
    assert task.status == "done"
    assert any(event.kind == "notion_sync_status" for event in events)


def test_notion_properties_for_sync_never_hard_deletes_or_overwrites_notes():
    props = notion_properties_for_sync(status="Running", hermes_task_id="t_1234", hermes_status="running")

    assert props["Status"] == {"select": {"name": "Running"}}
    assert props["Hermes Task ID"]["rich_text"][0]["text"]["content"] == "t_1234"
    assert "Notes" not in props
    assert "Blockers" not in props


class _FakeNotion:
    database_id = "db-test"

    def __init__(self, pages):
        self.pages = pages
        self.updated = []

    def ensure_properties(self, *, dry_run: bool):
        return {"missing_statuses": [], "properties_added": []}

    def query_tasks(self, *, limit=None, since=None):
        return self.pages[:limit] if limit else self.pages

    def update_page_properties(self, page_id, properties):
        self.updated.append((page_id, properties))

    def append_activity(self, page_id, text):
        pass


def _notion_page(page_id: str, title: str, status: str, hermes_task_id: str | None = None):
    properties = {
        "Task": {"type": "title", "title": [{"plain_text": title}]},
        "Status": {"type": "select", "select": {"name": status}},
        "Assigned Agent": {"type": "rich_text", "rich_text": [{"plain_text": "Dev"}]},
    }
    if hermes_task_id:
        properties["Hermes Task ID"] = {"type": "rich_text", "rich_text": [{"plain_text": hermes_task_id}]}
    return {
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "archived": False,
        "last_edited_time": "2026-05-14T13:00:00Z",
        "properties": properties,
    }


def test_initial_notion_import_preserves_todo_runtime_status(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    notion = _FakeNotion([_notion_page("page-todo", "Todo task", "To Do")])
    sync = NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = sync.run_once(dry_run=False, quiet=True)

    with kanban_db.connect(db_path=db) as conn:
        tasks = kanban_db.list_tasks(conn, include_archived=True)

    assert stats.hermes_tasks_created == 1
    assert len(tasks) == 1
    assert tasks[0].idempotency_key == "notion:page-todo"
    assert tasks[0].status == "todo"


def test_conflict_resolution_records_sync_state_to_prevent_repeated_comments(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    with kanban_db.connect(db_path=db) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="Conflicted task",
            assignee="dev",
            idempotency_key="notion:page-conflict",
        )
    notion = _FakeNotion([_notion_page("page-conflict", "Conflicted task", "To Do", hermes_task_id=task_id)])
    state_path.write_text(
        '{"pages":{"page-conflict":{"last_synced_at":"2026-05-14T00:00:00Z","hermes_task_id":"%s"}},"tasks":{}}' % task_id,
        encoding="utf-8",
    )
    sync = NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=state_path)

    stats, _ = sync.run_once(dry_run=False, quiet=True)

    assert stats.conflicts == 1
    assert sync.state["pages"]["page-conflict"]["last_synced_at"] > "2026-05-14T00:00:00Z"
    assert sync.state["tasks"][task_id]["notion_page_id"] == "page-conflict"
