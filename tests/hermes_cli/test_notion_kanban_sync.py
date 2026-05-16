from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import notion_kanban_sync as sync


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class FakeNotion(sync.NotionClient):
    def __init__(self, pages):
        self.database_id = "fake-db"
        self.pages = pages
        self.updated = []
        self.created_pages = []
        self.activity = []

    def retrieve_database(self):
        return {
            "properties": {
                "Task": {"type": "title"},
                "Status": {"type": "select", "select": {"options": [
                    {"name": "Triage"}, {"name": "Todo"}, {"name": "Ready"},
                    {"name": "Running"}, {"name": "Blocked"}, {"name": "Done"}, {"name": "Archived"},
                ]}},
                "Notes": {"type": "rich_text"},
                "Source": {"type": "rich_text"},
                "Hermes Task ID": {"type": "rich_text"},
                "Last Synced At": {"type": "date"},
                "Sync Source": {"type": "rich_text"},
                "Sync Error": {"type": "rich_text"},
            }
        }

    def ensure_properties(self, *, dry_run, prune_status_options=False):
        return {"missing_statuses": [], "extra_statuses": [], "properties_added": [], "retired_properties": []}

    def query_tasks(self, *, page_size=100, since=None, limit=None):
        return self.pages[:limit] if limit else list(self.pages)

    def update_page_properties(self, page_id, properties):
        self.updated.append((page_id, properties))

    def create_page_for_hermes_task(self, task, *, schema):
        raise AssertionError("not expected")

    def append_activity(self, page_id, text):
        self.activity.append((page_id, text))


def notion_page(page_id, title, status, hermes_task_id=None, last_edited_time="2026-01-01T00:10:00Z"):
    props = {
        "Task": {"type": "title", "title": [{"plain_text": title}]},
        "Status": {"type": "select", "select": {"name": status}},
        "Assigned Agent": {"type": "select", "select": {"name": "Dev"}},
        "Priority": {"type": "select", "select": {"name": "Medium"}},
        "Notes": {"type": "rich_text", "rich_text": []},
        "Source": {"type": "rich_text", "rich_text": []},
        "Hermes Task ID": {"type": "rich_text", "rich_text": []},
    }
    if hermes_task_id:
        props["Hermes Task ID"] = {"type": "rich_text", "rich_text": [{"plain_text": hermes_task_id}]}
    return {
        "id": page_id,
        "url": f"https://notion.test/{page_id}",
        "last_edited_time": last_edited_time,
        "archived": False,
        "properties": props,
    }


def test_new_notion_running_imports_as_ready_not_ghost_running(kanban_home, tmp_path):
    notion = FakeNotion([notion_page("page-running", "import me", "Running")])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = engine.run_once(dry_run=False, max_creates=5)

    assert stats.hermes_tasks_created == 1
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn, include_archived=True)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert task.current_run_id is None
    assert notion.updated
    props = notion.updated[-1][1]
    assert props["Status"]["select"]["name"] == "Ready"
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "Only the dispatcher may set a task to running" in sync_error


def test_existing_task_ignores_notion_running_transition(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="paired", assignee="dev")
    notion = FakeNotion([notion_page("page-paired", "paired", "Running", hermes_task_id=task_id)])
    engine = sync.NotionKanbanSync(notion=notion, report_dir=tmp_path / "reports", state_path=tmp_path / "state.json")

    stats, _ = engine.run_once(dry_run=False)

    assert stats.conflicts == 1
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        comments = kb.list_comments(conn, task_id)
        assert any("dispatcher owns runtime state" in c.body for c in comments)
    props = notion.updated[-1][1]
    assert props["Status"]["select"]["name"] == "Ready"
    sync_error = props["Sync Error"]["rich_text"][0]["text"]["content"]
    assert "kept Hermes status 'ready'" in sync_error
