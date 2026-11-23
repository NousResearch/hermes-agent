"""All-or-nothing contracts for combined assignee and scalar storage edits."""
import pytest
from pathlib import Path
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_edit import update_task_fields


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home



def test_mixed_edit_rolls_back_assignment_and_events_on_late_guard(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="original")
        before = kb.list_events(conn, task_id)
        real_append = kb._append_event

        def racing_append(conn, tid, kind, payload=None, **kwargs):
            real_append(conn, tid, kind, payload, **kwargs)
            if kind == "assigned":
                conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))

        monkeypatch.setattr(kb, "_append_event", racing_append)
        with pytest.raises(RuntimeError, match="finished task"):
            update_task_fields(conn, task_id, assign=True, assignee="worker", title="changed")
        task = kb.get_task(conn, task_id)
        assert (task.assignee, task.title, task.status) == (None, "original", "ready")
        assert kb.list_events(conn, task_id) == before


def test_mixed_edit_commits_assignment_and_scalars_together(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="original")
        assert update_task_fields(conn, task_id, assign=True, assignee="worker", title="changed", priority=3)
        task = kb.get_task(conn, task_id)
        assert (task.assignee, task.title, task.priority) == ("worker", "changed", 3)
        assert {event.kind for event in kb.list_events(conn, task_id)} >= {"assigned", "edited", "reprioritized"}
