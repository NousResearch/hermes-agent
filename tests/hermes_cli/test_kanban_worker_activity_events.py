"""Kanban worker activity event tests.

Workers run outside the live WebUI chat stream, so their observable progress must
be persisted as structured Kanban events that every UI can render.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated_kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _load_activity_journal_class():
    try:
        module = importlib.import_module("hermes_cli.kanban_worker_activity")
    except ModuleNotFoundError as exc:
        raise AssertionError("missing hermes_cli.kanban_worker_activity module") from exc
    assert hasattr(module, "KanbanWorkerActivityJournal")
    return module.KanbanWorkerActivityJournal


def test_worker_activity_journal_persists_structured_events(isolated_kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="visible worker", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        journal_cls = _load_activity_journal_class()
        journal = journal_cls(task_id=task_id, run_id=claimed.current_run_id)

        journal.tool_start("call-1", "terminal", {"command": "pytest -q"})
        journal.tool_end("call-1", "terminal", {"command": "pytest -q"}, "1 passed", duration=1.25, is_error=False)
        journal.assistant_text("Je lance les tests ciblés.")
        journal.progress_note("Lecture du journal worker")
        journal.heartbeat_note("toujours en cours")

        events = [e for e in kb.list_events(conn, task_id) if e.kind in journal.EVENT_KINDS]
        assert [e.kind for e in events] == [
            "tool_start",
            "tool_end",
            "assistant_text",
            "progress_note",
            "heartbeat_note",
        ]
        assert all(e.run_id == claimed.current_run_id for e in events)
        assert events[0].payload["tool"] == "terminal"
        assert events[0].payload["tool_call_id"] == "call-1"
        assert events[1].payload["duration"] == 1.25
        assert events[1].payload["is_error"] is False
        assert events[2].payload["text"] == "Je lance les tests ciblés."
        assert events[-1].payload["note"] == "toujours en cours"
    finally:
        conn.close()


def test_worker_activity_journal_consumes_steer_events(isolated_kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="steerable worker", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        journal_cls = _load_activity_journal_class()
        journal = journal_cls(task_id=task_id, run_id=claimed.current_run_id)
        journal._steer_cursor = 0
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "steer_note",
                {"message": "Arrête la piste A, teste plutôt B", "source": "webui"},
                run_id=claimed.current_run_id,
            )

        class FakeAgent:
            def __init__(self):
                self.steers = []
            def steer(self, text):
                self.steers.append(text)
                return True

        agent = FakeAgent()
        assert journal.consume_steer_once(agent) == 1
        assert agent.steers == ["Arrête la piste A, teste plutôt B"]
        events = [e for e in kb.list_events(conn, task_id) if e.kind == "steer_accepted"]
        assert len(events) == 1
        assert events[0].payload["message"] == "Arrête la piste A, teste plutôt B"
    finally:
        conn.close()


def test_cli_agent_setup_wires_kanban_worker_activity_callbacks():
    """HermesCLI must compose normal display callbacks with Kanban journal callbacks."""
    mixin = Path("hermes_cli/cli_agent_setup_mixin.py").read_text(encoding="utf-8")
    assert "KanbanWorkerActivityJournal.from_environment" in mixin
    assert "_compose_kanban_worker_callback" in mixin
    assert "kanban_activity.tool_progress" in mixin
    assert "kanban_activity.tool_start" in mixin
    assert "kanban_activity.tool_complete" in mixin
    assert "kanban_activity.stream_delta" in mixin
    assert "interim_assistant_callback=interim_assistant_callback" in mixin
    assert "kanban_activity.assistant_text" in mixin
    assert "kanban_activity.start_steer_polling(self.agent)" in mixin
