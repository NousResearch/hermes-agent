"""Tests for the interactive CLI Kanban event watcher."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_cli_events as cli_events


@pytest.fixture
def event_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    db = home / "kanban.db"
    kb.init_db(db, board="default")
    monkeypatch.setattr(cli_events, "kanban_db_path", lambda board=None: db)
    return db


def _event(db: Path, task_id: str, kind: str, payload: dict | None = None) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, 1, kind, json.dumps(payload) if payload else None, 1),
    )
    conn.commit()
    conn.close()


def test_poll_filters_terminal_events_and_persists_cursor(event_db: Path, tmp_path: Path) -> None:
    _event(event_db, "t_one", "created")
    _event(event_db, "t_one", "completed", {"summary": "done"})
    _event(event_db, "t_two", "blocked", {"reason": "needs input"})

    rendered: list[str] = []
    bridge = cli_events.KanbanEventBridge(on_render=rendered.append, poll_interval=0)
    events = bridge.poll()
    bridge.stop()

    assert [event.kind for event in events] == ["completed", "blocked"]
    assert bridge._cursor == 3
    cursor = tmp_path / ".hermes" / "kanban" / "cli_event_cursor"
    assert cursor.read_text() == "3"


def test_active_turn_queues_and_idle_delivery_flushes(event_db: Path) -> None:
    _event(event_db, "t_one", "completed", {"summary": "done"})
    running = True
    rendered: list[str] = []
    bridge = cli_events.KanbanEventBridge(
        on_render=rendered.append,
        is_agent_running=lambda: running,
        poll_interval=0,
    )

    bridge.deliver_events(bridge.poll())
    assert rendered == []
    assert len(bridge.drain_queue()) == 1

    running = False
    _event(event_db, "t_one", "blocked", {"reason": "review"})
    bridge.deliver_events(bridge.poll())
    bridge.stop()
    assert any("blocked" in line for line in rendered)


def test_format_event_is_compact_and_contains_handoff(event_db: Path) -> None:
    _event(event_db, "t_one", "completed", {"summary": "worker finished"})
    bridge = cli_events.KanbanEventBridge(poll_interval=0)
    event = bridge.poll()[0]
    bridge.stop()

    line = bridge.format_event(event)
    assert "t_one" in line
    assert "completed" in line
    assert "worker finished" in line


def test_resume_cursor_skips_acknowledged_events(event_db: Path) -> None:
    _event(event_db, "t_one", "completed")
    first = cli_events.KanbanEventBridge(poll_interval=0)
    assert len(first.poll()) == 1
    first.stop()

    _event(event_db, "t_one", "blocked")
    second = cli_events.KanbanEventBridge(poll_interval=0)
    events = second.catch_up()
    second.stop()
    assert [event.kind for event in events] == ["blocked"]
