"""Completed-event payloads must carry the worker's multi-line summary.

The gateway notifier renders ``payload["summary"]`` as the Slack ping; a
first-line-only clip collapsed 3-5 bullet briefings to a title line.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

MULTI_LINE = "*Daily outbound drafts — 2026-09-22*\n• 3 ICP prospects identified\n• Review: http://localhost:9100"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _last_completed_payload(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    import json

    return json.loads(row[0])


def test_completed_event_keeps_multiline_summary(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="daily outbound")
        assert kb.complete_task(conn, tid, result="ok", summary=MULTI_LINE)
        payload = _last_completed_payload(conn, tid)
    assert payload["summary"] == MULTI_LINE


def test_completed_event_summary_bounded(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="big summary")
        assert kb.complete_task(conn, tid, result="ok", summary="x" * 2000)
        payload = _last_completed_payload(conn, tid)
    assert len(payload["summary"]) == 1500
