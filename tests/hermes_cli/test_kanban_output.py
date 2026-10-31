"""Regression tests for Kanban text-output timestamp formatting."""

from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_output import _fmt_ts


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_fmt_ts_accepts_numeric_epoch():
    timestamp = 1790071200

    assert _fmt_ts(timestamp) == time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def test_fmt_ts_accepts_iso8601_string():
    raw = "2026-09-22T16:00:00+07:00"
    timestamp = datetime.fromisoformat(raw).timestamp()

    assert _fmt_ts(raw) == time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def test_fmt_ts_invalid_values_are_display_safe():
    for value in (None, "", "not-a-timestamp", "2026-99-99T00:00:00Z", float("nan")):
        assert _fmt_ts(value) == ""


def test_kanban_show_renders_iso_comment_timestamp(kanban_home):
    raw = "2026-09-22T16:00:00+07:00"
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy comment timestamp")
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (task_id, "agent", "legacy comment", raw),
        )
        conn.commit()

    output = kc.run_slash(f"show {task_id}")

    assert f"[{_fmt_ts(raw)}] agent: legacy comment" in output