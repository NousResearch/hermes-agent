"""Tests for Kanban projection repair / reconciliation CLI behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)


def _insert_notify_sub(
    conn,
    *,
    task_id: str,
    platform: str = "discord",
    chat_id: str = "forum-1",
    thread_id: str = "thread-1",
    user_id: str | None = "user-1",
    notifier_profile: str | None = "default",
    last_event_id: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO kanban_notify_subs (
            task_id, platform, chat_id, thread_id, user_id,
            notifier_profile, created_at, last_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, 1234, ?)
        """,
        (task_id, platform, chat_id, thread_id, user_id, notifier_profile, last_event_id),
    )
    conn.commit()


def test_kanban_repair_json_reports_projection_orphans_without_mutating(kanban_home, capsys):
    """Dry-run repair should report orphaned projection rows without deleting them."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="live projected task")
        _insert_notify_sub(conn, task_id=task_id, chat_id="forum-1", thread_id="thread-live")
        _insert_notify_sub(conn, task_id="t_missing", chat_id="forum-1", thread_id="thread-orphan")

    rc = _run_cli("repair", "--json")

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["projection_subscriptions"]["scanned"] == 2
    assert report["projection_subscriptions"]["orphaned"] == 1
    assert report["projection_subscriptions"]["removed"] == 0
    assert report["projection_subscriptions"]["orphans"] == [
        {
            "task_id": "t_missing",
            "platform": "discord",
            "chat_id": "forum-1",
            "thread_id": "thread-orphan",
        }
    ]

    with kb.connect() as conn:
        rows = kb.list_notify_subs(conn)
    assert {row["task_id"] for row in rows} == {task_id, "t_missing"}


def test_kanban_repair_apply_removes_orphaned_projection_rows_and_reports_count(kanban_home, capsys):
    """Apply mode should remove orphaned projection rows while preserving live rows."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="live projected task")
        _insert_notify_sub(conn, task_id=task_id, chat_id="forum-1", thread_id="thread-live")
        _insert_notify_sub(conn, task_id="t_missing", chat_id="forum-1", thread_id="thread-orphan")

    rc = _run_cli("repair", "--apply", "--json")

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is False
    assert report["projection_subscriptions"]["scanned"] == 2
    assert report["projection_subscriptions"]["orphaned"] == 1
    assert report["projection_subscriptions"]["removed"] == 1

    with kb.connect() as conn:
        rows = kb.list_notify_subs(conn)
    assert [(row["task_id"], row["thread_id"]) for row in rows] == [(task_id, "thread-live")]
