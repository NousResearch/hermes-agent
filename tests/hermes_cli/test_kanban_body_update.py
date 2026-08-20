"""Regression tests for audited canonical Kanban task-body updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_update_task_body_is_atomic_audited_and_idempotent(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="body drift",
            body="historical body",
            initial_status="blocked",
        )

        first = kb.update_task_body(
            conn,
            task_id,
            "current canonical body",
            author="p0-kanban-sync",
        )
        assert first is not None
        assert first["changed"] is True
        assert first["old_length"] == len("historical body")
        assert first["new_length"] == len("current canonical body")

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.body == "current canonical body"
        assert task.status == "blocked"
        assert task.assignee is None
        assert task.current_run_id is None

        event = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'body_updated'",
            (task_id,),
        ).fetchone()
        assert event is not None
        assert json.loads(event["payload"])["author"] == "p0-kanban-sync"

        second = kb.update_task_body(
            conn,
            task_id,
            "current canonical body",
            author="p0-kanban-sync",
        )
        assert second is not None
        assert second["changed"] is False
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'body_updated'",
            (task_id,),
        ).fetchone()[0] == 1


def test_update_body_cli_emits_machine_readable_result(kanban_home, capsys):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="cli body", body="old")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(
        [
            "kanban",
            "--board",
            "default",
            "update-body",
            task_id,
            "--body",
            "new from cli",
            "--author",
            "p0-kanban-sync",
            "--json",
        ]
    )

    assert kc.kanban_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == task_id
    assert payload["changed"] is True
    assert payload["body_length"] == len("new from cli")

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.body == "new from cli"
