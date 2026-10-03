"""Regression coverage for recoverable active-PR respawn guards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_db_connect as connections


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    connections.init_db()
    return home


def _add_pr_comment(conn, task_id: str, created_at: int) -> None:
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, 'worker', ?, ?)",
        (task_id, "Opened https://github.com/example/project/pull/42", created_at),
    )


def _add_blocked_run(conn, task_id: str, started_at: int) -> None:
    conn.execute(
        "INSERT INTO task_runs "
        "(task_id, profile, status, started_at, ended_at, outcome) "
        "VALUES (?, 'worker', 'blocked', ?, ?, 'blocked')",
        (task_id, started_at, started_at + 1),
    )


def test_active_pr_does_not_guard_a_task_that_has_never_run(kanban_home):
    """A first-run review/merge task can legitimately cite an existing PR."""
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Review PR", assignee="default")
        _add_pr_comment(conn, task_id, now - 10)

        assert dispatch.check_respawn_guard(conn, task_id) is None


def test_active_pr_still_guards_prior_worker_output_without_requeue(kanban_home):
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Implement fix", assignee="default")
        _add_blocked_run(conn, task_id, now - 30)
        _add_pr_comment(conn, task_id, now - 10)

        assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


@pytest.mark.parametrize(
    "event_kind",
    ["promoted_manual", "unblocked"],
)
def test_active_pr_yields_to_a_later_explicit_requeue(kanban_home, event_kind):
    """An explicit requeue after the PR handoff asks the task to continue."""
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Continue PR", assignee="default")
        _add_blocked_run(conn, task_id, now - 40)
        _add_pr_comment(conn, task_id, now - 20)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
            (task_id, event_kind, now - 10),
        )

        assert dispatch.check_respawn_guard(conn, task_id) is None


def test_active_pr_keeps_same_second_requeue_guarded(kanban_home):
    """Second-resolution ties fail closed because causality is ambiguous."""
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Continue PR", assignee="default")
        _add_blocked_run(conn, task_id, now - 40)
        _add_pr_comment(conn, task_id, now - 10)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'unblocked', ?)",
            (task_id, now - 10),
        )

        assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


def test_active_pr_uses_latest_pr_comment_for_requeue_order(kanban_home):
    """A requeue between two PR comments does not supersede newer evidence."""
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Continue PR", assignee="default")
        _add_blocked_run(conn, task_id, now - 50)
        _add_pr_comment(conn, task_id, now - 30)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'unblocked', ?)",
            (task_id, now - 20),
        )
        _add_pr_comment(conn, task_id, now - 10)

        assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


@pytest.mark.parametrize("event_kind", ["status", "promoted", "reclaimed"])
def test_automatic_recovery_does_not_supersede_worker_pr(kanban_home, event_kind):
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Retain PR guard", assignee="default")
        _add_blocked_run(conn, task_id, now - 40)
        _add_pr_comment(conn, task_id, now - 20)
        conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
                     (task_id, event_kind, now - 10))
        assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


@pytest.mark.parametrize("payload,expected", [
    ('{"from":"worker","assignee":"reviewer"}', None),
    ('{"from":"worker","assignee":"worker"}', "active_pr"),
    ('{"assignee":"reviewer","source":"kanban.default_assignee"}', "active_pr"),
])
def test_existing_handoff_policy_survives_recovery(kanban_home, payload, expected):
    now = int(time.time())
    with connections.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Retain handoff policy", assignee="default")
        _add_blocked_run(conn, task_id, now - 40)
        _add_pr_comment(conn, task_id, now - 20)
        conn.execute("INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'assigned', ?, ?)",
                     (task_id, payload, now - 10))
        assert dispatch.check_respawn_guard(conn, task_id) == expected


def test_dispatch_json_exposes_respawn_guarded(kanban_home, monkeypatch, capsys):
    """Deterministic supervisors must be able to reconcile guarded cards."""
    from hermes_cli import kanban_ops as kb_cli

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        dispatch,
        "dispatch_once",
        lambda conn, **kwargs: dispatch.DispatchResult(
            respawn_guarded=[("t_guarded", "active_pr")]
        ),
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=True)

    assert kb_cli._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["respawn_guarded"] == [
        {"task_id": "t_guarded", "reason": "active_pr"}
    ]
