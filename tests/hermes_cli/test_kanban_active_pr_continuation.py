"""Regression (2026-11-04): a scheduled continuation resumes its own PR instead of being held as a duplicate."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

from tests.hermes_cli.test_kanban_review_lifecycle import kanban_home  # noqa: F401  (fixture)


def _event(conn, task_id, kind, ts, payload=None):
    conn.execute("INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                 (task_id, kind, payload, ts))
    conn.commit()


def _card_with_pr(conn):
    tid = kb.create_task(conn, title="PRed card", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="Checkpoint: PR https://github.com/example/repo/pull/7")
    ts = conn.execute("SELECT MAX(created_at) FROM task_comments WHERE task_id = ?", (tid,)).fetchone()[0]
    return tid, int(ts)


def test_pr_without_continuation_stays_guarded(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid, _ = _card_with_pr(conn)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_scheduled_then_woken_resumes_own_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid, ts = _card_with_pr(conn)
        _event(conn, tid, "scheduled", ts, '{"reason": "SCHEDULED_UNTIL=x continue: CI on PR #7"}')   # same second
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"      # not woken yet
        _event(conn, tid, "unblocked", ts + 600)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_schedule_before_a_newer_pr_comment_does_not_lift(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid, ts = _card_with_pr(conn)
        _event(conn, tid, "scheduled", ts - 100)
        _event(conn, tid, "unblocked", ts - 50)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize("kind", ["specified", "promoted_manual", "unblocked", "reopened_done"])
def test_deliberate_requeue_after_pr_comment_resumes(kanban_home: Path, kind) -> None:
    with kbc.connect() as conn:
        tid, ts = _card_with_pr(conn)
        _event(conn, tid, kind, ts + 60)
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize("kind", ["promoted", "reclaimed", "crashed"])
def test_automatic_requeue_stays_guarded(kanban_home: Path, kind) -> None:
    with kbc.connect() as conn:
        tid, ts = _card_with_pr(conn)
        _event(conn, tid, kind, ts + 60)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
