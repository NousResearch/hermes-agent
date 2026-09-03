"""Assign on a review card must not leave the review lane.

CLI ``kanban assign`` is not a reviewer-naming command — that is
``request-review`` / ``reassign``. The DB helper still records an
assignee (dashboard + request-review internals) without flipping
status, started_at, or claim_lock.
"""

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


def _put_in_review(conn, *, assignee="worker", reviewer="reviewer"):
    tid = kb.create_task(conn, title="impl a feature", assignee=assignee)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id is not None
    ok = kb.request_review(
        conn, tid,
        summary="implementation complete",
        reviewer=reviewer,
        expected_run_id=run_id,
    )
    assert ok is True
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "review"
    return task


def test_assign_task_keeps_review_lane(kanban_home):
    with kb.connect_closing() as conn:
        task = _put_in_review(conn)
        tid = task.id
        started_at = task.started_at
        claim_lock = task.claim_lock
        assert claim_lock is None

        assert kb.assign_task(conn, tid, "other-reviewer") is True

        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.status == "review"
        assert after.started_at == started_at
        assert after.claim_lock is None
        assert after.assignee == "other-reviewer"


def test_cli_assign_refuses_review_card(kanban_home, capsys):
    with kb.connect_closing() as conn:
        task = _put_in_review(conn)
        tid = task.id
        started_at = task.started_at
        assignee = task.assignee

    rc = kc._cmd_assign(argparse.Namespace(task_id=tid, profile="other-reviewer"))
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot assign a card in review" in err
    assert "request-review" in err
    assert "reassign" in err

    with kb.connect_closing() as conn:
        after = kb.get_task(conn, tid)
    assert after is not None
    assert after.status == "review"
    assert after.started_at == started_at
    assert after.claim_lock is None
    assert after.assignee == assignee


def test_cli_reassign_swaps_reviewer_without_leaving_review(kanban_home):
    with kb.connect_closing() as conn:
        task = _put_in_review(conn)
        tid = task.id
        started_at = task.started_at

    rc = kc._cmd_reassign(
        argparse.Namespace(
            task_id=tid, profile="other-reviewer", reclaim=False, reason=None,
        )
    )
    assert rc == 0

    with kb.connect_closing() as conn:
        after = kb.get_task(conn, tid)
    assert after is not None
    assert after.status == "review"
    assert after.started_at == started_at
    assert after.claim_lock is None
    assert after.assignee == "other-reviewer"


def test_list_json_running_claimed_is_strictly_parseable(kanban_home, capsys):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="claimed running", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.get_task(conn, tid).claim_lock is not None

    rc = kc._cmd_list(
        argparse.Namespace(
            assignee=None,
            mine=False,
            status=None,
            tenant=None,
            session=None,
            archived=False,
            sort=None,
            workflow_template_id=None,
            current_step_key=None,
            json=True,
        )
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.lstrip()[:1] == "["
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    row = next(r for r in payload if r["id"] == tid)
    assert row["status"] == "running"
    assert captured.err == ""
