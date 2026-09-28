"""Request-level status distinct from orchestration status (P0 card).

``kanban_db.compute_request_view`` is the single implementation consumed by
`hermes kanban show --json`, `hermes kanban show` (text), the `kanban_show`
tool and `kanban_list`/`_task_summary_dict`: owner, a verified outcome that
does not just mirror `status` for a decomposed-but-undelivered task, next
action, and blocker age. No invented ETAs -- every field traces to a real
DB row/event.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _view(conn, task_id):
    task = kb.get_task(conn, task_id)
    assert task is not None
    return kb.compute_request_view(conn, task)


def test_done_with_no_children_is_delivered(conn):
    task_id = kb.create_task(conn, title="t", assignee="anika")
    assert kb.complete_task(conn, task_id, summary="shipped")
    v = _view(conn, task_id)
    assert v["owner"] == "anika"
    assert v["verified_outcome"] == "delivered"
    assert v["open_spawned_children"] == []
    assert v["next_action"] is None


def test_done_with_open_spawned_children_is_not_delivered(conn):
    """The exact falsely-completed-parent shape: orchestration says done,
    but the request itself is not delivered."""
    parent_id = kb.create_task(conn, title="orchestrator", assignee="anika")
    kb.create_task(conn, title="child", assignee="builder", creator_task_id=parent_id)
    assert kb.complete_task(conn, parent_id, summary="decomposed", force=True)

    v = _view(conn, parent_id)
    assert v["verified_outcome"] == "orchestration_done_delivery_pending"
    assert len(v["open_spawned_children"]) == 1
    assert "waiting on spawned children" in v["next_action"]


def test_review_reports_waiting_with_reviewer_next_action_and_no_blocker_age_yet(conn):
    task_id = kb.create_task(conn, title="t", assignee="anika")
    kb.claim_task(conn, task_id)
    assert kb.request_review(conn, task_id, summary="ready", reviewer="meera")

    v = _view(conn, task_id)
    assert v["verified_outcome"] == "waiting"
    assert "reviewer" in v["next_action"]
    assert "meera" in v["next_action"]
    assert v["blocker_age_seconds"] is not None
    assert v["blocker_age_seconds"] < 5


def test_blocker_age_grows_with_real_elapsed_time(conn):
    task_id = kb.create_task(conn, title="t", assignee="anika")
    assert kb.block_task(conn, task_id, reason="need creds", kind="needs_input")
    row = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'blocked' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE id = ?",
            (int(time.time()) - 7200, row["id"]),
        )
    v = _view(conn, task_id)
    assert v["verified_outcome"] == "waiting"
    assert v["blocker_age_seconds"] >= 7200
    assert "unblock" in v["next_action"]


def test_running_and_ready_are_in_progress_with_no_blocker_age(conn):
    ready_id = kb.create_task(conn, title="ready", assignee="anika")
    v = _view(conn, ready_id)
    assert v["verified_outcome"] == "in_progress"
    assert v["blocker_age_seconds"] is None
    assert "queued" in v["next_action"]

    running_id = kb.create_task(conn, title="running", assignee="anika")
    kb.claim_task(conn, running_id)
    v = _view(conn, running_id)
    assert v["verified_outcome"] == "in_progress"
    assert v["next_action"] == "worker in progress"


def test_task_summary_dict_surfaces_request_view(conn):
    """The tool-layer summary dict (kanban_list / kanban_show building block)
    carries the same request-level fields as compute_request_view."""
    from tools import kanban_tools as kt

    task_id = kb.create_task(conn, title="t", assignee="anika")
    task = kb.get_task(conn, task_id)
    summary = kt._task_summary_dict(kb, conn, task)
    assert summary["owner"] == "anika"
    assert summary["verified_outcome"] == "in_progress"
    assert summary["blocker_age_seconds"] is None
    assert "queued" in summary["next_action"]
    assert summary["open_spawned_children"] == []
