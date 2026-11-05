"""A review gate whose worker died before its first turn must be recoverable.

The stranded shape observed on ``t_5c02fad2``: the reviewer claim was booked ``crashed`` and its
claim released, but the card came back to ``review`` still naming that dead run
(``current_run_id``). Every operator transition then refused — ``kanban_block`` and
``kanban_request_review`` both only accepted ``running``/``ready`` — while ``unblock`` returned the
card to the same under-provisioned reviewer and the same crash. These tests pin the recovery:

* ``block_task`` parks a claim-less ``review`` card (``kind='capability'``) and ``unblock`` resumes
  it into the review lane, not the implementer lane;
* ``request_review`` re-routes that card to a named, provisioned reviewer and keeps the implementer
  provenance intact;
* both still refuse when a caller asserts run ownership, so a worker's CAS fence is unchanged;
* a live claim is never cleared by either widening.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _row(conn, tid):
    return conn.execute(
        "SELECT status, block_kind, block_recurrences, current_run_id, assignee "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _latest_event(conn, tid, kind):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (tid, kind),
    ).fetchone()
    import json
    return json.loads(row["payload"]) if row else None


def _stranded_review_card(conn, *, reviewer: str = "portfolio-lead") -> tuple[str, int]:
    """A review card whose reviewer worker died in preload: review status, dead run, no claim."""
    tid = kb.create_task(conn, title="gate", body="review me", assignee="nico")
    implementation = kb.claim_task(conn, tid)
    assert implementation is not None
    assert kb.request_review(conn, tid, summary="implementation done",
                             expected_run_id=implementation.current_run_id, reviewer=reviewer)
    review = kb.claim_review_task(conn, tid)
    assert review is not None and review.current_run_id
    # The worker's first turn never started: the attempt is booked, the claim released, and the
    # card returns to ``review`` still naming the dead run — the stranded state.
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET status='crashed', outcome='crashed', ended_at=? WHERE id=?",
                     (now, review.current_run_id))
        conn.execute("UPDATE tasks SET status='review', claim_lock=NULL, claim_expires=NULL, "
                     "worker_pid=NULL, worker_started_at=NULL WHERE id=?", (tid,))
    row = _row(conn, tid)
    assert row["status"] == "review" and row["current_run_id"] == review.current_run_id
    return tid, review.current_run_id


def test_block_task_parks_a_stranded_review_card_and_unblock_resumes_the_review_lane(kanban_home):
    with kbc.connect() as conn:
        tid, _run = _stranded_review_card(conn)
        assert kb.block_task(conn, tid, kind="capability",
                             reason="ported: sdlc-review missing") is True
        row = _row(conn, tid)
        assert row["status"] == "blocked" and row["block_kind"] == "capability"
        assert row["current_run_id"] is None

        # The block event remembers where the card came from, so unblock returns it to review.
        assert kb.unblock_task(conn, tid) is True
        assert _row(conn, tid)["status"] == "review"


def test_request_review_reroutes_a_stranded_review_card_to_a_provisioned_reviewer(kanban_home):
    with kbc.connect() as conn:
        tid, stale_run = _stranded_review_card(conn)
        assert kb.request_review(conn, tid, reviewer="engineering-lead",
                                 summary="re-routed after the reviewer could not start") is True
        row = _row(conn, tid)
        assert row["status"] == "review" and row["assignee"] == "engineering-lead"
        handoff = _latest_event(conn, tid, "review_requested")
        assert handoff is not None
        assert handoff["reviewer"] == "engineering-lead"
        # The implementer provenance survives the re-route: request_changes still routes back to
        # the profile that did the work, not to the reviewer that never started.
        assert handoff["implementer"] == "nico"
        assert stale_run  # the dead attempt is still what the card was recovered from


def test_a_caller_asserting_run_ownership_is_still_refused(kanban_home):
    with kbc.connect() as conn:
        tid, stale_run = _stranded_review_card(conn)
        assert kb.block_task(conn, tid, kind="capability", reason="stale worker",
                             expected_run_id=stale_run) is False
        assert kb.request_review(conn, tid, reviewer="engineering-lead",
                                 expected_run_id=stale_run) is False
        assert _row(conn, tid)["status"] == "review"


def test_a_live_claim_is_never_cleared_by_the_rescue_paths(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid, _run = _stranded_review_card(conn)
        monkeypatch.setattr(kb, "_claim_is_live", lambda _row: True)
        assert kb.block_task(conn, tid, kind="capability", reason="still running") is False
        assert kb.request_review(conn, tid, reviewer="engineering-lead") is False
        assert _row(conn, tid)["assignee"] == "portfolio-lead"
