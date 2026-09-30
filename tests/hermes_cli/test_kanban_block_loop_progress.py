"""The unblock-loop breaker counts only re-blocks WITHOUT intervening progress.

``block_recurrences`` exists to stop a cron-unblock <-> re-block spin for the
same cause. It must not escalate a card to ``triage`` when real progress
happened between two blocks of the same kind:

* a review cycle on the card itself (``review_requested`` /
  ``changes_requested``) -- e.g. a reviewer approves a stage and then blocks
  for the owner's release handoff;
* the card the previous block was waiting on (cited by id in its reason)
  completed -- the wait was satisfied and a *new* wait began.

A genuine loop -- the same kind re-filed after an unblock with neither kind of
progress, even when the reason text is reworded -- still reaches ``triage``.
"""

from __future__ import annotations

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


def _claim(conn, tid, claimer="worker"):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=? AND status<>'running'", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=claimer)
    assert claimed is not None
    return claimed.current_run_id


def _finish(conn, tid):
    _claim(conn, tid)
    assert kb.complete_task(conn, tid, result="done")


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


def test_owner_handoff_after_review_approval_does_not_triage(kanban_home: Path) -> None:
    """Estate t_7aeb849a: block (waiting on review admission) -> unblock ->
    request_review -> reviewer claims and approves, then blocks for the owner's
    release handoff. Same kind, but a review cycle happened in between."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="saxzi nav", assignee="generalist")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason="Waiting on Gurney review of candidate", kind="needs_input")
        assert kb.get_task(conn, tid).block_recurrences == 1

        assert kb.unblock_task(conn, tid)
        run_id = _claim(conn, tid)
        assert kb.request_review(conn, tid, summary="stage 1 handoff", reviewer="gurney", expected_run_id=run_id)
        assert kb.claim_review_task(conn, tid, claimer="gurney") is not None

        assert kb.block_task(
            conn, tid, reason="Stage-1 APPROVED; owner handoff: John resumes for release", kind="needs_input",
        )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert (task.block_kind, task.block_recurrences) == ("needs_input", 1)
        assert "block_loop_detected" not in _kinds(conn, tid)
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1].payload
        assert blocked["streak_reset"] == ["review_requested"]
        # The handoff resumes the review lane, exactly as before.
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "review"


def test_new_dependency_after_cited_blocker_completed_does_not_triage(kanban_home: Path) -> None:
    """Estate t_7528681d: a flat (unlinked) ``dependency`` block citing card A
    is re-kinded ``needs_input``. A completes, the card is unblocked, makes
    progress, and blocks again on a *different* card B."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="activate backup fix", assignee="generalist")
        card_a = kb.create_task(conn, title="publish PR", assignee="cody")
        card_b = kb.create_task(conn, title="vanished sqlite fix", assignee="cody")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"No upstream PR yet; Cody's flat card {card_a} owns it.", kind="dependency")
        assert (kb.get_task(conn, tid).status, kb.get_task(conn, tid).block_kind) == ("blocked", "needs_input")

        _finish(conn, card_a)
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"Producer run failed; waiting on {card_b}.", kind="dependency")

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert (task.block_kind, task.block_recurrences) == ("needs_input", 1)
        assert "block_loop_detected" not in _kinds(conn, tid)
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1].payload
        assert blocked["streak_reset"] == [f"resolved:{card_a}"]

        # ...but if B then never lands and the same wait is re-filed, that IS a loop.
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"Still waiting on {card_b}.", kind="dependency")
        assert kb.get_task(conn, tid).status == "triage"


@pytest.mark.parametrize("reasons", [
    ["cannot publish to Slack", "cannot publish to Slack"],
    # Rewording alone is not progress (estate t_1ee8149d / t_853e5b26 pattern).
    ["cannot publish to Slack", "still cannot publish or verify on Slack"],
])
def test_genuine_loop_without_progress_still_triages(kanban_home: Path, reasons) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="loop", assignee="worker")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=reasons[0], kind="capability")
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=reasons[1], kind="capability")
        task = kb.get_task(conn, tid)
        assert (task.status, task.block_recurrences) == ("triage", kb.BLOCK_RECURRENCE_LIMIT)
        loop = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload
        assert "streak_reset" not in loop


def test_cited_card_still_open_or_completed_earlier_is_not_progress(kanban_home: Path) -> None:
    """Only a cited card completing *between* the two blocks counts; one that
    was already done before the first block, or is still open, does not."""
    with kbc.connect_closing() as conn:
        done_before = kb.create_task(conn, title="old", assignee="cody")
        still_open = kb.create_task(conn, title="open", assignee="cody")
        _finish(conn, done_before)
        tid = kb.create_task(conn, title="waiter", assignee="generalist")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"waiting on {done_before} and {still_open}", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"waiting on {still_open}", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"


def test_self_reference_in_reason_is_not_progress(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="self", assignee="worker")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"{tid} needs a human", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"{tid} needs a human", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"
