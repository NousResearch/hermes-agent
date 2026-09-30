"""The unblock-loop breaker counts only re-blocks WITHOUT intervening progress.

``block_recurrences`` exists to stop a cron-unblock <-> re-block spin for the
same cause. It must not escalate a card to ``triage`` when real progress
happened between two blocks of the same kind:

* a review cycle on the card itself (``review_requested`` /
  ``changes_requested``) -- e.g. a reviewer approves a stage and then blocks
  for the owner's release handoff;

A re-block whose reason changed is discounted: it escalates only at
``BLOCK_RECURRENCE_CHANGED_LIMIT``. Card ids cited in reason prose are never
read as progress: completing a card mentioned in an unchanged block (context
or not) does not launder that block. A genuine loop -- the same cause re-filed
with no progress -- still reaches ``triage`` at ``BLOCK_RECURRENCE_LIMIT``, and
a reworded spin still reaches it at the higher limit.
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


def test_new_dependency_after_first_wait_resolved_does_not_triage(kanban_home: Path) -> None:
    """Estate t_7528681d: a flat (unlinked) ``dependency`` block citing card A
    is re-kinded ``needs_input``. A completes, the card is unblocked, makes
    progress, and blocks again on a *different* card B: a changed cause,
    discounted rather than triaged."""
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
        assert task.block_kind == "needs_input"
        assert "block_loop_detected" not in _kinds(conn, tid)
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1].payload
        assert blocked["reason_changed"] is True and "streak_reset" not in blocked

        # ...but if B never lands and the same wait is re-filed, that IS a loop.
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"Producer run failed; waiting on {card_b}.", kind="dependency")
        assert kb.get_task(conn, tid).status == "triage"


def _cycle(conn, tid, reason, kind):
    """unblock (if parked) -> claim -> block; returns the task afterwards."""
    if kb.get_task(conn, tid).status == "blocked":
        assert kb.unblock_task(conn, tid)
    _claim(conn, tid)
    assert kb.block_task(conn, tid, reason=reason, kind=kind)
    return kb.get_task(conn, tid)


@pytest.mark.parametrize("reasons", [
    ["cannot publish to Slack", "cannot publish to Slack"],
    # Whitespace/case noise is the same reason, not a changed cause.
    ["cannot publish to Slack", "  Cannot publish   to SLACK "],
])
def test_genuine_loop_without_progress_still_triages(kanban_home: Path, reasons) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="loop", assignee="worker")
        for reason in reasons:
            task = _cycle(conn, tid, reason, "capability")
        assert (task.status, task.block_recurrences) == ("triage", kb.BLOCK_RECURRENCE_LIMIT)
        loop = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload
        assert "streak_reset" not in loop and "reason_changed" not in loop
        assert loop["limit"] == kb.BLOCK_RECURRENCE_LIMIT


def test_changed_block_reason_is_discounted_not_triaged(kanban_home: Path) -> None:
    """Reviewer probe (round 1): a needs_input block whose cause moved on --
    registry access received, now waiting on a maintenance window -- is a new
    block, not a loop, even with no review cycle or cited card in between."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="publish package", assignee="worker")
        _cycle(conn, tid, "Need registry access from owner before publishing", "needs_input")
        task = _cycle(
            conn, tid,
            "Registry access received and package published; need owner\u2019s maintenance window for activation",
            "needs_input",
        )
        assert (task.status, task.block_recurrences) == ("blocked", 2)
        assert "block_loop_detected" not in _kinds(conn, tid)
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1].payload
        assert blocked["reason_changed"] is True and "streak_reset" not in blocked
        # The SAME changed cause re-filed with no progress is a loop again.
        task = _cycle(
            conn, tid,
            "Registry access received and package published; need owner\u2019s maintenance window for activation",
            "needs_input",
        )
        assert task.status == "triage"


def test_reworded_spin_still_escalates_at_changed_limit(kanban_home: Path) -> None:
    """Estate t_1ee8149d / t_853e5b26 pattern: the same unsatisfiable wait
    reworded on every re-block. Discounted, but still bounded."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="spin", assignee="worker")
        for n in range(1, kb.BLOCK_RECURRENCE_CHANGED_LIMIT):
            task = _cycle(conn, tid, f"cannot publish to Slack (attempt {n})", "capability")
            assert (task.status, task.block_recurrences) == ("blocked", n)
        task = _cycle(conn, tid, "still cannot publish or verify on Slack", "capability")
        assert (task.status, task.block_recurrences) == ("triage", kb.BLOCK_RECURRENCE_CHANGED_LIMIT)
        loop = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload
        assert loop["limit"] == kb.BLOCK_RECURRENCE_CHANGED_LIMIT and loop["reason_changed"] is True


def test_completed_card_cited_as_context_does_not_reset_unchanged_blocker(kanban_home: Path) -> None:
    """Reviewer negative control (round 1): an id mentioned as context in a
    non-dependency block is not the blocking obligation. Completing that card
    is no progress on the missing credentials; the identical re-block triages."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="post to slack", assignee="worker")
        other = kb.create_task(conn, title="unrelated docs", assignee="cody")
        reason = f"Slack credentials are missing. Context only: {other}; this is not a dependency."
        _cycle(conn, tid, reason, "capability")
        _finish(conn, other)
        task = _cycle(conn, tid, reason, "capability")
        assert (task.status, task.block_recurrences) == ("triage", kb.BLOCK_RECURRENCE_LIMIT)
        loop = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload
        assert "streak_reset" not in loop


def test_completed_context_card_in_dependency_block_does_not_reset(kanban_home: Path) -> None:
    """Reviewer negative control (round 2): a dependency block names the real
    prerequisite A and, as context, unrelated docs B. Completing only B while A
    stays open is no progress; the identical re-block triages."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="waiter", assignee="generalist")
        prereq = kb.create_task(conn, title="prerequisite", assignee="cody")
        docs = kb.create_task(conn, title="unrelated docs", assignee="cody")
        reason = f"Blocked on {prereq}. Background context only: {docs}; documentation is not a prerequisite."
        _cycle(conn, tid, reason, "dependency")
        _finish(conn, docs)
        task = _cycle(conn, tid, reason, "dependency")
        assert (task.status, task.block_recurrences) == ("triage", kb.BLOCK_RECURRENCE_LIMIT)
        assert kb.get_task(conn, prereq).status != "done"
        loop = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload
        assert "streak_reset" not in loop


def test_completed_cited_prerequisite_with_identical_reason_still_triages(kanban_home: Path) -> None:
    """Even the cited card itself completing does not reset an identical
    re-block: the unchanged reason says the worker is still stuck on the same
    thing, which is what a human must look at."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="waiter", assignee="generalist")
        prereq = kb.create_task(conn, title="prerequisite", assignee="cody")
        reason = f"waiting on {prereq}"
        _cycle(conn, tid, reason, "dependency")
        _finish(conn, prereq)
        task = _cycle(conn, tid, reason, "dependency")
        assert task.status == "triage"


def test_self_reference_in_reason_is_not_progress(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="self", assignee="worker")
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"{tid} needs a human", kind="dependency")
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        assert kb.block_task(conn, tid, reason=f"{tid} needs a human", kind="dependency")
        assert kb.get_task(conn, tid).status == "triage"
