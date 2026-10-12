"""Review-round ceiling — a review cycle must not run forever on the board.

Measured pathology (12/09/2026): one card accumulated 18 runs and 10 review
rounds for 194 M tokens (~1,61 $), because nothing on the board stops
implementer -> reviewer -> implementer, and each extra round re-runs BOTH agents
with their full context. Past ``REVIEW_ROUND_LIMIT`` the board must park the
card ``blocked`` (needs_input) and emit a ``blocked`` event — the ordinary
notification chain then reaches the origin, instead of a silent stop.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _new_card_in_review(conn, *, title: str = "boucle de revue") -> str:
    task_id = kb.create_task(conn, title=title, assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", summary="implementation prete",
        expected_run_id=implementation.current_run_id,
    )
    return task_id


def _hand_back(conn, task_id: str, *, turn: int, reason: str):
    """One full review round: claim the review, then record the changes request."""
    review = kb.claim_review_task(conn, task_id, claimer=f"reviewer:{turn}")
    assert review is not None
    return kb.request_changes(conn, task_id, reason=reason, expected_run_id=review.current_run_id)


def _resume(conn, task_id: str, *, turn: int) -> None:
    implementation = kb.claim_task(conn, task_id, claimer=f"builder:{turn}")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", summary=f"tour {turn}",
        expected_run_id=implementation.current_run_id,
    )


def _count(conn, task_id: str, kind: str) -> int:
    return len([event for event in kb.list_events(conn, task_id) if event.kind == kind])


def test_rounds_up_to_the_limit_still_hand_back_to_the_implementer(conn):
    task_id = _new_card_in_review(conn)
    for turn in range(1, kb.REVIEW_ROUND_LIMIT + 1):
        ok, implementer = _hand_back(conn, task_id, turn=turn, reason=f"retour {turn}")
        assert ok is True
        assert implementer == "builder"
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status in ("ready", "todo")
        assert task.current_run_id is None
        if turn < kb.REVIEW_ROUND_LIMIT:
            _resume(conn, task_id, turn=turn + 1)

    assert _count(conn, task_id, "changes_requested") == kb.REVIEW_ROUND_LIMIT
    assert _count(conn, task_id, "blocked") == 0


def test_the_round_after_the_limit_parks_the_card_for_a_human(conn):
    task_id = _new_card_in_review(conn)
    for turn in range(1, kb.REVIEW_ROUND_LIMIT + 1):
        assert _hand_back(conn, task_id, turn=turn, reason=f"retour {turn}")[0] is True
        _resume(conn, task_id, turn=turn + 1)

    ok, detail = _hand_back(
        conn, task_id, turn=kb.REVIEW_ROUND_LIMIT + 1, reason="encore un defaut de meme famille",
    )
    assert ok is False
    assert str(kb.REVIEW_ROUND_LIMIT) in str(detail)

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert task.current_run_id is None

    # Le tour refuse n'est PAS enregistre : la carte ne peut pas repartir pour un
    # tour de plus, meme apres un deblocage humain (le plafond reste atteint).
    assert _count(conn, task_id, "changes_requested") == kb.REVIEW_ROUND_LIMIT
    blocked = [event for event in kb.list_events(conn, task_id) if event.kind == "blocked"][-1]
    assert blocked.payload["limit"] == kb.REVIEW_ROUND_LIMIT
    assert "encore un defaut de meme famille" in blocked.payload["reason"]
    last_run = kb.list_runs(conn, task_id)[-1]
    assert last_run.outcome == "changes_requested"
    assert last_run.status == "blocked"
