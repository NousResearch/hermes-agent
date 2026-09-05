"""Model-B reject fallback for ``kanban_request_changes``.

Model A is the first-class review handoff: a single card passes ``running ->
review`` via ``request_review``, the reviewer claims it from the ``review``
column with ``claim_review_task``, and ``request_changes`` recovers the
implementer from a ``review_requested`` event.

Model B is the pre-created review child: the reviewer is dispatched as an
ordinary task claimed from ``ready`` — there is no ``review_requested`` event
and the claimed payload carries no ``source_status=review``. Historically
``request_changes`` hard-returned False on both, so that reviewer had no legal
reject verb (only ``kanban_complete`` or ``kanban_block``). These tests pin the
fallback that gives Model B a native reject path:

* derive the implementer from the *newest-done* NON-REVIEWER parent (the
  artifact under review) and route a ``[rework]`` card back to it,
* link the rework card as a PARENT of the review card so the next review pass
  re-promotes when rework lands,
* emit a real ``changes_requested`` event on the review card.

Zero candidates and completed_at ties must still resolve to False — the gate
never fabricates provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _parents(conn, tid):
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (tid,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def _rework_cards(conn):
    rows = conn.execute(
        "SELECT id, title, assignee, status FROM tasks "
        "WHERE title LIKE '[rework]%' ORDER BY id",
    ).fetchall()
    return rows


def _complete_done(conn, tid, claimed):
    assert kb.complete_task(
        conn, tid, summary="done", expected_run_id=claimed.current_run_id,
    ) is True
    done = kb.get_task(conn, tid)
    assert done is not None and done.status == "done"
    assert done.completed_at is not None
    return done


# ---------------------------------------------------------------------------
# Model B: pre-created review child rejects via the fallback
# ---------------------------------------------------------------------------


def test_model_b_precreated_review_child_rejects_via_fallback(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        # The artifact under review (implementation card) is done.
        artifact = kb.create_task(conn, title="build the widget", assignee="worker")
        _complete_done(conn, artifact, kb.claim_task(conn, artifact))

        # Pre-created review child: review profile as assignee, parent is the
        # completed artifact, and NO review_requested event exists.
        review_id = kb.create_task(
            conn, title="review the widget", assignee="reviewer",
            parents=[artifact],
        )
        claimed = kb.claim_task(conn, review_id)  # claimed from ready, not review
        assert claimed is not None
        assert _events(conn, review_id, kind="review_requested") == []
        assert kb.get_task(conn, review_id).status == "running"

        ok, implementer = kb.request_changes(
            conn, review_id, reason="the widget needs fixing",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True
        assert implementer == "worker"

        # A rework card was created and routed back to the implementer.
        reworks = _rework_cards(conn)
        assert len(reworks) == 1
        rework = reworks[0]
        assert rework["title"].startswith("[rework] build the widget")
        assert rework["assignee"] == "worker"

        # Rework's parent is the artifact; the rework is a PARENT of the review.
        assert _parents(conn, rework["id"]) == [artifact]
        assert rework["id"] in _parents(conn, review_id)

        # A real changes_requested event rides the review card.
        cr = _events(conn, review_id, kind="changes_requested")
        assert len(cr) == 1
        payload = cr[0][1]
        assert payload["implementer"] == "worker"
        assert payload["reason"] == "the widget needs fixing"
        assert payload["rework"] == rework["id"]

        # The review run is closed and the card sits in the dependency lane.
        review_now = kb.get_task(conn, review_id)
        assert review_now.status in ("todo", "ready")
        assert review_now.current_run_id is None


# ---------------------------------------------------------------------------
# Model A regression: same-card review is unchanged
# ---------------------------------------------------------------------------


def test_model_a_same_card_review_unchanged(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="same card", assignee="worker")
        impl = kb.claim_task(conn, tid)
        assert impl is not None
        assert kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer",
            expected_run_id=impl.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "review"

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        ok, implementer = kb.request_changes(
            conn, tid, reason="rework it",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, "worker")
        assert kb.get_task(conn, tid).status in ("todo", "ready")

        # The same-card path must NOT spawn a rework card.
        assert _rework_cards(conn) == []
        assert len(_events(conn, tid, kind="changes_requested")) == 1


# ---------------------------------------------------------------------------
# Gate preserved: zero candidates and ties still return False
# ---------------------------------------------------------------------------


def test_model_b_zero_candidates_still_false(kanban_home: Path) -> None:
    with kb.connect() as conn:
        # A review card whose parent set has no done NON-REVIEWER parent.
        review_id = kb.create_task(conn, title="orphan review", assignee="reviewer")
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None

        ok, reason = kb.request_changes(
            conn, review_id, reason="changes",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is False
        assert reason is not None and "parent" in reason

        # Nothing was fabricated: no rework, no event, task untouched.
        assert _rework_cards(conn) == []
        assert _events(conn, review_id, kind="changes_requested") == []
        assert kb.get_task(conn, review_id).status == "running"


def test_model_b_tie_candidates_still_false(kanban_home: Path) -> None:
    with kb.connect() as conn:
        p1 = kb.create_task(conn, title="impl one", assignee="worker-one")
        p1c = kb.claim_task(conn, p1)
        assert p1c is not None
        _complete_done(conn, p1, p1c)

        p2 = kb.create_task(conn, title="impl two", assignee="worker-two")
        p2c = kb.claim_task(conn, p2)
        assert p2c is not None
        _complete_done(conn, p2, p2c)

        # Force an exact completed_at tie so selection is ambiguous.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completed_at = 1000 WHERE id IN (?, ?)",
                (p1, p2),
            )

        review_id = kb.create_task(
            conn, title="tied review", assignee="reviewer", parents=[p1, p2],
        )
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None

        ok, reason = kb.request_changes(
            conn, review_id, reason="changes",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is False
        assert reason is not None and "ambiguous" in reason
        assert _rework_cards(conn) == []
        assert _events(conn, review_id, kind="changes_requested") == []


def test_derive_idempotency_key_deterministic_and_truncated(
    kanban_home: Path,
) -> None:
    key_ab = kb._derive_idempotency_key(["t_a", "t_b"])
    assert len(key_ab) == 32
    # Order-independent and stable for the same single-parent set.
    assert key_ab == kb._derive_idempotency_key(["t_b", "t_a"])
    assert kb._derive_idempotency_key(["t_a"]) == kb._derive_idempotency_key(["t_a"])
    assert key_ab != kb._derive_idempotency_key(["t_a"])