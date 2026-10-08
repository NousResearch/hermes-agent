"""Review-lane self-review gate (t_d3290431).

The incident: an implementer called ``request_review`` without ``reviewer=``
on a first review. The kernel left ``assignee`` = the implementer, the
dispatcher spawns the row's assignee for the review lane, and the implementer
ran its own "independent review APPROVED". These tests pin the fail-closed
contract at both layers:

* Kernel: a first-review ``request_review`` without ``reviewer=`` refuses on
  any card with implementer provenance; only the never-claimed operator flow
  (``kanban create --assignee <reviewer>`` -> request-review) may infer
  ``reviewer := assignee``.
* Dispatch: a review row whose assignee equals the latest implementer can
  never be spawned under that profile (``self_review:`` guard reason), while
  a crashed reviewer stays re-spawnable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Kernel: first review without reviewer= refuses on implementer cards
# ---------------------------------------------------------------------------


def test_first_request_review_without_reviewer_refuses_when_implementer_known(
    kanban_home: Path,
) -> None:
    """The t_d3290431 incident replayed at kernel level: sab-coder-style
    implementer claims a card and calls request_review without reviewer=. The
    transition must refuse instead of leaving assignee = implementer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="sab-coder")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        ok, reason = kb.request_review(
            conn, tid, summary="done", with_reason=True,
            expected_run_id=claimed.current_run_id,
        )
        assert ok is False
        assert "self-review" in reason

        row = kb.get_task(conn, tid)
        # Nothing moved: still owned by the implementer, still running.
        assert row.status == "running"
        assert row.assignee == "sab-coder"
        # No review_requested handoff was recorded.
        assert [
            e for e in kb.list_events(conn, tid) if e.kind == "review_requested"
        ] == []


def test_first_request_review_infers_dedicated_reviewer_assignee(
    kanban_home: Path,
) -> None:
    """Operator flow preserved: a never-claimed card whose assignee is the
    intended reviewer (``kanban create --assignee <reviewer>`` ->
    ``request-review``) infers reviewer := assignee."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review-bound", assignee="sab-reviewer")
        assert kb.request_review(conn, tid, summary="ready") is True

        row = kb.get_task(conn, tid)
        assert row.status == "review"
        assert row.assignee == "sab-reviewer"
        handoff = [
            e for e in kb.list_events(conn, tid) if e.kind == "review_requested"
        ][-1]
        assert handoff.payload["reviewer"] == "sab-reviewer"
        # No implementer provenance exists on an operator-created card.
        assert handoff.payload["implementer"] is None


# ---------------------------------------------------------------------------
# Dispatch: the review lane never spawns the implementer's profile
# ---------------------------------------------------------------------------


def _land_review_row_with_implementer_assignee(conn, task_id: str, run_profile: str) -> None:
    """Force the incident geometry directly (legacy rows written before the
    kernel fix): a review row whose assignee IS the implementer."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'review', assignee = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = ?",
            (run_profile, task_id),
        )
        kb._append_event(
            conn, task_id, "review_requested",
            {"summary": "done", "implementer": run_profile, "reviewer": None},
        )


def test_review_lane_refuses_to_spawn_the_implementer_profile(
    kanban_home: Path,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="sab-coder")
        kb.claim_task(conn, tid)
        _land_review_row_with_implementer_assignee(conn, tid, "sab-coder")

        reason = kbd.check_respawn_guard(conn, tid, lane="review")
        assert reason is not None
        assert reason.startswith("self_review:sab-coder")

        result = kbd.dispatch_once(conn, dry_run=True)
        # Never surfaced as a spawnable review row under the implementer.
        assert not [t for t in result.spawned if t[0] == tid]


def test_review_lane_allows_independent_reviewer_and_reassign_escape(
    kanban_home: Path,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="sab-coder")
        kb.claim_task(conn, tid)
        # Fixed handoff: implementer provenance kept, reviewer reassigned.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', assignee = 'sab-reviewer' "
                "WHERE id = ?",
                (tid,),
            )
            kb._append_event(
                conn, tid, "review_requested",
                {"summary": "done", "implementer": "sab-coder",
                 "reviewer": "sab-reviewer"},
            )
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        # Operator escape hatch: reassigning to any other profile clears it.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee = 'human-ops' WHERE id = ?", (tid,),
            )
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_review_lane_still_respawns_a_crashed_reviewer(kanban_home: Path) -> None:
    """A reviewer that crashed mid-review owns the latest run but is NOT an
    implementer: the guard must not strand its re-spawn."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="sab-reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET outcome = 'crashed', ended_at = 1 WHERE id = ?",
                (run_id,),
            )
            conn.execute(
                "UPDATE tasks SET status = 'review', current_run_id = NULL, "
                "claim_lock = NULL WHERE id = ?",
                (tid,),
            )
        # No review_requested handoff: fallback runs are terminal-implementer
        # outcomes only, so a crashed run must not trip the guard.
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_dispatch_never_spawns_self_review_even_when_kernel_bypassed(
    kanban_home: Path,
) -> None:
    """End-to-end: a legacy review row (assignee = implementer) survives a
    full dispatch tick without a spawn under the implementer, and the refusal
    leaves board evidence."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="sab-coder")
        kb.claim_task(conn, tid)
        _land_review_row_with_implementer_assignee(conn, tid, "sab-coder")

        import hermes_cli.profiles as profmod
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(profmod, "profile_exists", lambda _name: True)
            result = kbd.dispatch_once(conn, dry_run=False)
        assert not [t for t in result.spawned if t[0] == tid]
        events = [
            e.payload if isinstance(e.payload, dict) else json.loads(e.payload or "{}")
            for e in kb.list_events(conn, tid)
            if e.kind == "respawn_guarded" and e.payload
        ]
        assert any(str(ev.get("reason", "")).startswith("self_review:") for ev in events)
