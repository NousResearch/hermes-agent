"""A worker that cannot resolve its preload parks the card instead of stranding it.

``t_5c02fad2``: four managed workers on ``portfolio-lead`` died in ``_run_task_turns`` with
``ValueError: kanban_skills_unavailable`` and the card stayed ``running`` each time, so an
orchestrator saw neither the cause nor a state it could act on. The guard now blocks the card with
a reason naming the missing identifiers *before* it raises, and the raise names them too, so the
worker's own crash record (``worker_lost`` cause) carries the diagnosis.

These are unit tests over the real board: a claimed task, a real run, the real block transition.

Each case runs through :func:`_report`, which reduces a failure to ``Type: message``. The shared
test interpreter lives under the real Hermes home, so formatting a deep traceback trips
``tests/home_io_guard.py`` and hides the real error; a one-line message keeps failures readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

# Deliberately not ``sdlc-review``: this test proves the *absence* path, so the identifier must be
# one nothing can resolve (the repo's bundled skills stay visible from a bare home).
ABSENT_SKILL = "absent-skill-probe-t98"


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    (tmp_path / "home").mkdir(exist_ok=True)
    kb.create_board("default")
    return tmp_path


def _claimed(conn, *, assignee: str = "portfolio-lead"):
    tid = kb.create_task(conn, title="Tradieo internal route", body="review the note",
                         assignee="nico")
    implementation = kb.claim_task(conn, tid)
    assert implementation is not None
    assert kb.request_review(conn, tid, summary="note ready",
                             expected_run_id=implementation.current_run_id, reviewer=assignee)
    reviewed = kb.claim_review_task(conn, tid)
    assert reviewed is not None
    return reviewed


def _context(task, path: Path) -> dict:
    """The claim frame a worker carries; ``db`` is the board file, exactly as the dispatcher sets it."""
    return {"task_id": task.id, "run_id": task.current_run_id, "claim_lock": task.claim_lock,
            "db": str(kb.kanban_db_path()), "profile": task.assignee}


def _latest(conn, tid, kind):
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind=? "
                       "ORDER BY id DESC LIMIT 1", (tid, kind)).fetchone()
    return json.loads(row["payload"]) if row else None


def _report(case) -> str:
    try:
        case()
    except Exception as exc:  # surfaced as the assertion message
        return f"{type(exc).__name__}: {exc}"
    return ""


def test_an_unsatisfiable_preload_blocks_the_card_with_the_missing_skills_named(board):
    from gateway.session_kanban import _park_unsatisfiable_preload

    def case():
        with connect() as conn:
            task = _claimed(conn)
            run_id = task.current_run_id
            reason = _park_unsatisfiable_preload(_context(task, board), [ABSENT_SKILL])
            parked = kb.get_task(conn, task.id)
            assert parked is not None
            assert parked.status == "blocked" and parked.block_kind == "capability", (
                parked.status, parked.block_kind)
            assert parked.current_run_id is None, parked.current_run_id
            assert ABSENT_SKILL in reason and "portfolio-lead" in reason, reason

            event = _latest(conn, task.id, "blocked")
            assert event is not None and event["kind"] == "capability", event
            assert ABSENT_SKILL in event["reason"], event
            # The block remembers the lane it came from: unblock re-drives the review, not the build.
            assert event["source_status"] == "review", event
            run = conn.execute("SELECT status, outcome, summary FROM task_runs WHERE id=?",
                               (run_id,)).fetchone()
            assert run["status"] == "blocked" and run["outcome"] == "blocked", dict(run)

    assert _report(case) == ""


def test_the_raise_names_the_missing_skill_and_the_profile(board):
    """The managed worker's cause quotes the exception: it must carry the diagnosis."""
    import gateway.session_kanban as sk

    def case():
        with connect() as conn:
            task = _claimed(conn)
            agent = type("Agent", (), {"session_id": "local-probe"})()
            context = _context(task, board)
            context["skills"] = [ABSENT_SKILL]
            try:
                sk._run_task_turns(agent, {"text": "work the card"}, [], context)
            except ValueError as exc:
                message = str(exc)
                assert message.startswith("kanban_skills_unavailable"), message
                assert ABSENT_SKILL in message, message
            else:
                raise AssertionError("the preload guard did not refuse an unsatisfiable preload")
            parked = kb.get_task(conn, task.id)
            assert parked is not None and parked.status == "blocked", parked.status

    assert _report(case) == ""
