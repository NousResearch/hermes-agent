"""A worker that dies of budget exhaustion has still done work — its final message
must survive on the closed run.

The chain this covers, which silently discarded the findings:

    turn_finalizer._record_kanban_budget_exhausted
      -> kanban_db_dispatch._record_task_failure      (had no partial_summary param)
        -> kanban_db._end_run(summary=...)             (already accepted summary)

``_handle_max_iterations`` asks the model to summarise what it completed before the
budget ran out, and that summary was dropped on the floor: the run row's ``summary``
stayed NULL and the operator was left with a timed-out card and no account of what
happened. Observed live as runs 24/25 of the hermes-team-arcade board, both NULL with
zero partial_unverified comments.

Each test here fails on the pre-fix code (the parameter did not exist / the column
stayed NULL) and passes after.
"""

from __future__ import annotations

import logging
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


def _claimed_task(conn) -> str:
    """A task with an OPEN run, which is what ``_end_run`` closes."""
    tid = kb.create_task(conn, title="budget-exhausted worker", assignee="code_agent")
    kb.claim_task(conn, tid, claimer="host:w1")
    return tid


def _run_summary(conn, tid: str) -> str | None:
    row = conn.execute(
        "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert row is not None, "no run was closed"
    return row["summary"]


def _events(conn, tid: str) -> list:
    return conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()


def test_partial_summary_is_recorded_on_the_closed_run(kanban_home):
    """Dispatch layer: the summary reaches ``task_runs.summary`` and the event payload."""
    with kbc.connect() as conn:
        tid = _claimed_task(conn)
        assert kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (60/60)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            partial_summary="Wrote 3 of 5 files; tests not yet run.",
        ) is False

        assert _run_summary(conn, tid) == "Wrote 3 of 5 files; tests not yet run."
        event = [e for e in _events(conn, tid) if e["kind"] == "timed_out"]
        assert event, "no timed_out event was written"
        assert "Wrote 3 of 5 files" in event[-1]["payload"]


def test_no_partial_summary_when_the_worker_left_nothing(kanban_home):
    """Negative control: an empty summary must not overwrite the column or fake a payload.

    Without this the fix would be indistinguishable from always writing something.
    """
    with kbc.connect() as conn:
        tid = _claimed_task(conn)
        kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (60/60)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            partial_summary=None,
        )
        assert _run_summary(conn, tid) is None, "a silent worker must not gain a summary"


def test_budget_exhausted_wrapper_carries_the_workers_final_message(kanban_home):
    """Finalizer layer: ``_record_kanban_budget_exhausted`` forwards the summary it is
    given. Before the fix the parameter did not exist, so the message could not arrive
    here at all — this is the seam where it was lost.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    with kbc.connect() as conn:
        tid = _claimed_task(conn)
        _record_kanban_budget_exhausted(
            tid, 60, 60, logging.getLogger("test"),
            partial_summary="Half the mission done: recon complete, fix not applied.",
        )
        assert _run_summary(conn, tid) == "Half the mission done: recon complete, fix not applied."


@pytest.mark.parametrize("value,expected", [
    ("what I completed so far", "what I completed so far"),
    ("  padded  ", "padded"),
    (None, None),
    ("", None),
    ("   \n  ", None),
    ([{"type": "text", "text": "structured block"}], "structured block"),
])
def test_extract_partial_summary(value, expected):
    """Extraction: text is normalised, and genuine emptiness stays None so the negative
    control above holds."""
    from agent.turn_finalizer import _extract_partial_summary

    assert _extract_partial_summary(value) == expected


def test_extract_partial_summary_is_capped():
    """A long final message cannot exceed what the run/event columns are dimensioned for."""
    from agent.turn_finalizer import _extract_partial_summary

    summary = _extract_partial_summary("x" * 10000)
    assert summary is not None
    assert len(summary) == 4096
