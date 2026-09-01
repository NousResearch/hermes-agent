"""`_record_task_failure` must persist the dying attempt's summary.

Both branches matter. The normal branch is the ordinary retry; the
``gave_up`` branch is the circuit breaker tripping, and a task that tripped
the breaker still owes its summary to whoever unblocks it later.
"""

from __future__ import annotations

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


def _last_run_summary(conn, task_id: str) -> str | None:
    row = conn.execute(
        "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row[0] if row else None


def test_summary_is_persisted_on_the_normal_failure_branch(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="budget death")
        kb.claim_task(conn, tid)

        kb._record_task_failure(
            conn,
            tid,
            error="Iteration budget exhausted (150/150)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            summary="Wrote the migration, tests still red on table 3.",
        )

        assert _last_run_summary(conn, tid) == (
            "Wrote the migration, tests still red on table 3."
        )


def test_summary_is_persisted_when_the_breaker_trips(kanban_home: Path) -> None:
    # failure_limit=1 forces the ``gave_up`` branch on the first failure.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="budget death, breaker trips")
        kb.claim_task(conn, tid)

        kb._record_task_failure(
            conn,
            tid,
            error="Iteration budget exhausted (150/150)",
            outcome="timed_out",
            failure_limit=1,
            release_claim=True,
            end_run=True,
            summary="Got as far as the schema diff.",
        )

        assert _last_run_summary(conn, tid) == "Got as far as the schema diff."


def test_a_summary_dropped_by_the_cas_is_logged(kanban_home, monkeypatch, caplog) -> None:
    """`_end_run` returns None when its compare-and-swap loses to the crash
    detector. The run is already closed and the summary is discarded; that is
    rare and acceptable, but it must not be invisible."""
    import logging

    monkeypatch.setattr(kb, "_end_run", lambda *a, **kw: None)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cas loser")
        kb.claim_task(conn, tid)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            kb._record_task_failure(
                conn,
                tid,
                error="Iteration budget exhausted (150/150)",
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                summary="work that will not be persisted",
            )

    assert "dropping a" in caplog.text
    assert tid in caplog.text


def test_no_warning_when_there_was_no_summary_to_drop(kanban_home, monkeypatch, caplog) -> None:
    """`run_id is None` is routine when there was never an open run to close.
    Only a *lost summary* is worth a warning."""
    import logging

    monkeypatch.setattr(kb, "_end_run", lambda *a, **kw: None)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no run, no summary")
        kb.claim_task(conn, tid)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            kb._record_task_failure(
                conn,
                tid,
                error="pid 4242 not alive",
                outcome="crashed",
                release_claim=True,
                end_run=True,
            )

    assert "dropping a" not in caplog.text


def test_omitting_the_summary_still_works(kanban_home: Path) -> None:
    """The parameter is optional: every existing caller passes nothing."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no summary available")
        kb.claim_task(conn, tid)

        kb._record_task_failure(
            conn,
            tid,
            error="pid 4242 not alive",
            outcome="crashed",
            release_claim=True,
            end_run=True,
        )

        assert _last_run_summary(conn, tid) is None
