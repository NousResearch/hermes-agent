"""One occurrence runs its work once: the ledger refuses a re-fire of a COMPLETED occurrence.

``scheduled_instant`` is the occurrence's exact identity (``cron/occurrences.py``). Firing an
occurrence that already completed would run the job's side effects a second time and write a second
terminal row for work that is already done — the ledger, the seam every fire crosses to record its
attempt, refuses it.

The boundary matters as much as the refusal, and is pinned here as its own contract: the ledger
deliberately holds several attempts per occurrence. A retry after a transient failure, the
restore-once of a slot whose owner is provably gone (#107485), and a manual run with no occurrence
identity must all stay fireable.
"""

from __future__ import annotations

import sqlite3

import pytest

INSTANT = "2026-09-13T20:00:00+00:00"


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


def test_second_fire_of_a_completed_occurrence_is_refused(executions):
    from cron.executions import DuplicateFireAttempt

    first = executions.create_execution("job-done", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(first["id"])
    executions.finish_execution(first["id"], success=True)

    with pytest.raises(DuplicateFireAttempt) as excinfo:
        executions.create_execution("job-done", source="builtin", scheduled_instant=INSTANT)

    assert excinfo.value.existing_id == first["id"]
    assert excinfo.value.existing_status == "completed"
    assert INSTANT in str(excinfo.value)
    # Nothing was written for the refused fire.
    assert len(executions.list_executions(job_id="job-done")) == 1


def test_a_non_terminal_sibling_is_a_retry_not_a_duplicate(executions):
    """A live attempt for the occurrence must not block the fire path's own retry handling."""
    first = executions.create_execution("job-live", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(first["id"])

    retry = executions.create_execution("job-live", source="builtin", scheduled_instant=INSTANT)

    assert retry["status"] == "claimed"
    assert executions.get_execution(first["id"])["status"] == "running"


def test_a_failed_or_unknown_attempt_stays_eligible(executions):
    for status in ("failed", "unknown"):
        first = executions.create_execution(
            "job-retry", source="builtin", scheduled_instant=INSTANT)
        executions.mark_execution_running(first["id"])
        executions.finish_execution(first["id"], success=False, error="transient")
        if status == "unknown":
            with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
                conn.execute("UPDATE executions SET status='unknown' WHERE id=?", (first["id"],))

        retry = executions.create_execution(
            "job-retry", source="builtin", scheduled_instant=INSTANT)

        assert retry["status"] == "claimed"


def test_a_dead_owners_attempt_does_not_block_the_restore_retry(executions):
    """#107485: a slot whose owner is provably gone is restored ONCE and fired again."""
    executions.create_execution("job-restore", source="builtin", scheduled_instant=INSTANT)

    restored = executions.create_execution(
        "job-restore", source="builtin", scheduled_instant=INSTANT)

    assert restored["status"] == "claimed"


def test_a_different_occurrence_of_the_same_job_still_fires(executions):
    executions.create_execution("job-next", source="builtin", scheduled_instant=INSTANT)

    later = executions.create_execution(
        "job-next", source="builtin", scheduled_instant="2026-09-13T21:00:00+00:00")

    assert later["status"] == "claimed"


def test_a_manual_run_without_an_instant_is_never_refused(executions):
    first = executions.create_execution("job-manual", source="direct", scheduled_instant=INSTANT)
    executions.mark_execution_running(first["id"])
    executions.finish_execution(first["id"], success=True)

    manual = executions.create_execution("job-manual", source="direct")

    assert manual["status"] == "claimed"
    assert manual["scheduled_instant"] is None


def test_superseded_is_a_terminal_status_a_later_write_cannot_reopen(executions):
    row = executions.create_execution("job-superseded", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(row["id"])

    superseded = executions.finish_execution(
        row["id"], success=False, superseded=True, error="overtaken by a later fire")

    assert superseded["status"] == "superseded"
    # Terminal attempts stay immutable: a second terminal write is refused.
    assert executions.finish_execution(row["id"], success=True) is None
    assert executions.get_execution(row["id"])["status"] == "superseded"
    # And a superseded occurrence is still eligible: its work was not confirmed done.
    assert executions.create_execution(
        "job-superseded", source="builtin", scheduled_instant=INSTANT)["status"] == "claimed"


def test_binding_an_already_completed_occurrence_is_refused(executions):
    """The provider path learns its instant after the row exists; the refusal must land there too."""
    from cron.executions import DuplicateFireAttempt

    winner = executions.create_execution("job-bind", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(winner["id"])
    executions.finish_execution(winner["id"], success=True)
    row = executions.create_execution("job-bind", source="provider")

    with pytest.raises(DuplicateFireAttempt):
        executions.set_execution_occurrence(row["id"], INSTANT)

    # The refused attempt is closed as superseded, never as another failure for the occurrence.
    assert executions.get_execution(row["id"])["status"] == "superseded"


def test_an_existing_ledger_migrates_to_the_superseded_vocabulary(monkeypatch, tmp_path):
    """A live profile's ledger predates ``superseded``; opening it must adopt the new vocabulary
    without losing rows or rewriting terminal outcomes."""
    db_path = tmp_path / "cron" / "executions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE executions (
          id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
          process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
          status TEXT NOT NULL CHECK(status IN
            ('claimed','running','completed','failed','unknown')),
          handoff_pending INTEGER NOT NULL DEFAULT 0, handoff_started_at REAL,
          claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT,
          delivery_outcome TEXT, scheduled_instant TEXT
        );
        INSERT INTO executions
          (id, job_id, source, process_id, pid, status, handoff_pending, claimed_at)
        VALUES ('legacy-row','legacy-job','builtin','other-process',1,'completed',0,
                '2026-09-12T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    import cron.executions as executions_mod

    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", db_path)

    assert executions_mod.get_execution("legacy-row")["status"] == "completed"
    fresh = executions_mod.create_execution(
        "legacy-job", source="builtin", scheduled_instant=INSTANT)
    executions_mod.mark_execution_running(fresh["id"])
    assert executions_mod.finish_execution(
        fresh["id"], success=False, superseded=True, error="overtaken"
    )["status"] == "superseded"
