"""Durable cron execution-ledger behavior."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_late_hermes_home_repoint_scopes_execution_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "late-home"))

    assert executions._current_executions_file() == (
        tmp_path / "late-home" / "cron" / "executions.db"
    ).resolve()


def test_execution_source_is_categorical(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    private_source = "PRIVATE_PROVIDER_SOURCE_5c2a"

    record = executions.create_execution("source-job", source=private_source)

    assert record["source"] == "unknown"
    assert private_source.encode() not in executions.EXECUTIONS_FILE.read_bytes()

    external = executions.create_execution("external-job", source="chronos")
    assert external["source"] == "external"


def test_execution_transitions_are_durable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    claimed = executions.create_execution("job-1", source="builtin")
    assert claimed["status"] == "claimed"
    assert claimed["claimed_at"]
    assert claimed["started_at"] is None
    assert claimed["finished_at"] is None

    running = executions.mark_execution_running(claimed["id"])
    assert running["status"] == "running"
    assert running["started_at"]

    completed = executions.finish_execution(claimed["id"], success=True)
    assert completed["status"] == "completed"
    assert completed["finished_at"]
    assert completed["error"] is None

    persisted = executions.list_executions(job_id="job-1")
    assert persisted == [completed]


def test_terminal_execution_cannot_be_rewritten(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)

    assert executions.finish_execution(
        record["id"], success=False, error="late writer"
    ) is None
    assert executions.latest_execution("immutable")["status"] == "completed"


def test_retention_bounds_terminal_history_but_preserves_inflight(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 3)
    inflight = executions.create_execution("live", source="builtin")
    executions.mark_execution_running(inflight["id"])
    for index in range(8):
        row = executions.create_execution(f"done-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)

    records = executions.list_executions(limit=100)
    assert len([row for row in records if row["status"] == "completed"]) == 3
    assert executions.latest_execution("live")["status"] == "running"


def test_execution_storage_is_owner_only(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("private", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.EXECUTIONS_FILE.parent.stat().st_mode & 0o777 == 0o700
    for path in executions.EXECUTIONS_FILE.parent.glob("executions.db*"):
        assert path.stat().st_mode & 0o777 == 0o600


def test_corrupt_store_fails_closed_without_overwrite(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")

    with __import__("pytest").raises(sqlite3.DatabaseError):
        executions.create_execution("new", source="builtin")
    assert executions.EXECUTIONS_FILE.read_bytes() == b"not a sqlite database"


def test_execution_history_is_paginated(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    ids = []
    for _index in range(5):
        row = executions.create_execution("paged", source="builtin")
        executions.finish_execution(row["id"], success=True)
        ids.append(row["id"])

    first = executions.list_executions(job_id="paged", limit=2)
    second = executions.list_executions(
        job_id="paged", limit=2, before_claimed_at=first[-1]["claimed_at"]
    )
    assert [row["id"] for row in first] == list(reversed(ids))[:2]
    assert set(row["id"] for row in first).isdisjoint(row["id"] for row in second)


def test_cron_runs_cli_prints_categorical_execution_history(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    private_error = "PRIVATE_CLI_ERROR_TEXT_2b8d"
    row = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(row["id"], success=False, error=private_error)
    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)

    output = capsys.readouterr().out
    assert row["id"] in output
    assert "failed" in output
    assert "execution_failed" in output
    assert "result=execution_failed" in output
    assert "exit=?" in output
    assert "delivery=not_requested" in output
    assert private_error not in output


def test_exit_code_rejects_private_boolean_and_unbounded_values(
    monkeypatch, tmp_path, capsys
):
    executions = _point_ledger(monkeypatch, tmp_path)
    private_exit = "PRIVATE_EXIT_CODE_TEXT_3a91"
    invalid_values = (private_exit, True, 2**63)

    for index, invalid in enumerate(invalid_values):
        record = executions.create_execution(f"invalid-exit-{index}", source="builtin")
        executions.finish_execution(
            record["id"],
            success=False,
            result_kind="execution_failed",
            exit_code=invalid,  # type: ignore[arg-type] - exercise runtime boundary
        )
        latest = executions.latest_execution(f"invalid-exit-{index}")
        assert latest is not None
        assert latest["exit_code"] is None

    from hermes_cli.cron import cron_runs

    cron_runs(limit=10)
    assert private_exit not in capsys.readouterr().out
    persisted = b"".join(
        path.read_bytes()
        for path in executions.EXECUTIONS_FILE.parent.glob("executions.db*")
    )
    assert private_exit.encode() not in persisted


def test_quick_backup_includes_execution_ledger():
    from hermes_cli.backup import _QUICK_STATE_FILES

    assert "cron/executions.db" in _QUICK_STATE_FILES


def test_failed_execution_keeps_only_categorical_error(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    private_error = "PRIVATE_PROVIDER_FAILURE_TEXT_7f3e"

    record = executions.create_execution("job-2", source="external")
    failed = executions.finish_execution(
        record["id"],
        success=False,
        error=private_error,
        error_code="execution_failed",
    )

    assert failed["status"] == "failed"
    assert failed["error"] == "execution_failed"
    assert private_error.encode() not in executions.EXECUTIONS_FILE.read_bytes()


def test_delivery_outcome_is_categorical_metadata(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("job-delivery", source="builtin")

    finished = executions.finish_execution(
        record["id"],
        success=True,
        result_kind="silent",
        exit_code=0,
        delivery_requested=True,
        delivery_attempted=False,
        delivery_failed=False,
    )

    assert finished["status"] == "completed"
    assert finished["result_kind"] == "silent"
    assert finished["exit_code"] == 0
    assert finished["delivery_requested"] == 1
    assert finished["delivery_attempted"] == 0
    assert finished["delivery_failed"] == 0


def test_upgrade_scrubs_historical_raw_errors(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    private_error = "PRIVATE_LEGACY_ERROR_TEXT_9c41"
    private_source = "PRIVATE_LEGACY_SOURCE_TEXT_a13d"
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            """CREATE TABLE executions (
                 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
                 process_id TEXT NOT NULL, pid INTEGER NOT NULL,
                 process_started_at INTEGER, status TEXT NOT NULL,
                 claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                 error TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO executions VALUES
               ('legacy-1','job-legacy',?,'old-process',1,NULL,'failed',
                '2026-01-01T00:00:00+00:00',NULL,'2026-01-01T00:00:01+00:00',?)""",
            (private_source, private_error),
        )

    rows = executions.list_executions(job_id="job-legacy")

    assert rows[0]["error"] == "legacy_error_redacted"
    assert rows[0]["source"] == "unknown"
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        assert conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == "5"
    persisted = b"".join(
        path.read_bytes()
        for path in executions.EXECUTIONS_FILE.parent.glob("executions.db*")
        if path.is_file()
    )
    assert private_error.encode() not in persisted
    assert private_source.encode() not in persisted


def test_busy_privacy_checkpoint_stays_retryable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("busy-migration", source="builtin")
    executions.finish_execution(record["id"], success=False)
    private_error = "PRIVATE_BUSY_ERROR_4b17"
    private_source = "PRIVATE_BUSY_SOURCE_5c28"
    private_exit = "PRIVATE_BUSY_EXIT_6d39"

    with sqlite3.connect(executions.EXECUTIONS_FILE) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "UPDATE executions SET error=?,source=?,exit_code=? WHERE id=?",
            (private_error, private_source, private_exit, record["id"]),
        )
        writer.execute(
            "UPDATE schema_meta SET value='1' WHERE key='schema_version'"
        )
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    reader = sqlite3.connect(executions.EXECUTIONS_FILE)
    reader.execute("BEGIN")
    assert reader.execute(
        "SELECT error FROM executions WHERE id=?", (record["id"],)
    ).fetchone()[0] == private_error
    try:
        try:
            executions.list_executions(job_id="busy-migration")
        except sqlite3.OperationalError as exc:
            assert "checkpoint" in str(exc).lower()
        else:
            raise AssertionError("busy privacy checkpoint was accepted")
        with sqlite3.connect(executions.EXECUTIONS_FILE) as observer:
            assert observer.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0] != "5"
    finally:
        reader.close()

    migrated = executions.list_executions(job_id="busy-migration")
    assert migrated[0]["error"] == "legacy_error_redacted"
    assert migrated[0]["source"] == "unknown"
    assert migrated[0]["exit_code"] is None
    persisted = b"".join(
        path.read_bytes()
        for path in executions.EXECUTIONS_FILE.parent.glob("executions.db*")
    )
    assert private_error.encode() not in persisted
    assert private_source.encode() not in persisted
    assert private_exit.encode() not in persisted


def test_schema_v4_genericizes_existing_chronos_source(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("legacy-external", source="builtin")
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute("UPDATE executions SET source='chronos' WHERE id=?", (record["id"],))
        conn.execute(
            "UPDATE schema_meta SET value='4' WHERE key='schema_version'"
        )

    migrated = executions.latest_execution("legacy-external")

    assert migrated is not None
    assert migrated["source"] == "external"
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        assert conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == "5"


def test_recovery_does_not_mark_live_process_execution_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("still-live", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("still-live")["status"] == "running"


def test_recovery_does_not_mark_other_live_owner_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("other-live", source="builtin")
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            "UPDATE executions SET process_id=?, pid=? WHERE id=?",
            ("another-import", os.getpid(), record["id"]),
        )

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("other-live")["status"] == "claimed"


def test_recovery_rejects_recycled_pid(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("recycled", source="builtin")
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute(
            "UPDATE executions SET process_id=?, process_started_at=? WHERE id=?",
            ("old-import", -1, record["id"]),
        )

    assert executions.recover_interrupted_executions() == 1
    assert executions.latest_execution("recycled")["status"] == "unknown"


def test_restart_marks_interrupted_execution_unknown_without_requeue(tmp_path):
    """Real temp-HERMES_HOME subprocess restart: in-flight is audit-only unknown."""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cron.executions import create_execution, mark_execution_running; "
            "r=create_execution('restart-job', source='builtin'); "
            "mark_execution_running(r['id']); print(r['id'])",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    execution_id = create.stdout.strip()

    recover = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from cron.executions import recover_interrupted_executions, list_executions; "
            "print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='restart-job'))) ",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1"
    records = json.loads(lines[1])
    assert len(records) == 1
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown"
    assert records[0]["finished_at"]
    assert records[0]["error"] == "scheduler_restarted"
    assert records[0]["result_kind"] == "scheduler_restarted"
    # Recovery only classifies the old attempt. It must not manufacture a new
    # claimed record (which would imply an automatic retry).
    assert [r["status"] for r in records] == ["unknown"]


def test_generic_submit_failure_finishes_attempt_and_releases_guard(monkeypatch):
    import cron.scheduler as scheduler

    class BrokenPool:
        def submit(self, _callable):
            raise ValueError("executor rejected")

    finished = []
    monkeypatch.setattr(
        scheduler, "create_execution",
        lambda *_args, **_kwargs: {"id": "exec-submit-fail"},
    )
    monkeypatch.setattr(
        scheduler, "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [{"id": "submit-fail"}])
    monkeypatch.setattr(scheduler, "advance_next_run", lambda _job_id: None)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())

    assert scheduler.tick(verbose=False, sync=False) == 0
    assert finished == [
        ("exec-submit-fail", {
            "success": False,
            "error_code": "executor_dispatch_failed",
            "result_kind": "executor_dispatch_failed",
        })
    ]
    assert "submit-fail" not in scheduler.get_running_job_ids()


def test_run_one_job_records_running_then_terminal(monkeypatch):
    import cron.scheduler as scheduler

    events = []
    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda execution_id: events.append(("running", execution_id)),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: events.append(("finish", execution_id, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, *, defer_agent_teardown=None: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job({"id": "job-3", "execution_id": "exec-3"}) is True
    assert events[0] == ("running", "exec-3")
    assert events[-1][0:2] == ("finish", "exec-3")
    assert events[-1][2]["success"] is True
    assert events[-1][2]["result_kind"] == "output_produced"
    assert events[-1][2]["exit_code"] == 0
    assert events[-1][2]["delivery_requested"] is False
    assert events[-1][2]["delivery_attempted"] is False
    assert events[-1][2]["delivery_failed"] is False


def test_local_only_delivery_list_is_not_requested(monkeypatch):
    import cron.scheduler as scheduler

    finished = []
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, *, defer_agent_teardown=None: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job(
        {"id": "job-local-list", "execution_id": "exec-local-list", "deliver": ["local"]}
    ) is True

    result = finished[-1][1]
    assert result["delivery_requested"] is False
    assert result["delivery_attempted"] is False


def test_execution_failure_remains_primary_when_delivery_also_fails(monkeypatch):
    import cron.scheduler as scheduler

    finished = []
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, *, defer_agent_teardown=None: (
            False,
            "output",
            "failure response",
            "PRIVATE_EXECUTION_ERROR_18cf",
        ),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_args, **_kwargs: "PRIVATE_DELIVERY_ERROR_29df",
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job(
        {"id": "job-both-fail", "execution_id": "exec-both-fail", "deliver": "origin"}
    ) is True

    result = finished[-1][1]
    assert result["success"] is False
    assert result["result_kind"] == "execution_failed"
    assert result["error_code"] == "execution_failed"
    assert result["delivery_requested"] is True
    assert result["delivery_attempted"] is True
    assert result["delivery_failed"] is True
    assert "PRIVATE" not in repr(result)


def _fail_execution_creation(*_args, **_kwargs):
    raise sqlite3.OperationalError("synthetic ledger failure")


def test_ledger_creation_failure_restores_real_stale_cron_catchup(
    monkeypatch, tmp_path
):
    from datetime import datetime, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    home = tmp_path / "home"
    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    old_due = "2026-07-16T14:00:00+00:00"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "create_execution", _fail_execution_creation)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    job = jobs.create_job(
        prompt="stale catchup",
        schedule="0 * * * *",
        name="stale catchup",
    )
    stored = jobs.load_jobs()
    stored[0]["next_run_at"] = old_due
    jobs.save_jobs(stored)

    assert scheduler.tick(verbose=False, sync=False) == 0

    restored = jobs.get_job(job["id"])
    assert restored["next_run_at"] == old_due
    assert job["id"] not in scheduler.get_running_job_ids()


def test_ledger_creation_failure_releases_real_builtin_oneshot_claim(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    home = tmp_path / "home"
    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "create_execution", _fail_execution_creation)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    job = jobs.create_job(prompt="one shot", schedule=due_at, name="one shot")

    assert scheduler.tick(verbose=False, sync=False) == 0

    restored = jobs.get_job(job["id"])
    assert restored["next_run_at"] == due_at
    assert restored.get("run_claim") is None
    assert job["id"] not in scheduler.get_running_job_ids()


def test_external_ledger_creation_failure_restores_claim_and_schedule(
    monkeypatch, tmp_path
):
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import CronScheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    job = jobs.create_job(prompt="external", schedule="every 1h", name="external")
    before = jobs.get_job(job["id"])
    dispatched = []
    monkeypatch.setattr(executions, "create_execution", _fail_execution_creation)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: dispatched.append(True) or True,
    )

    class ExternalScheduler(CronScheduler):
        @property
        def name(self):
            return "external-test"

        def start(self, stop_event, **kwargs):
            return None

    try:
        ExternalScheduler().fire_due(job["id"])
    except sqlite3.OperationalError:
        pass
    else:
        raise AssertionError("ledger creation failure must propagate")

    restored = jobs.get_job(job["id"])
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored.get("fire_claim") == before.get("fire_claim")
    assert dispatched == []


def test_dispatch_rollback_refuses_to_clobber_concurrent_claim(monkeypatch, tmp_path):
    import cron.jobs as jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    job = jobs.create_job(prompt="race", schedule="every 1h", name="race")
    receipt = jobs.claim_job_for_fire_with_receipt(job["id"])
    assert receipt is not None

    concurrent_claim = {"at": "2026-07-18T18:00:00+00:00", "by": "other:42"}
    stored = jobs.load_jobs()
    stored[0]["fire_claim"] = concurrent_claim
    jobs.save_jobs(stored)

    assert jobs.restore_job_dispatch_state(receipt) is False
    assert jobs.get_job(job["id"])["fire_claim"] == concurrent_claim


def test_tick_does_not_adopt_or_rollback_concurrent_schedule_change(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    concurrent_next = "2031-09-09T09:09:09+00:00"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "create_execution", _fail_execution_creation)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    job = jobs.create_job(prompt="schedule race", schedule="every 1h", name="schedule race")
    stored = jobs.load_jobs()
    stored[0]["next_run_at"] = due_at
    jobs.save_jobs(stored)

    original_refresh = scheduler.refresh_dispatch_receipt

    def refresh_after_concurrent_schedule_write(job_id, receipt):
        current = jobs.load_jobs()
        current[0]["next_run_at"] = concurrent_next
        jobs.save_jobs(current)
        return original_refresh(job_id, receipt)

    monkeypatch.setattr(
        scheduler,
        "refresh_dispatch_receipt",
        refresh_after_concurrent_schedule_write,
    )

    try:
        scheduler.tick(verbose=False, sync=False)
    except RuntimeError:
        pass

    assert jobs.get_job(job["id"])["next_run_at"] == concurrent_next


def test_tick_unwinds_earlier_jobs_when_later_preparation_becomes_ambiguous(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    concurrent_next = "2034-04-04T04:04:04+00:00"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    first = jobs.create_job(prompt="first", schedule="every 1h", name="first")
    second = jobs.create_job(prompt="second", schedule="every 1h", name="second")
    stored = jobs.load_jobs()
    for item in stored:
        item["next_run_at"] = due_at
    jobs.save_jobs(stored)

    ledger_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_args, **_kwargs: ledger_calls.append(True),
    )
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: dispatch_calls.append(True),
    )
    original_refresh = scheduler.refresh_dispatch_receipt
    refresh_count = 0

    def race_on_second_refresh(job_id, receipt):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 2:
            current = jobs.load_jobs()
            for item in current:
                if item["id"] == job_id:
                    item["next_run_at"] = concurrent_next
            jobs.save_jobs(current)
        return original_refresh(job_id, receipt)

    monkeypatch.setattr(scheduler, "refresh_dispatch_receipt", race_on_second_refresh)

    error = None
    try:
        scheduler.tick(verbose=False, sync=False)
    except RuntimeError as exc:
        error = str(exc)

    assert error is not None
    assert second["id"] in error
    assert first["id"] not in error
    assert jobs.get_job(first["id"])["next_run_at"] == due_at
    assert jobs.get_job(second["id"])["next_run_at"] == concurrent_next
    assert ledger_calls == []
    assert dispatch_calls == []


def test_tick_unwinds_later_jobs_after_ambiguous_submission_failure(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    concurrent_next = "2035-05-05T05:05:05+00:00"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    first = jobs.create_job(prompt="first submit", schedule="every 1h", name="first submit")
    second = jobs.create_job(prompt="second submit", schedule="every 1h", name="second submit")
    stored = jobs.load_jobs()
    for item in stored:
        item["next_run_at"] = due_at
    jobs.save_jobs(stored)

    ledger_calls = []
    dispatch_calls = []

    def fail_first_ledger(job_id, **_kwargs):
        ledger_calls.append(job_id)
        current = jobs.load_jobs()
        for item in current:
            if item["id"] == job_id:
                item["next_run_at"] = concurrent_next
        jobs.save_jobs(current)
        raise sqlite3.OperationalError("ledger unavailable")

    monkeypatch.setattr(scheduler, "create_execution", fail_first_ledger)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: dispatch_calls.append(True),
    )

    error = None
    try:
        scheduler.tick(verbose=False, sync=False)
    except RuntimeError as exc:
        error = str(exc)

    assert error is not None
    assert first["id"] in error
    assert jobs.get_job(first["id"])["next_run_at"] == concurrent_next
    assert jobs.get_job(second["id"])["next_run_at"] == due_at
    assert ledger_calls == [first["id"]]
    assert dispatch_calls == []


def test_tick_restores_prepared_jobs_when_interpreter_is_shutting_down(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: True)

    first = jobs.create_job(prompt="shutdown one", schedule="every 1h", name="shutdown one")
    second = jobs.create_job(prompt="shutdown two", schedule="every 1h", name="shutdown two")
    stored = jobs.load_jobs()
    for item in stored:
        item["next_run_at"] = due_at
    jobs.save_jobs(stored)

    ledger_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_args, **_kwargs: ledger_calls.append(True),
    )
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: dispatch_calls.append(True),
    )

    assert scheduler.tick(verbose=False, sync=False) == 0

    assert jobs.get_job(first["id"])["next_run_at"] == due_at
    assert jobs.get_job(second["id"])["next_run_at"] == due_at
    assert ledger_calls == []
    assert dispatch_calls == []


def test_tick_restores_new_occurrence_when_previous_run_is_still_active(
    monkeypatch, tmp_path
):
    from datetime import datetime, timedelta, timezone

    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    now = datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc)
    due_at = (now - timedelta(seconds=10)).isoformat()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *_args: False)

    job = jobs.create_job(prompt="long running", schedule="every 1h", name="long running")
    running = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(running["id"])
    stored = jobs.load_jobs()
    stored[0]["next_run_at"] = due_at
    jobs.save_jobs(stored)

    ledger_calls = []
    dispatch_calls = []
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_args, **_kwargs: ledger_calls.append(True),
    )
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: dispatch_calls.append(True),
    )

    scheduler._running_job_ids.add(job["id"])
    try:
        assert scheduler.tick(verbose=False, sync=False) == 0
    finally:
        scheduler._running_job_ids.discard(job["id"])

    assert jobs.get_job(job["id"])["next_run_at"] == due_at
    assert len(executions.list_executions(job_id=job["id"])) == 1
    assert ledger_calls == []
    assert dispatch_calls == []


def test_provider_start_recovers_interrupted_records_before_tick(monkeypatch):
    import cron.scheduler_provider as provider

    events = []
    stop = __import__("threading").Event()
    stop.set()
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
        raising=False,
    )
    monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", lambda **_kwargs: events.append("heartbeat"))

    provider.InProcessCronScheduler().start(stop, interval=1)

    assert events[:2] == ["recover", "heartbeat"]


def test_external_provider_start_recovers_interrupted_records(monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    provider._client = type("Client", (), {"arm": lambda self, **kwargs: None})()
    events = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
    )
    monkeypatch.setattr(provider, "reconcile", lambda: events.append("reconcile"))

    provider.start(__import__("threading").Event())

    assert events == ["recover", "reconcile"]


def test_job_listing_exposes_latest_execution(monkeypatch, tmp_path):
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    executions = _point_ledger(monkeypatch, tmp_path)

    job = jobs.create_job(prompt="audit me", schedule="every 1h", name="audit")
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])

    listed = jobs.list_jobs(include_disabled=True)
    assert listed[0]["latest_execution"]["id"] == record["id"]
    assert listed[0]["latest_execution"]["status"] == "running"
