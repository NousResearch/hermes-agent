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


def test_execution_persists_schedule_output_and_delivery_provenance(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    schedule = {"minutes": 30, "kind": "interval", "z": 1, "a": 2}

    claimed = executions.create_execution(
        "observed-job",
        source="builtin",
        scheduled_for="2026-08-19T10:30:00+00:00",
        schedule=schedule,
    )
    secret = "SyntheticDeliveryPassword123456789"
    completed = executions.finish_execution(
        claimed["id"],
        success=True,
        output_path="/tmp/observed-job.md",
        delivery_outcome="failed",
        delivery_error=f'RuntimeError({{"password": "{secret}"}})',
    )

    assert claimed["scheduled_for"] == "2026-08-19T10:30:00+00:00"
    assert claimed["schedule_json"] == '{"a":2,"kind":"interval","minutes":30,"z":1}'
    assert completed["output_path"] == "/tmp/observed-job.md"
    assert completed["delivery_outcome"] == "failed"
    assert "RuntimeError" in completed["delivery_error"]
    assert secret not in completed["delivery_error"]


def test_delivery_error_redacts_url_query_credentials(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("query-secret", source="builtin")

    completed = executions.finish_execution(
        record["id"],
        success=True,
        delivery_outcome="failed",
        delivery_error="GET https://example.test/send?token=super-secret-token&mode=fast failed",
    )

    assert "super-secret-token" not in completed["delivery_error"]
    assert "example.test" in completed["delivery_error"]


def test_delivery_error_redacts_url_userinfo(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("userinfo-secret", source="builtin")

    completed = executions.finish_execution(
        record["id"],
        success=True,
        delivery_outcome="failed",
        delivery_error="POST https://alice:hunter2@example.test/hook failed",
    )

    assert "alice:hunter2@" not in completed["delivery_error"]
    assert "hunter2" not in completed["delivery_error"]
    assert "example.test" in completed["delivery_error"]


def test_legacy_execution_schema_is_migrated_in_place(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
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
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, status, claimed_at)
               VALUES ('legacy', 'old-job', 'builtin', 'old-process', 1,
                       'completed', '2026-08-18T00:00:00+00:00')"""
        )

    row = executions.create_execution(
        "new-job",
        source="builtin",
        scheduled_for="2026-08-19T10:30:00+00:00",
        schedule={"kind": "cron", "expr": "30 10 * * *"},
    )

    assert row["scheduled_for"] == "2026-08-19T10:30:00+00:00"
    assert row["schedule_json"] == '{"expr":"30 10 * * *","kind":"cron"}'
    legacy = executions.latest_execution("old-job")
    assert legacy["status"] == "completed"
    assert legacy["delivery_outcome"] is None


def test_legacy_execution_schema_migration_is_cross_process_safe(tmp_path):
    home = tmp_path / "home"
    db_path = home / "cron" / "executions.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE executions (
                 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
                 process_id TEXT NOT NULL, pid INTEGER NOT NULL,
                 process_started_at INTEGER, status TEXT NOT NULL,
                 claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                 error TEXT
               )"""
        )

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)
    gate = tmp_path / "start-migration"
    script = (
        "import os, time; from pathlib import Path; "
        "gate=Path(os.environ['MIGRATION_GATE']); "
        "deadline=time.monotonic()+10; "
        "\nwhile not gate.exists() and time.monotonic() < deadline: time.sleep(0.001)"
        "\nfrom cron.executions import create_execution; "
        "create_execution(os.environ['WORKER_ID'], source='test')"
    )
    workers = []
    for index in range(8):
        worker_env = env.copy()
        worker_env["MIGRATION_GATE"] = str(gate)
        worker_env["WORKER_ID"] = f"worker-{index}"
        workers.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=repo,
                env=worker_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )

    gate.touch()
    results = [worker.communicate(timeout=20) for worker in workers]

    assert [(worker.returncode, stderr) for worker, (_stdout, stderr) in zip(workers, results)] == [
        (0, "")
    ] * len(workers)
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(executions)")}
        count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    assert {
        "scheduled_for",
        "schedule_json",
        "output_path",
        "delivery_outcome",
        "delivery_error",
    } <= columns
    assert count == len(workers)


def test_current_schema_read_does_not_acquire_immediate_writer(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.create_execution("current", source="test")
    statements = []
    real_connect = executions._connect

    def traced_connect():
        conn = real_connect()
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(executions, "_connect", traced_connect)
    assert executions.latest_execution("current")["status"] == "claimed"
    assert not any("BEGIN IMMEDIATE" in sql.upper() for sql in statements)


def test_execution_ledger_follows_the_current_profile_home(monkeypatch, tmp_path):
    import cron.executions as executions

    current_home = {"path": tmp_path / "default"}
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(executions, "get_hermes_home", lambda: current_home["path"])

    default_row = executions.create_execution("default-job", source="builtin")
    current_home["path"] = tmp_path / "worker"
    worker_row = executions.create_execution("worker-job", source="builtin")

    assert executions.list_executions() == [worker_row]
    current_home["path"] = tmp_path / "default"
    assert executions.list_executions() == [default_row]
    assert (tmp_path / "default" / "cron" / "executions.db").is_file()
    assert (tmp_path / "worker" / "cron" / "executions.db").is_file()


def test_terminal_execution_cannot_be_rewritten(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)

    assert executions.finish_execution(
        record["id"], success=False, error="late writer"
    ) is None
    assert executions.latest_execution("immutable")["status"] == "completed"


def test_later_terminal_success_cannot_erase_delivery_failure(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable-delivery", source="builtin")
    failed = executions.finish_execution(
        record["id"],
        success=False,
        error="agent failed",
        output_path="/tmp/first.md",
        delivery_outcome="failed",
        delivery_error="first send failed",
    )

    assert executions.finish_execution(
        record["id"],
        success=True,
        output_path="/tmp/later.md",
        delivery_outcome="delivered",
    ) is None
    persisted = executions.latest_execution("immutable-delivery")
    assert persisted == failed
    assert persisted["delivery_outcome"] == "failed"
    assert persisted["output_path"] == "/tmp/first.md"


def test_claimed_execution_can_receive_external_claim_snapshot(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("external-slot", source="chronos")

    updated = executions.record_execution_schedule(
        record["id"],
        scheduled_for="2026-08-19T10:00:00+00:00",
        schedule={"kind": "cron", "expr": "0 * * * *"},
    )

    assert updated["scheduled_for"] == "2026-08-19T10:00:00+00:00"
    assert updated["schedule_json"] == '{"expr":"0 * * * *","kind":"cron"}'


def test_external_claim_flows_through_sqlite_ledger_to_terminal_state(monkeypatch, tmp_path):
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    profile_home = tmp_path / "profile"
    with jobs.use_cron_store(profile_home):
        created = jobs.create_job(
            prompt="record provenance",
            schedule="every 5m",
            name="integrated external fire",
        )
        original_slot = created["next_run_at"]

        claimed = InProcessCronScheduler().claim_fire(created["id"])

        assert claimed is not None
        execution_id = claimed["execution_id"]
        owner = claimed["fire_claim"]["by"]
        ledger_claim = executions.latest_execution(created["id"])
        assert ledger_claim["id"] == execution_id
        assert ledger_claim["status"] == "claimed"
        assert ledger_claim["scheduled_for"] == original_slot
        assert json.loads(ledger_claim["schedule_json"])["kind"] == "interval"
        stored_claim = jobs.get_job(created["id"])
        assert stored_claim["next_run_at"] != original_slot
        assert stored_claim["fire_claim"]["by"] == owner

        executions.mark_execution_running(execution_id)
        terminal = executions.finish_execution(
            execution_id,
            success=True,
            output_path="/tmp/integrated-output.md",
            delivery_outcome="delivered",
        )
        assert jobs.mark_job_run(
            created["id"], True, expected_fire_owner=owner
        ) is True

        assert terminal["status"] == "completed"
        assert terminal["output_path"] == "/tmp/integrated-output.md"
        assert terminal["delivery_outcome"] == "delivered"
        assert jobs.get_job(created["id"])["fire_claim"] is None


def test_fire_due_runs_shared_body_and_terminalizes_real_ledger(monkeypatch, tmp_path):
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (True, "full output", "final response", None),
    )
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda job_id, _output: tmp_path / f"{job_id}.md",
    )

    with jobs.use_cron_store(tmp_path / "profile"):
        created = jobs.create_job(
            prompt="run hermetic integration",
            schedule="every 5m",
            name="real terminal path",
            deliver="local",
        )
        original_slot = created["next_run_at"]

        assert InProcessCronScheduler().fire_due(created["id"]) is True

        terminal = executions.latest_execution(created["id"])
        assert terminal["status"] == "completed"
        assert terminal["scheduled_for"] == original_slot
        assert terminal["output_path"] == str(tmp_path / f"{created['id']}.md")
        assert terminal["delivery_outcome"] == "suppressed"
        stored = jobs.get_job(created["id"])
        assert stored["fire_claim"] is None
        assert stored["last_status"] == "ok"


def test_external_snapshot_miss_rearms_real_claim_and_fails_ledger(monkeypatch, tmp_path):
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    profile_home = tmp_path / "profile"
    dispatched = []
    monkeypatch.setattr(executions, "record_execution_schedule", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda job, **kwargs: dispatched.append(job) or True,
    )

    with jobs.use_cron_store(profile_home):
        created = jobs.create_job(
            prompt="retry only after provenance",
            schedule="every 5m",
            name="provenance fail closed",
        )
        original_slot = created["next_run_at"]

        assert InProcessCronScheduler().fire_due(created["id"]) is False

        stored = jobs.get_job(created["id"])
        assert stored["next_run_at"] == original_slot
        assert stored["fire_claim"] is None
        assert dispatched == []
        terminal = executions.latest_execution(created["id"])
        assert terminal["status"] == "failed"
        assert "not persisted" in terminal["error"]


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


def test_corrupt_store_fails_closed_without_overwrite(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")

    with __import__("pytest").raises(sqlite3.DatabaseError):
        executions.create_execution("new", source="builtin")
    assert executions.EXECUTIONS_FILE.read_bytes() == b"not a sqlite database"


def test_cron_runs_cli_prints_execution_history(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    row = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(row["id"], success=False, error="boom")
    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)

    output = capsys.readouterr().out
    assert row["id"] in output
    assert "failed" in output
    assert "boom" in output


def test_quick_backup_includes_execution_ledger():
    from hermes_cli.backup import _QUICK_STATE_FILES

    assert "cron/executions.db" in _QUICK_STATE_FILES


def test_failed_execution_keeps_error(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    record = executions.create_execution("job-2", source="external")
    failed = executions.finish_execution(record["id"], success=False, error="provider exploded")

    assert failed["status"] == "failed"
    assert failed["error"] == "provider exploded"


def test_recovery_does_not_mark_live_process_execution_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("still-live", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("still-live")["status"] == "running"


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
    assert "restart" in records[0]["error"].lower()
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
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())

    assert scheduler.tick(verbose=False, sync=False) == 0
    assert finished == [
        ("exec-submit-fail", {
            "success": False,
            "error": "Executor dispatch failed: executor rejected",
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
        lambda job, *, defer_agent_teardown=None, **_kw: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job({"id": "job-3", "execution_id": "exec-3"}) is True
    assert events[0] == ("running", "exec-3")
    assert events[-1][0:2] == ("finish", "exec-3")
    assert events[-1][2]["success"] is True


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


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__
    and its class methods can't be monkeypatched, so open/close tracking is
    done via a delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _count_open_connections(executions, monkeypatch):
    """Wrap sqlite3.connect to track open/close balance for the ledger module."""
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _TrackingConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)
    return opened_ids, closed_ids


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Regression for #69567: every ledger call must close its connection
    deterministically instead of relying on garbage collection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    record = executions.create_execution("leak-check", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)
    executions.list_executions(job_id="leak-check")
    executions.latest_executions(["leak-check"])
    executions.recover_interrupted_executions()

    assert len(opened) == 6
    assert len(closed) == 6
    assert set(opened) == set(closed)


def test_early_return_still_closes_connection(monkeypatch, tmp_path):
    """mark_execution_running returns None mid-block on a bad transition;
    the connection must still be closed rather than leaked."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    assert executions.mark_execution_running("does-not-exist") is None

    assert len(opened) == 1
    assert len(closed) == 1


def test_exception_during_operation_still_closes_connection(monkeypatch, tmp_path):
    """A failing statement inside the transaction must roll back and close,
    not leak the connection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    with __import__("pytest").raises(sqlite3.IntegrityError):
        with executions._transaction() as conn:
            conn.execute(
                "INSERT INTO executions (id, job_id, source, process_id, pid, "
                "status, claimed_at) VALUES ('x', 'x', 'x', 'x', 1, 'bogus-status', 'now')"
            )

    assert len(opened) == 1
    assert len(closed) == 1


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """If PRAGMA/DDL setup in _connect() fails after sqlite3.connect()
    succeeds, the partially-initialized connection must still be closed."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _FailingSchemaConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)

    with __import__("pytest").raises(sqlite3.OperationalError):
        executions.create_execution("init-fail", source="builtin")

    assert len(opened_ids) == 1
    assert len(closed_ids) == 1


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
