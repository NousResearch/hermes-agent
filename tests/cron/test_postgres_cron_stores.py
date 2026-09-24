"""Cron stores preserve their contracts on PostgreSQL."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing

import pytest


def test_execution_incident_and_notepad_lifecycle(postgres_db, postgres_home, monkeypatch):
    from cron import executions, incidents, notepad, occurrences

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(notepad, "NOTEPAD_FILE", None)

    instant = "2026-09-19T08:00:00+00:00"
    claimed = executions.create_execution(
        "daily-report", source="builtin", scheduled_instant=instant
    )
    assert claimed["status"] == "claimed"
    assert claimed["owner_lease_expires_at"] > time.time()
    assert executions.mark_execution_running(claimed["id"])["status"] == "running"
    assert executions.heartbeat_execution(claimed["id"]) is True
    completed = executions.finish_execution(claimed["id"], success=True)
    assert completed["status"] == "completed"
    assert completed["owner_lease_expires_at"] is None
    assert occurrences.completed_occurrence({"id": "daily-report"}, instant) is True

    incident_id, is_new = incidents.upsert_incident(
        "daily-report", "provider timeout after 60s"
    )
    assert is_new is True
    assert incidents.upsert_incident(
        "daily-report", "provider timeout after 60s"
    ) == (incident_id, False)
    assert incidents.ack_incident(incident_id) is True
    assert incidents.get_incident(incident_id)["state"] == "closed"

    saved = notepad.set_note("daily-report", "cursor", "résumé-42")
    assert saved["value"] == "résumé-42"
    assert notepad.get_note("daily-report", "cursor") == "résumé-42"
    assert notepad.list_notes("daily-report") == [saved]
    assert notepad.clear_notepad("daily-report") == 1

    assert not (postgres_home / "cron" / "executions.db").exists()
    assert not (postgres_home / "cron" / "notepad.db").exists()


def test_expired_execution_lease_recovers_without_pid_assumptions(
    postgres_db, monkeypatch
):
    from cron import executions

    record = executions.create_execution("lease-job", source="builtin")
    assert executions.mark_execution_running(record["id"]) is not None
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-container")

    assert executions.recover_interrupted_executions() == 0
    postgres_db._execute_write(
        lambda connection: connection.execute(
            "UPDATE executions SET owner_lease_expires_at=? WHERE id=?",
            (time.time() - 1, record["id"]),
        )
    )
    assert executions.recover_interrupted_executions() == 1
    assert executions.get_execution(record["id"])["status"] == "unknown"


def test_scheduler_renews_execution_lease_during_long_run(postgres_db, monkeypatch):
    from cron import executions, scheduler

    record = executions.create_execution("heartbeat-job", source="direct")
    heartbeats = 0
    renewed = threading.Event()
    real_heartbeat = scheduler.heartbeat_execution

    def observe(execution_id):
        nonlocal heartbeats
        result = real_heartbeat(execution_id)
        heartbeats += 1
        if heartbeats >= 2:
            renewed.set()
        return result

    monkeypatch.setattr(scheduler, "heartbeat_execution", observe)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    assert scheduler._run_with_fire_claim_heartbeat(
        {"id": "heartbeat-job", "execution_id": record["id"]},
        lambda _lost: renewed.wait(2),
    ) is True
    assert heartbeats >= 2
    assert executions.get_execution(record["id"])["owner_lease_expires_at"] > time.time()


def test_queued_execution_renews_lease_before_worker_starts(postgres_db, monkeypatch):
    from cron import executions, scheduler

    record = executions.create_execution("queued-job", source="builtin")
    renewed = threading.Event()
    real_heartbeat = scheduler.heartbeat_execution

    def observe(execution_id):
        result = real_heartbeat(execution_id)
        renewed.set()
        return result

    monkeypatch.setattr(scheduler, "heartbeat_execution", observe)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    future = concurrent.futures.Future()

    assert scheduler._start_queued_execution_heartbeat(record["id"], future) is True
    assert renewed.wait(2)
    assert executions.get_execution(record["id"])["owner_lease_expires_at"] > time.time()
    assert future.cancel() is True


def test_delivery_is_claimed_once_and_tombstone_blocks_replay(
    postgres_db, postgres_home, monkeypatch
):
    from cron import delivery_queue

    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", None)
    monkeypatch.setattr(delivery_queue, "MAX_TERMINAL_DELIVERIES", 0)
    sent = []
    job = {"id": "delivery-job", "deliver": "origin"}
    delivery_queue.enqueue("delivery-execution", job, "finished")

    assert delivery_queue.drain(
        lambda queued_job, content, for_failure: sent.append(
            (queued_job, content, for_failure)
        )
    ) == 1
    assert sent == [(job, "finished", False)]
    assert delivery_queue.get_status("delivery-execution")["status"] == "delivered"

    delivery_queue.enqueue("delivery-execution", job, "duplicate")
    assert delivery_queue.drain(lambda *_args: sent.append("duplicate")) == 0
    assert len(sent) == 1
    assert not (postgres_home / "cron" / "deliveries.db").exists()


def test_two_processes_cannot_claim_one_delivery(postgres_db, postgres_home):
    from cron import delivery_queue

    delivery_queue.enqueue("competing-delivery", {"id": "job"}, "result")
    script = (
        "import json; "
        "from cron.delivery_queue import claim_next; "
        "print(json.dumps(claim_next()))"
    )
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(postgres_home)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(2)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout))

    assert sorted(result is None for result in results) == [False, True]
    row = delivery_queue.get_status("competing-delivery")
    assert row["status"] == "delivering"
    assert row["owner_lease_expires_at"] > time.time()


def test_expired_delivery_lease_becomes_unknown_without_resend(postgres_db, monkeypatch):
    from cron import delivery_queue

    delivery_queue.enqueue("expired-delivery", {"id": "job"}, "result")
    assert delivery_queue.claim_next() is not None
    monkeypatch.setattr(delivery_queue, "_PROCESS_ID", "replacement-container")

    assert delivery_queue.recover_abandoned() == 0
    postgres_db._execute_write(
        lambda connection: connection.execute(
            "UPDATE deliveries SET owner_lease_expires_at=? WHERE execution_id=?",
            (time.time() - 1, "expired-delivery"),
        )
    )
    assert delivery_queue.recover_abandoned() == 1
    assert delivery_queue.get_status("expired-delivery")["status"] == "unknown"
    assert delivery_queue.drain(lambda *_args: pytest.fail("delivery was retried")) == 0


def test_same_gateway_fences_unrecorded_delivery_result_immediately(postgres_db):
    from cron import delivery_queue

    delivery_queue.enqueue("uncertain-delivery", {"id": "job"}, "result")
    assert delivery_queue.claim_next() is not None
    with delivery_queue._lock:
        delivery_queue._ACTIVE_DELIVERIES.discard("uncertain-delivery")

    assert delivery_queue.recover_abandoned() == 1
    status = delivery_queue.get_status("uncertain-delivery")
    assert status["status"] == "unknown"
    assert "not retried" in status["error"]


def test_cron_sqlite_migration_is_atomic_and_preserves_sources(
    postgres_db, postgres_home, tmp_path
):
    from cron import delivery_queue, executions, incidents, notepad
    from scripts.migrate_cron_sqlite_to_postgres import (
        migrate_cron_sqlite_to_postgres,
    )

    source_dir = tmp_path / "source-cron"
    source_dir.mkdir()
    executions_path = source_dir / "executions.db"
    with closing(sqlite3.connect(executions_path)) as connection:
        executions._initialize_schema(connection)
        incidents._initialize_schema(connection)
        connection.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, status, claimed_at,
                finished_at, scheduled_instant)
               VALUES ('exec-1', 'job-1', 'builtin', 'old-owner', 10,
                       'completed', '2026-09-19T08:00:00+00:00',
                       '2026-09-19T08:01:00+00:00',
                       '2026-09-19T08:00:00+00:00')"""
        )
        connection.execute(
            """INSERT INTO cron_incidents
               (id, job_id, error_sig, state, first_seen_at, last_seen_at, error)
               VALUES ('incident-1', 'job-1', 'sig', 'closed', 'first', 'last', 'error')"""
        )
        connection.commit()

    deliveries_path = source_dir / "deliveries.db"
    with closing(sqlite3.connect(deliveries_path)) as connection:
        delivery_queue._initialize_schema(connection)
        connection.execute(
            """INSERT INTO deliveries
               (execution_id, job_json, content, status, created_at)
               VALUES ('delivery-1', '{"id":"job-1"}', 'result', 'pending', 'created')"""
        )
        connection.execute(
            """INSERT INTO delivery_tombstones
               (execution_id, terminal_status, finished_at)
               VALUES ('delivery-old', 'delivered', 'finished')"""
        )
        connection.commit()

    notepad_path = source_dir / "notepad.db"
    with closing(sqlite3.connect(notepad_path)) as connection:
        notepad._initialize_schema(connection)
        connection.execute(
            """INSERT INTO cron_notepad (job_id, key, value, updated_at)
               VALUES ('job-1', 'cursor', '42', 'updated')"""
        )
        connection.commit()

    source_bytes = {
        path.name: path.read_bytes()
        for path in (executions_path, deliveries_path, notepad_path)
    }
    counts = migrate_cron_sqlite_to_postgres(source_dir, postgres_db, batch_size=1)

    assert counts == {
        "executions": 1,
        "cron_incidents": 1,
        "deliveries": 1,
        "delivery_tombstones": 1,
        "cron_notepad": 1,
    }
    assert executions.get_execution("exec-1")["status"] == "completed"
    assert incidents.get_incident("incident-1")["state"] == "closed"
    assert delivery_queue.get_status("delivery-1")["content"] == "result"
    assert delivery_queue.get_status("delivery-old")["status"] == "delivered"
    assert notepad.get_note("job-1", "cursor") == "42"
    assert source_bytes == {
        path.name: path.read_bytes()
        for path in (executions_path, deliveries_path, notepad_path)
    }
    assert not (postgres_home / "cron" / "executions.db").exists()


def test_cron_migration_failure_rolls_back_earlier_tables(
    postgres_db, tmp_path
):
    import psycopg

    from cron import executions, notepad
    from scripts.migrate_cron_sqlite_to_postgres import (
        migrate_cron_sqlite_to_postgres,
    )

    source_dir = tmp_path / "rollback-source"
    source_dir.mkdir()
    with closing(sqlite3.connect(source_dir / "executions.db")) as connection:
        executions._initialize_schema(connection)
        connection.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, status, claimed_at)
               VALUES ('exec-rollback', 'job', 'builtin', 'owner', 1,
                       'completed', 'claimed')"""
        )
        connection.commit()
    with closing(sqlite3.connect(source_dir / "notepad.db")) as connection:
        notepad._initialize_schema(connection)
        connection.execute(
            """INSERT INTO cron_notepad (job_id, key, value, updated_at)
               VALUES ('job', 'reject', 'value', 'updated')"""
        )
        connection.commit()

    postgres_db._execute_write(
        lambda connection: connection.execute(
            """ALTER TABLE cron_notepad ADD CONSTRAINT reject_migration_note
               CHECK (key <> 'reject')"""
        )
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        migrate_cron_sqlite_to_postgres(source_dir, postgres_db, batch_size=1)

    with postgres_db._read_ctx() as connection:
        assert connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM cron_notepad").fetchone()[0] == 0
