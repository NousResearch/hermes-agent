"""Durable cron execution-ledger behavior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_owner_liveness_fails_safe_when_start_time_is_unavailable(monkeypatch):
    """A live PID with unprovable birth time must never be recovered as dead."""
    import cron.executions as executions
    import gateway.status as status

    monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(executions, "_process_start_time", lambda _pid: None)

    assert executions._owner_is_live(424242, None) is True
    assert executions._owner_is_live(424242, 123456) is True


def test_schema_v8_migrates_every_legacy_version_without_pending_side_effects(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    for legacy_version in range(8):
        path = tmp_path / f"legacy-{legacy_version}.db"
        monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                """CREATE TABLE executions (
                     id TEXT PRIMARY KEY,
                     job_id TEXT NOT NULL,
                     source TEXT NOT NULL,
                     process_id TEXT NOT NULL,
                     pid INTEGER NOT NULL,
                     process_started_at INTEGER,
                     status TEXT NOT NULL,
                     claimed_at TEXT NOT NULL,
                     started_at TEXT,
                     finished_at TEXT,
                     error TEXT
                   )"""
            )
            conn.execute(
                """INSERT INTO executions
                   (id, job_id, source, process_id, pid, status, claimed_at,
                    finished_at)
                   VALUES ('legacy', 'job', 'builtin', 'old-process', 1,
                           'completed', '2026-07-30T00:00:00+00:00',
                           '2026-07-30T00:00:01+00:00')"""
            )
            conn.execute(f"PRAGMA user_version={legacy_version}")

        migrated = executions.get_execution("legacy")
        assert migrated is not None
        assert migrated["phase"] == "completed"
        assert migrated["delivery_state"] == "not_applicable"
        assert migrated["transcript_state"] == "not_applicable"
        assert migrated["requires_job_accounting"] == 0
        assert migrated["admitted_binding_version"] is None
        assert migrated["admitted_route_instance_id"] is None
        assert executions.list_pending_contextual_transcripts() == []
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 8


def test_contextual_admission_and_typed_outcomes_are_durable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("contextual", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.seal_contextual_admission(
        record["id"],
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_route_instance_id="route-instance-a",
        admitted_binding_version=2,
    ) is True
    assert executions.seal_contextual_admission(
        record["id"],
        session_key="telegram:dm:42:42",
        admitted_session_id="session-2",
    ) is False
    admitted = executions.get_execution(record["id"])
    assert admitted["admitted_at"]

    finished = executions.finish_contextual_execution(
        record["id"],
        outcome="no_action",
    )
    assert finished is not None
    assert finished["status"] == "completed"
    assert finished["outcome"] == "no_action"
    assert finished["session_key"] == "telegram:dm:42:42"
    assert finished["admitted_binding_version"] == 2
    assert finished["admitted_route_instance_id"] == "route-instance-a"
    assert finished["admitted_session_id"] == "session-1"
    assert json.loads(finished["result_json"])["kind"] == "no_action"
    assert executions.finish_contextual_execution(
        record["id"], outcome="notify", final_response="duplicate"
    ) is None


def test_contextual_result_atomically_stages_immutable_transcript_outbox(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("contextual-transcript", source="builtin")
    executions.mark_execution_running(record["id"])
    entries = [
        {
            "role": "user",
            "content": "hidden trigger",
            "display_kind": "hidden",
            "message_id": f"contextual-cron:{record['id']}:0",
        },
        {
            "role": "assistant",
            "content": "final answer",
            "message_id": f"contextual-cron:{record['id']}:1",
        },
    ]

    staged = executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="final answer",
        transcript_session_id="session-1",
        transcript_entries=entries,
        transcript_base_message_count=4,
        transcript_base_revision=9,
        transcript_last_prompt_tokens=77,
    )

    assert staged is not None
    assert staged["phase"] == "agent_completed"
    assert staged["transcript_state"] == "pending"
    assert staged["transcript_session_id"] == "session-1"
    assert staged["transcript_base_message_count"] == 4
    assert staged["transcript_last_prompt_tokens"] == 77
    assert json.loads(staged["transcript_json"]) == entries
    assert [item["id"] for item in executions.list_pending_contextual_transcripts()] == [
        record["id"]
    ]

    # The result + outbox join is idempotent for the same immutable payload,
    # but neither side may be rewritten independently after the transaction.
    assert executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="final answer",
        transcript_session_id="session-1",
        transcript_entries=entries,
        transcript_base_message_count=4,
        transcript_base_revision=9,
    ) is not None
    assert executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="final answer",
        transcript_session_id="session-1",
        transcript_entries=[{**entries[0], "content": "retargeted"}],
        transcript_base_message_count=4,
        transcript_base_revision=9,
    ) is None

    assert executions.mark_contextual_transcript_applied(record["id"]) is True
    assert executions.mark_contextual_transcript_applied(record["id"]) is False
    applied = executions.get_execution(record["id"])
    assert applied is not None
    assert applied["transcript_state"] == "applied"
    assert executions.list_pending_contextual_transcripts() == []


def test_contextual_transcript_outbox_rejects_invalid_base_counts(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    pytest = __import__("pytest")
    entries = [
        {
            "role": "assistant",
            "content": "answer",
            "message_id": "contextual-cron:invalid-base:0",
        }
    ]
    for invalid in (-1, True, "0"):
        record = executions.create_execution(
            f"invalid-base-{invalid!r}", source="builtin"
        )
        executions.mark_execution_running(record["id"])
        with pytest.raises(ValueError, match="base message count"):
            executions.persist_contextual_agent_result(
                record["id"],
                outcome="no_action",
                transcript_session_id="session-1",
                transcript_entries=entries,
                transcript_base_message_count=cast(Any, invalid),
                transcript_base_revision=0,
            )


def test_contextual_transcript_causal_conflict_is_terminal_and_not_deliverable(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("contextual-conflict", source="builtin")
    executions.mark_execution_running(record["id"])
    staged = executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="scheduled answer",
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "assistant",
                "content": "scheduled answer",
                "message_id": f"contextual-cron:{record['id']}:0",
            }
        ],
        transcript_base_message_count=2,
        transcript_base_revision=5,
    )
    assert staged is not None

    assert executions.mark_contextual_transcript_conflict(
        record["id"],
        error="Transcript advanced before recovery.",
    ) is True
    assert executions.mark_contextual_transcript_conflict(
        record["id"],
        error="duplicate",
    ) is False

    conflicted = executions.get_execution(record["id"])
    assert conflicted is not None
    assert conflicted["status"] == "unknown"
    assert conflicted["phase"] == "unknown"
    assert conflicted["transcript_state"] == "conflict"
    assert conflicted["delivery_state"] == "unknown"
    assert conflicted["error"] == "Transcript advanced before recovery."
    assert executions.list_pending_contextual_transcripts() == []
    assert executions.claim_contextual_delivery(record["id"]) is None


def test_retention_preserves_terminal_execution_until_transcript_outbox_applies(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    record = executions.create_execution("contextual-transcript", source="builtin")
    executions.mark_execution_running(record["id"])

    staged = executions.persist_contextual_agent_result(
        record["id"],
        outcome="no_action",
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "user",
                "content": "hidden trigger",
                "display_kind": "hidden",
                "message_id": f"contextual-cron:{record['id']}:0",
            }
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    assert staged is not None
    assert executions.get_execution(record["id"]) is not None
    assert executions.mark_contextual_transcript_applied(record["id"]) is True
    assert executions.get_execution(record["id"]) is None


def test_execution_ledger_follows_task_local_cron_store_after_import(monkeypatch, tmp_path):
    import cron.executions as executions
    from cron.jobs import use_cron_store

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", executions._IMPORT_EXECUTIONS_FILE)
    first = tmp_path / "first"
    second = tmp_path / "second"
    with use_cron_store(first):
        executions.create_execution("first-job", source="test")
    with use_cron_store(second):
        executions.create_execution("second-job", source="test")

    assert (first / "cron" / "executions.db").is_file()
    assert (second / "cron" / "executions.db").is_file()
    with sqlite3.connect(first / "cron" / "executions.db") as conn:
        assert conn.execute("SELECT job_id FROM executions").fetchall() == [("first-job",)]
    with sqlite3.connect(second / "cron" / "executions.db") as conn:
        assert conn.execute("SELECT job_id FROM executions").fetchall() == [("second-job",)]


def test_contextual_delivery_target_and_job_accounting_are_immutable(monkeypatch, tmp_path):
    import concurrent.futures

    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable-target", source="builtin")
    first = {
        "id": "immutable-target",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "42"},
    }
    second = {
        "id": "immutable-target",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "99"},
    }

    assert executions.seal_contextual_delivery_target(record["id"], target=first)
    assert not executions.seal_contextual_delivery_target(record["id"], target=second)
    stored = executions.get_execution(record["id"])
    assert json.loads(stored["delivery_target_json"]) == first

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(
            pool.map(
                lambda _index: executions.claim_contextual_job_accounting(record["id"]),
                range(8),
            )
        )
    assert claims.count(True) == 1


def test_contextual_job_accounting_reconciles_without_double_increment(monkeypatch, tmp_path):
    import cron.scheduler as scheduler
    from cron import jobs as cron_jobs

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(
            prompt="account once",
            schedule="every 1h",
            repeat=5,
            deliver="local",
        )
        record = executions.create_execution(job["id"], source="builtin")
        assert executions.seal_contextual_delivery_target(
            record["id"],
            target={"id": job["id"], "deliver": "local", "origin": {}},
        )
        executions.mark_execution_running(record["id"])
        executions.persist_contextual_agent_result(record["id"], outcome="no_action")

        # Simulate a crash after jobs.json committed but before the ledger's
        # job_accounted bit was acknowledged.
        cron_jobs.mark_job_run(
            job["id"],
            True,
            None,
            execution_id=record["id"],
        )
        assert scheduler.reconcile_contextual_job_accounting() == 1
        assert scheduler.reconcile_contextual_job_accounting() == 0

        stored = cron_jobs.get_job(job["id"])
        assert stored is not None
        assert stored["repeat"]["completed"] == 1
        assert "_pending_accounting_execution_ids" not in stored
        accounted = executions.get_execution(record["id"])
        assert accounted is not None
        assert accounted["job_accounted"] == 1


def test_contextual_accounting_waits_for_jobs_lock_then_retries(
    monkeypatch, tmp_path
):
    import cron.scheduler as scheduler
    from cron import jobs as cron_jobs

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(
            prompt="retry protected accounting",
            schedule="every 1h",
            repeat=5,
            deliver="local",
        )
        record = executions.create_execution(
            job["id"], source="builtin", requires_job_accounting=True
        )
        executions.finish_contextual_execution(record["id"], outcome="no_action")
        jobs_file = cron_jobs._current_cron_store().jobs_file
        before = jobs_file.read_bytes()
        real_jobs_lock = cron_jobs._jobs_lock

        @contextmanager
        def degraded_lock():
            yield False

        monkeypatch.setattr(cron_jobs, "_jobs_lock", degraded_lock)

        assert scheduler._account_contextual_job_run(
            execution_id=record["id"],
            job_id=job["id"],
            success=True,
            error=None,
            delivery_error=None,
        ) is False
        assert jobs_file.read_bytes() == before
        unaccounted = executions.get_execution(record["id"])
        assert unaccounted is not None
        assert unaccounted["job_accounted"] == 0

        monkeypatch.setattr(cron_jobs, "_jobs_lock", real_jobs_lock)
        assert scheduler._account_contextual_job_run(
            execution_id=record["id"],
            job_id=job["id"],
            success=True,
            error=None,
            delivery_error=None,
        ) is True
        stored = cron_jobs.get_job(job["id"])
        assert stored is not None
        assert stored["repeat"]["completed"] == 1
        accounted = executions.get_execution(record["id"])
        assert accounted is not None
        assert accounted["job_accounted"] == 1


def test_reconcile_multiple_crash_gap_markers_never_double_counts(monkeypatch, tmp_path):
    from cron import executions
    from cron import jobs as cron_jobs
    from cron import scheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(
            "count once each",
            "every 1h",
            repeat=10,
            deliver="local",
        )
        records = []
        for _ in range(2):
            record = executions.create_execution(job["id"], source="builtin")
            executions.seal_contextual_delivery_target(
                record["id"],
                target={"id": job["id"], "deliver": "local"},
            )
            executions.persist_contextual_agent_result(
                record["id"],
                outcome="no_action",
            )
            cron_jobs.mark_job_run(
                job["id"],
                True,
                execution_id=record["id"],
            )
            records.append(record)

        before = cron_jobs.get_job(job["id"])
        assert before is not None
        assert before["repeat"]["completed"] == 2
        assert set(before["_pending_accounting_execution_ids"]) == {
            record["id"] for record in records
        }

        assert scheduler.reconcile_contextual_job_accounting() == 2
        after = cron_jobs.get_job(job["id"])
        assert after is not None
        assert after["repeat"]["completed"] == 2
        assert "_pending_accounting_execution_ids" not in after
        assert scheduler.reconcile_contextual_job_accounting() == 0


def test_concurrent_contextual_accounting_has_one_linearization_winner(
    monkeypatch, tmp_path
):
    from cron import jobs as cron_jobs
    from cron import scheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(
            "concurrent accounting",
            "every 1h",
            repeat=10,
            deliver="local",
        )
        record = executions.create_execution(job["id"], source="builtin")
        executions.persist_contextual_agent_result(record["id"], outcome="no_action")

    claim_entered = threading.Event()
    release_claim = threading.Event()
    real_claim = scheduler.claim_contextual_job_accounting

    def slow_claim(execution_id):
        claim_entered.set()
        assert release_claim.wait(timeout=5)
        return real_claim(execution_id)

    monkeypatch.setattr(scheduler, "claim_contextual_job_accounting", slow_claim)
    results = []

    def account():
        with cron_jobs.use_cron_store(tmp_path):
            results.append(
                scheduler._account_contextual_job_run(
                    execution_id=record["id"],
                    job_id=job["id"],
                    success=True,
                    error=None,
                    delivery_error=None,
                )
            )

    first = threading.Thread(target=account)
    second = threading.Thread(target=account)
    first.start()
    assert claim_entered.wait(timeout=5)
    second.start()
    time.sleep(0.1)
    assert second.is_alive(), "second reconciler bypassed the accounting transaction lock"
    release_claim.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert results == [True, True]
    with cron_jobs.use_cron_store(tmp_path):
        stored = cron_jobs.get_job(job["id"])
    assert stored is not None
    assert stored["repeat"]["completed"] == 1


def test_retention_never_prunes_unaccounted_contextual_occurrence(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1)

    protected = executions.create_execution(
        "protected",
        source="builtin",
        requires_job_accounting=True,
    )
    executions.finish_contextual_execution(protected["id"], outcome="no_action")

    for index in range(3):
        ordinary = executions.create_execution(f"ordinary-{index}", source="builtin")
        executions.finish_execution(ordinary["id"], success=True)

    retained = executions.get_execution(protected["id"])
    assert retained is not None
    assert retained["job_accounted"] == 0
    assert retained["requires_job_accounting"] == 1


def test_pending_delivery_uses_ledger_snapshot_not_mutated_job(monkeypatch, tmp_path):
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True)
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("snapshot-job", source="builtin")
    executions.mark_execution_running(record["id"])
    original = {
        "id": "snapshot-job",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "42", "user_id": "42"},
    }
    assert executions.seal_contextual_delivery_target(record["id"], target=original)
    executions.persist_contextual_agent_result(
        record["id"], outcome="notify", final_response="one message"
    )
    record = executions.get_execution(record["id"])

    delivered = []
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [object()])
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, content, **_kwargs: delivered.append((job, content)) or None,
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler._resume_contextual_delivery_record(
        {
            "id": "snapshot-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "99"},
        },
        record,
    )
    assert delivered == [(original, "one message")]


def test_contextual_columns_migrate_an_existing_ledger(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    conn = sqlite3.connect(executions.EXECUTIONS_FILE)
    conn.execute(
        """CREATE TABLE executions (
             id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
             process_id TEXT NOT NULL, pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL, claimed_at TEXT NOT NULL,
             started_at TEXT, finished_at TEXT, error TEXT
           )"""
    )
    conn.commit()
    conn.close()

    executions.create_execution("migrated", source="builtin")
    conn = sqlite3.connect(executions.EXECUTIONS_FILE)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(executions)")}
    conn.close()
    assert {
        "session_key",
        "admitted_session_id",
        "admitted_at",
        "outcome",
        "result_json",
    } <= columns


def test_typed_failure_preserves_status_compatibility(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("stale", source="builtin")
    result = executions.finish_contextual_execution(
        record["id"], outcome="stale", error="reset after admission"
    )
    assert result["status"] == "failed"
    assert result["outcome"] == "stale"
    assert result["error"] == "reset after admission"


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
    assert records[0]["outcome"] == "unknown"
    assert json.loads(records[0]["result_json"])["kind"] == "unknown"
    assert records[0]["finished_at"]
    assert "restart" in records[0]["error"].lower()
    # Recovery only classifies the old attempt. It must not manufacture a new
    # claimed record (which would imply an automatic retry).
    assert [r["status"] for r in records] == ["unknown"]


def test_recovery_does_not_consume_contextual_job_before_occurrence_claim(
    monkeypatch, tmp_path
):
    import cron.jobs as cron_jobs

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        due_at = (cron_jobs._hermes_now() - timedelta(minutes=1)).isoformat()
        cron_jobs.save_jobs(
            [
                {
                    "id": "pre-claim",
                    "prompt": "continue",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": due_at,
                    "enabled": True,
                    "session_target": "current",
                }
            ]
        )
        record = executions.create_execution(
            "pre-claim", source="builtin", requires_job_accounting=True
        )
        assert executions.seal_contextual_delivery_target(
            record["id"], target={"deliver": "local"}
        )
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_id='dead-owner', pid=999999 WHERE id=?",
                (record["id"],),
            )
        before = cron_jobs._current_cron_store().jobs_file.read_bytes()

        assert executions.recover_interrupted_executions() == 1
        recovered = executions.get_execution(record["id"])
        assert recovered is not None
        assert recovered["status"] == "failed"
        assert recovered["outcome"] == "rejected"
        assert recovered["job_accounted"] == 1
        assert cron_jobs._current_cron_store().jobs_file.read_bytes() == before
        job = cron_jobs.get_job("pre-claim")
        assert job is not None
        assert job["next_run_at"] == due_at
        assert job.get("last_run_at") is None


def test_recovery_classifies_contextual_job_after_occurrence_claim_unknown(
    monkeypatch, tmp_path
):
    import cron.jobs as cron_jobs

    executions = _point_ledger(monkeypatch, tmp_path)
    with cron_jobs.use_cron_store(tmp_path):
        due_at = (cron_jobs._hermes_now() - timedelta(minutes=1)).isoformat()
        cron_jobs.save_jobs(
            [
                {
                    "id": "post-claim",
                    "prompt": "continue",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": due_at,
                    "enabled": True,
                    "session_target": "current",
                }
            ]
        )
        record = executions.create_execution(
            "post-claim", source="builtin", requires_job_accounting=True
        )
        assert executions.seal_contextual_delivery_target(
            record["id"], target={"deliver": "local"}
        )
        assert cron_jobs.claim_contextual_occurrence(
            "post-claim",
            execution_id=record["id"],
            expected_next_run_at=due_at,
        )
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_id='dead-owner', pid=999999 WHERE id=?",
                (record["id"],),
            )

        assert executions.recover_interrupted_executions() == 1
        recovered = executions.get_execution(record["id"])
        assert recovered is not None
        assert recovered["status"] == "unknown"
        assert recovered["job_accounted"] == 0


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
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 0)
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
        lambda execution_id: (
            events.append(("running", execution_id)) or {"status": "running"}
        ),
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

    assert len(opened) == 7
    assert len(closed) == 7
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


def test_contextual_delivery_waits_until_transcript_outbox_is_applied(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("job", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="final",
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "assistant",
                "content": "final",
                "message_id": f"contextual-cron:{record['id']}:0",
            }
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    assert executions.claim_contextual_delivery(record["id"]) is None
    executions.mark_contextual_transcript_applied(record["id"])
    assert executions.claim_contextual_delivery(record["id"]) is not None


def test_contextual_delivery_has_one_durable_claim_winner(monkeypatch, tmp_path):
    import concurrent.futures

    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("deliver-once", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="one message",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(
            pool.map(
                lambda _index: executions.claim_contextual_delivery(record["id"]),
                range(8),
            )
        )

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0]["phase"] == "delivering"
    assert winners[0]["delivery_state"] == "claimed"

    finished = executions.finish_contextual_delivery(
        record["id"],
        delivery_state="sent",
    )
    assert finished["status"] == "completed"
    assert finished["phase"] == "completed"
    assert finished["delivery_state"] == "sent"


def test_safe_pending_delivery_survives_restart_but_inflight_send_becomes_unknown(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    pending = executions.create_execution("pending", source="builtin")
    executions.mark_execution_running(pending["id"])
    executions.persist_contextual_agent_result(
        pending["id"], outcome="notify", final_response="safe to resume"
    )

    inflight = executions.create_execution("inflight", source="builtin")
    executions.mark_execution_running(inflight["id"])
    executions.persist_contextual_agent_result(
        inflight["id"], outcome="notify", final_response="may have sent"
    )
    executions.claim_contextual_delivery(inflight["id"])

    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-owner', pid=? "
            "WHERE id IN (?, ?)",
            (999999, pending["id"], inflight["id"]),
        )

    assert executions.recover_interrupted_executions() == 1
    pending_after = executions.get_execution(pending["id"])
    inflight_after = executions.get_execution(inflight["id"])
    assert pending_after["status"] == "running"
    assert pending_after["delivery_state"] == "pending"
    assert inflight_after["status"] == "unknown"
    assert inflight_after["delivery_state"] == "unknown"
    assert "delivery" in inflight_after["error"].lower()
    assert [row["id"] for row in executions.list_pending_contextual_deliveries()] == [
        pending["id"]
    ]


def test_non_notify_contextual_result_finalizes_without_delivery(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("silent", source="builtin")
    executions.mark_execution_running(record["id"])

    finished = executions.persist_contextual_agent_result(
        record["id"], outcome="no_action"
    )

    assert finished["status"] == "completed"
    assert finished["phase"] == "completed"
    assert finished["delivery_state"] == "not_applicable"
    assert executions.claim_contextual_delivery(record["id"]) is None


def test_scheduler_persists_notify_before_claiming_delivery(monkeypatch):
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True)
    events = []
    outcome = type(
        "Outcome",
        (),
        {"kind": "notify", "final_response": "hello", "error": None},
    )()
    monkeypatch.setattr(
        scheduler,
        "get_execution",
        lambda _id: {
            "status": "claimed",
            "phase": "claimed",
            "delivery_target_json": '{"id":"job","deliver":"origin"}',
        },
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _id: True)
    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    monkeypatch.setattr(
        scheduler,
        "persist_contextual_agent_result",
        lambda execution_id, **kwargs: events.append(("persist", kwargs))
        or {"id": execution_id, "delivery_state": "pending"},
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda execution_id: events.append(("claim", execution_id))
        or {"id": execution_id},
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda execution_id, **kwargs: events.append(("finish_delivery", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_args, **_kwargs: events.append(("send", None)) or None,
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    job = {
        "id": "job",
        "execution_id": "exec",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
        "context_binding": {
            "platform": "telegram",
            "chat_id": "42",
            "chat_type": "dm",
            "user_id": "42",
            "session_key": "telegram:dm:42:42",
            "session_id": "session-1",
            "routing_revision": 0,
            "profile": "",
        },
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "42", "user_id": "42"},
    }
    assert scheduler.run_one_job(
        job,
        contextual_dispatch=lambda *_args, **_kwargs: outcome,
    ) is True

    names = [event[0] for event in events]
    assert names.index("persist") < names.index("claim") < names.index("send")
    assert names.count("send") == 1
