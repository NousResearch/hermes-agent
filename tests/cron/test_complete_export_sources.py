"""TEST-FIRST contracts for the remaining durable stock-cron source families.

These fixtures use the real producer seams for incidents, delivery tombstones,
legacy Bot Chat handoff, ticker markers, recovery telemetry, and usage audit.
They never invoke a model, gateway, external worker, or provider.  The positive
captures intentionally remain RED until the owner-scoped factory is installed.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
import json
import sqlite3

import pytest

import hermes_cron_complete_export as module
from cron import (
    bot_chat_legacy,
    delivery_queue,
    executions,
    incidents,
    jobs,
    occurrences,
    scheduler,
)


_MISSING = object()


def _all_file_bytes(root: Path) -> dict[str, bytes]:
    """Snapshot persistent profile bytes while excluding only transient coordination files."""

    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.name.endswith("-shm")
        and path.relative_to(root).as_posix() != ".cron-export-custody.lock"
    }
    # Native SQLite creates a zero-byte WAL on an ordinary read when the
    # existing database is in WAL mode and close-time checkpointing is disabled.
    # Only these three exact empty coordination files are excluded. Every
    # nonempty WAL, main DB, rollback journal, and other file stays byte-compared.
    empty_wal_paths = {
        "cron/executions.db-wal", "cron/deliveries.db-wal", "cron/notepad.db-wal",
    }
    return {name: raw for name, raw in files.items() if name not in empty_wal_paths or raw}



def _field(record: module.CronCaptureRecord, name: str) -> module.CronCaptureField:
    for field in record.fields:
        if field.name == name:
            return field
    raise AssertionError(f"capture record has no {name!r} field")


def _assert_field(
    record: module.CronCaptureRecord,
    name: str,
    expected: object = _MISSING,
    disposition: module.CronFieldDisposition | None = None,
) -> module.CronCaptureField:
    field = _field(record, name)
    if expected is not _MISSING:
        assert field.value == expected
    if disposition is not None:
        assert field.disposition is disposition
    return field


def _assert_relation(record: module.CronCaptureRecord, target_key: str) -> None:
    assert any(relation.target_key == target_key for relation in record.relations)


def _matches_jsonl(record: module.CronCaptureRecord, raw: bytes) -> bool:
    if record.raw_bytes is not None:
        return record.raw_bytes == raw
    actual = {field.name: field.value for field in record.fields}
    return all(actual.get(name, _MISSING) == value for name, value in json.loads(raw).items())


def _assert_jsonl_lossless(record: module.CronCaptureRecord, raw: bytes) -> None:
    """Accept whole-line custody or the contract's per-scalar raw-byte custody."""

    assert record.raw_bytes == raw or (
        record.raw_bytes is None
        and record.fields
        and all(field.raw_bytes for field in record.fields)
    )


def _seed_job(home: Path) -> dict:
    with jobs.use_cron_store(home):
        return jobs.create_job(
            prompt="opaque source-contract cron prompt",
            schedule="every 1h",
            name="opaque source-contract job",
            deliver="local",
        )


def _capture() -> tuple[module.CronCaptureManifest, tuple[module.CronCaptureRecord, ...]]:
    with module.open_existing_cron_complete_capture(
        budget=module.CronCaptureBudget(max_records=500, max_bytes=2_000_000),
    ) as capture:
        return capture.manifest, tuple(capture.iter_records())


def test_capture_preserves_incidents_tombstones_and_pending_bot_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared execution DB and both durable delivery stores retain all outcomes."""

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    executions_db = home / "cron" / "executions.db"
    deliveries_db = home / "cron" / "deliveries.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", executions_db)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", deliveries_db)

    job = _seed_job(home)
    execution = executions.create_execution(
        job["id"], source="opaque-source-contract", scheduled_instant=job["next_run_at"]
    )
    assert executions.mark_execution_running(execution["id"]) is not None
    execution = executions.finish_execution(
        execution["id"], success=False, error="opaque execution failure"
    )
    assert execution is not None

    incident_id, is_new = incidents.upsert_incident(
        job["id"],
        "opaque incident failure",
        failure_type="config",
        output_file="cron/output/opaque-run.md",
    )
    assert is_new
    assert incidents.set_incident_state(incident_id, "closed")

    # _finish is the delivery owner's durable terminal writer.  Retention zero
    # forces the real owner to move the terminal row into delivery_tombstones;
    # no provider or gateway is started by this fixture.
    previous_limit = delivery_queue.MAX_TERMINAL_DELIVERIES
    try:
        delivery_queue.MAX_TERMINAL_DELIVERIES = 0
        delivery_queue.enqueue(
            execution["id"],
            {"id": job["id"], "name": job["name"], "deliver": "local"},
            "opaque tombstone content",
        )
        claimed = delivery_queue.claim_next()
        assert claimed is not None
        assert delivery_queue._finish(
            execution["id"], error="opaque delivery failure"
        )
        # The producer removes this in-memory owner marker in drain's finally;
        # mirror that source-owned cleanup after calling its terminal seam.
        delivery_queue._ACTIVE_DELIVERIES.discard(execution["id"])
    finally:
        delivery_queue.MAX_TERMINAL_DELIVERIES = previous_limit

    with closing(sqlite3.connect(deliveries_db)) as connection:
        tombstone = connection.execute(
            "SELECT execution_id, terminal_status, finished_at "
            "FROM delivery_tombstones WHERE execution_id=?",
            (execution["id"],),
        ).fetchone()
    assert tombstone is not None
    assert tombstone[1] == "failed"
    assert tombstone[2]

    pending_key = "opaque-bot-chat-pending"
    bot_chat_legacy.defer(
        pending_key,
        {"id": job["id"], "name": job["name"]},
        "opaque pending Bot Chat content",
        home,
        for_failure=False,
    )
    pending_path = home / "cron" / "bot_chat_pending" / f"{pending_key}.json"
    pending_raw = pending_path.read_bytes()
    pending_payload = json.loads(pending_raw)
    assert pending_payload["status"] == "queued"
    assert pending_payload["content"] == "opaque pending Bot Chat content"

    before = _all_file_bytes(home)
    _manifest, records = _capture()

    incident_record = next(
        record
        for record in records
        if record.source_id == module.CRON_EXECUTIONS_SOURCE_ID
        and record.key == incident_id
    )
    _assert_field(
        incident_record,
        "job_id",
        job["id"],
        module.CronFieldDisposition.BUSINESS_METADATA,
    )
    _assert_field(
        incident_record,
        "state",
        "closed",
        module.CronFieldDisposition.BUSINESS_METADATA,
    )
    _assert_field(
        incident_record,
        "failure_type",
        "config",
        module.CronFieldDisposition.BUSINESS_METADATA,
    )
    _assert_field(
        incident_record,
        "error",
        "opaque incident failure",
        module.CronFieldDisposition.BUSINESS_PAYLOAD,
    )
    _assert_relation(incident_record, job["id"])

    tombstone_record = next(
        record
        for record in records
        if record.source_id == module.CRON_DELIVERIES_SOURCE_ID
        and record.key == execution["id"]
    )
    _assert_field(
        tombstone_record,
        "terminal_status",
        "failed",
        module.CronFieldDisposition.DELIVERY_METADATA,
    )
    _assert_field(
        tombstone_record,
        "finished_at",
        tombstone[2],
        module.CronFieldDisposition.DELIVERY_METADATA,
    )
    _assert_relation(tombstone_record, execution["id"])

    bot_record = next(
        record
        for record in records
        if record.source_id == module.CRON_BOT_CHAT_PENDING_SOURCE_ID
        and record.raw_bytes == pending_raw
    )
    assert bot_record.raw_bytes == pending_raw
    _assert_field(
        bot_record,
        "status",
        "queued",
        module.CronFieldDisposition.DELIVERY_METADATA,
    )
    _assert_field(
        bot_record,
        "content",
        pending_payload["content"],
        module.CronFieldDisposition.BUSINESS_PAYLOAD,
    )
    _assert_field(
        bot_record,
        "home",
        pending_payload["home"],
        module.CronFieldDisposition.SECURITY_IDENTITY,
    )
    _assert_relation(bot_record, job["id"])

    # Capture is a read-only custody operation; it must not prune, settle, or
    # rewrite any of the producer-owned persistent bytes.
    assert _all_file_bytes(home) == before


def test_capture_preserves_markers_recovery_telemetry_and_usage_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marker and JSONL producers are captured with exact bytes and typed values."""

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    job = _seed_job(home)

    with jobs.use_cron_store(home):
        jobs.record_ticker_heartbeat(success=True)
        jobs.record_ticker_error("opaque ticker failure")
        occurrences.record_catch_up_occurrence()
        occurrences.record_catch_up_occurrence()
        jobs._record_persisted_error_recovery(job, "opaque previous next run")
        jobs._record_timezone_migration_catchup(
            job,
            datetime(2026, 10, 17, 5, tzinfo=UTC),
            datetime(2026, 10, 17, 6, tzinfo=UTC),
        )

    scheduler._record_forced_release(job["id"], job["name"], 7.5, 60.0)
    audit = scheduler._FireAudit(job, job["id"], "opaque-audit-model")
    audit.write(
        {
            "prompt_tokens": 7,
            "completion_tokens": 11,
            "total_tokens": 18,
            "response_silent": False,
        },
        None,
    )

    marker_paths = {
        "ticker_heartbeat": home / "cron" / "ticker_heartbeat",
        "ticker_last_success": home / "cron" / "ticker_last_success",
        "ticker_last_error": home / "cron" / "ticker_last_error",
        "catch_up_occurrences": home / "cron" / "catch_up_occurrences",
    }
    marker_raw = {name: path.read_bytes() for name, path in marker_paths.items()}
    recovery_paths = {
        "persisted_error_recoveries": home / "cron" / "persisted_error_recoveries.jsonl",
        "timezone_migration_catchups": home / "cron" / "timezone_migration_catchups.jsonl",
        "inflight_forced_releases": home / "cron" / "inflight_forced_releases.jsonl",
    }
    recovery_raw = {name: path.read_bytes() for name, path in recovery_paths.items()}
    usage_path = home / "cron" / "usage_audit.jsonl"
    usage_raw = usage_path.read_bytes()
    usage_payload = json.loads(usage_raw.decode("utf-8").splitlines()[0])
    before = _all_file_bytes(home)

    _manifest, records = _capture()

    for filename, raw in marker_raw.items():
        record = next(
            record
            for record in records
            if record.source_id == module.CRON_CONTROL_MARKERS_SOURCE_ID
            and record.key == filename
        )
        assert record.raw_bytes == raw
        _assert_field(
            record,
            "value",
            raw.decode("utf-8"),
            module.CronFieldDisposition.CONTROL_RECOVERY,
        )
        assert record.relations

    for filename, raw in recovery_raw.items():
        record = next(
            record
            for record in records
            if record.source_id == module.CRON_RECOVERY_TELEMETRY_SOURCE_ID
            and _matches_jsonl(record, raw)
        )
        _assert_jsonl_lossless(record, raw)
        _assert_field(
            record,
            "job_id",
            job["id"],
            module.CronFieldDisposition.CONTROL_RECOVERY,
        )
        assert record.fields
        assert all(
            field.disposition is module.CronFieldDisposition.CONTROL_RECOVERY
            for field in record.fields
        )
        _assert_relation(record, job["id"])

    usage_record = next(
        record
        for record in records
        if record.source_id == module.CRON_USAGE_AUDIT_SOURCE_ID
    )
    _assert_jsonl_lossless(usage_record, usage_raw)
    for name, expected in usage_payload.items():
        assert _field(usage_record, name).value == expected
    _assert_field(
        usage_record,
        "job_id",
        job["id"],
        module.CronFieldDisposition.BUSINESS_METADATA,
    )
    _assert_field(
        usage_record,
        "model",
        "opaque-audit-model",
        module.CronFieldDisposition.BUSINESS_METADATA,
    )
    _assert_field(usage_record, "error", None, module.CronFieldDisposition.BUSINESS_PAYLOAD)
    _assert_relation(usage_record, job["id"])

    assert _all_file_bytes(home) == before


def test_capture_reports_absence_for_unseeded_families_after_real_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real jobs record does not authorize creation of missing optional stores."""

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    job = _seed_job(home)
    before = _all_file_bytes(home)

    manifest, records = _capture()

    assert any(
        record.source_id == module.CRON_JOBS_SOURCE_ID and record.key == job["id"]
        for record in records
    )
    missing = set(module.STOCK_CRON_SOURCE_IDS) - {module.CRON_JOBS_SOURCE_ID}
    assert missing.issubset(set(manifest.absent_source_ids))
    assert not any(record.source_id in missing for record in records)
    assert _all_file_bytes(home) == before
