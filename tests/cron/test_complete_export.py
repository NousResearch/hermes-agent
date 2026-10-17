"""TEST-FIRST contracts for the complete stock-cron export component.

The fixtures use the real cron owners to create a job, output, execution, queued
delivery, notepad value, and suggestion.  They do not start a model provider or
read a source file to prove implementation shape.  The positive capture tests
are intentionally expected to fail until the owner-scoped factory exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
import json
import sqlite3
import threading

import pytest

import hermes_cron_complete_export as module
from cron import delivery_queue, executions, jobs, notepad, scheduler_authority, suggestions


@dataclass(frozen=True)
class _SeededCron:
    home: Path
    job: dict
    output_path: Path
    execution: dict
    delivery_db: Path
    notepad_db: Path
    suggestions_file: Path


def _all_file_bytes(root: Path) -> dict[str, bytes]:
    # SQLite's shared-memory index is coordination, not persistent business state.
    # The exact profile-root custody file is also transient coordination.
    # Keep database, nonempty WAL and rollback-journal bytes in this oracle; never repair
    # sidecars to make an equality assertion pass.
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith("-shm")
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



def _record_field(record: module.CronCaptureRecord, name: str):
    for field in record.fields:
        if field.name == name:
            return field.value
    raise AssertionError(f"capture record has no {name!r} field")


@pytest.fixture()
def seeded_cron(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SeededCron:
    """Seed every business source whose stock owner can be used without a provider."""

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    executions_file = home / "cron" / "executions.db"
    delivery_db = home / "cron" / "deliveries.db"
    notepad_db = home / "cron" / "notepad.db"
    suggestions_file = home / "cron" / "suggestions.json"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", executions_file)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", delivery_db)
    monkeypatch.setattr(notepad, "NOTEPAD_FILE", notepad_db)
    monkeypatch.setattr(suggestions, "SUGGESTIONS_FILE", suggestions_file)

    with jobs.use_cron_store(home):
        job = jobs.create_job(
            prompt="opaque synthetic cron business prompt",
            schedule="every 1h",
            name="opaque synthetic job",
            deliver="local",
        )
        output_path = jobs.save_job_output(
            job["id"], "opaque synthetic run output\nsecond line\n"
        )

    execution = executions.create_execution(
        job["id"], source="synthetic-test", scheduled_instant=job["next_run_at"]
    )
    assert executions.mark_execution_running(execution["id"]) is not None
    execution = executions.finish_execution(
        execution["id"], success=False, error="synthetic opaque failure"
    )
    assert execution is not None

    delivery_queue.enqueue(
        execution["id"],
        {"id": job["id"], "name": job["name"], "deliver": "local"},
        "opaque queued delivery",
    )
    notepad.set_note(job["id"], "opaque-cursor", "opaque-value")
    suggestions.add_suggestion(
        title="Opaque suggestion",
        description="Opaque suggestion description",
        source="catalog",
        job_spec={"prompt": "opaque suggested prompt", "schedule": "every 2h"},
        dedup_key="opaque-dedup",
    )
    return _SeededCron(
        home=home,
        job=job,
        output_path=output_path,
        execution=execution,
        delivery_db=delivery_db,
        notepad_db=notepad_db,
        suggestions_file=suggestions_file,
    )


def test_budget_rejects_unbounded_or_malformed_arguments() -> None:
    with pytest.raises(ValueError):
        module.CronCaptureBudget(max_records=0)
    with pytest.raises(ValueError):
        module.CronCaptureBudget(max_bytes=-1)
    with pytest.raises(ValueError):
        module.CronCaptureBudget(timeout_seconds=float("inf"))
    with pytest.raises(TypeError):
        module.open_existing_cron_complete_capture(budget=object())


def test_positive_capture_preserves_real_owner_records_losslessly(
    seeded_cron: _SeededCron,
) -> None:
    """The missing factory is the intended RED for this first component."""

    home = seeded_cron.home
    with closing(sqlite3.connect(seeded_cron.delivery_db)) as conn:
        queued = conn.execute(
            "SELECT execution_id, status, content FROM deliveries WHERE execution_id=?",
            (seeded_cron.execution["id"],),
        ).fetchone()
    assert queued == (seeded_cron.execution["id"], "pending", "opaque queued delivery")
    with closing(sqlite3.connect(seeded_cron.notepad_db)) as conn:
        note = conn.execute(
            "SELECT job_id, key, value FROM cron_notepad WHERE job_id=? AND key=?",
            (seeded_cron.job["id"], "opaque-cursor"),
        ).fetchone()
    assert note == (seeded_cron.job["id"], "opaque-cursor", "opaque-value")
    # Take the byte oracle after the independent readers have really closed.
    raw_before = _all_file_bytes(home)
    jobs_raw = raw_before["cron/jobs.json"]
    output_key = seeded_cron.output_path.relative_to(home).as_posix()
    output_raw = raw_before[output_key]
    suggestions_raw = raw_before["cron/suggestions.json"]

    with module.open_existing_cron_complete_capture(
        budget=module.CronCaptureBudget(max_records=100, max_bytes=1_000_000)
    ) as capture:
        manifest = capture.manifest
        records = tuple(capture.iter_records())

    assert manifest.source_id == module.STOCK_CRON_SOURCE_ID
    assert set(module.STOCK_CRON_SOURCE_IDS).issubset(manifest.source_ids)
    job_records = [record for record in records if record.source_id == module.CRON_JOBS_SOURCE_ID]
    output_records = [record for record in records if record.source_id == module.CRON_OUTPUTS_SOURCE_ID]
    execution_records = [
        record for record in records if record.source_id == module.CRON_EXECUTIONS_SOURCE_ID
    ]
    delivery_records = [
        record for record in records if record.source_id == module.CRON_DELIVERIES_SOURCE_ID
    ]
    assert any(record.raw_bytes == jobs_raw for record in job_records)
    assert any(record.raw_bytes == output_raw for record in output_records)
    execution_record = next(
        record for record in execution_records if record.key == seeded_cron.execution["id"]
    )
    assert _record_field(execution_record, "status") == "failed"
    delivery_record = next(
        record for record in delivery_records if record.key == seeded_cron.execution["id"]
    )
    assert _record_field(delivery_record, "status") == queued[1]
    assert _record_field(delivery_record, "content") == queued[2]
    notepad_records = [
        record for record in records if record.source_id == module.CRON_NOTEPAD_SOURCE_ID
    ]
    assert any(
        _record_field(record, "key") == note[1]
        and _record_field(record, "value") == note[2]
        for record in notepad_records
        if any(field.name == "key" for field in record.fields)
    )
    suggestion_records = [
        record for record in records if record.source_id == module.CRON_SUGGESTIONS_SOURCE_ID
    ]
    assert any(
        record.raw_bytes == suggestions_raw
        or _record_field(record, "dedup_key") == "opaque-dedup"
        for record in suggestion_records
    )

    # A complete capture must not mutate any persistent source while it is read.
    assert _all_file_bytes(home) == raw_before


def test_empty_profile_reports_absent_optional_sources_without_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unused cron store is an explicit absence, never a missing provider."""

    home = tmp_path / "empty-profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = _all_file_bytes(home)

    with module.open_existing_cron_complete_capture() as capture:
        manifest = capture.manifest
        records = tuple(capture.iter_records())

    assert not records
    assert set(module.STOCK_CRON_SOURCE_IDS).issubset(manifest.absent_source_ids)
    assert _all_file_bytes(home) == before
    assert not (home / "cron").exists()


def test_malformed_jobs_store_refuses_without_repair_or_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "malformed-profile"
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    jobs_file = cron_dir / "jobs.json"
    jobs_file.write_bytes(b'{"jobs": [')
    before = _all_file_bytes(home)

    with pytest.raises(module.CronCaptureRefused):
        with module.open_existing_cron_complete_capture() as capture:
            tuple(capture.iter_records())

    assert _all_file_bytes(home) == before


def test_unknown_cron_artifact_refuses_without_repair_or_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "unknown-profile"
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    unknown = cron_dir / "unexpected-business-store.bin"
    unknown.write_bytes(b"opaque unknown source")
    before = _all_file_bytes(home)

    with pytest.raises(module.CronCaptureRefused):
        with module.open_existing_cron_complete_capture() as capture:
            tuple(capture.iter_records())

    assert _all_file_bytes(home) == before


def test_configured_monitor_state_without_snapshot_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "configured-missing-profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with jobs.use_cron_store(home):
        job = jobs.create_job(
            prompt="opaque monitor job",
            schedule="every 1h",
            monitor_script="configured-source.sh",
        )
        jobs.update_job(
            job["id"],
            {"monitor_state": {"last_output_hash": "a" * 64, "last_changed_at": "opaque"}},
        )
    before = _all_file_bytes(home)

    with pytest.raises(module.CronCaptureRefused):
        with module.open_existing_cron_complete_capture() as capture:
            tuple(capture.iter_records())

    assert _all_file_bytes(home) == before


def test_unknown_execution_table_refuses_without_sqlite_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "unknown-sqlite-profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    execution_db = home / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", execution_db)
    row = executions.create_execution("opaque-job", source="synthetic-test")
    assert row["status"] == "claimed"
    with closing(sqlite3.connect(execution_db)) as conn:
        conn.execute("CREATE TABLE unknown_business_state (value TEXT)")
        conn.execute("INSERT INTO unknown_business_state(value) VALUES ('opaque')")
        conn.commit()
    before = _all_file_bytes(home)

    with pytest.raises(module.CronCaptureRefused):
        with module.open_existing_cron_complete_capture() as capture:
            tuple(capture.iter_records())

    assert _all_file_bytes(home) == before


def test_capture_rejects_incomplete_record_consumption(
    seeded_cron: _SeededCron,
) -> None:
    """A future implementation must make partial iteration a typed failure."""

    with pytest.raises(module.CronCaptureFailed):
        with module.open_existing_cron_complete_capture() as capture:
            iterator = iter(capture.iter_records())
            first = next(iterator)
            assert first.source_id in module.STOCK_CRON_SOURCE_IDS
            # Do not consume the iterator's tail: the context must refuse publication
            # rather than report a partial success.


def test_capture_detects_a_real_output_writer_drift(
    seeded_cron: _SeededCron, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output saving has no jobs lock and therefore needs an owner-scoped hook."""

    writer_started = threading.Event()
    release_writer = threading.Event()
    original_write = jobs.atomic_write_text
    errors = []
    expected = "writer drift"

    def blocked_write(path, output, **kwargs):
        # Pause at the real writer's I/O boundary, after its source-owned
        # admission/fence would have been acquired. An outer function wrapper
        # would be invisible to the capture and could not prove this contract.
        if output == expected:
            writer_started.set()
            assert release_writer.wait(timeout=10.0)
        return original_write(path, output, **kwargs)

    monkeypatch.setattr(jobs, "atomic_write_text", blocked_write)

    def save_concurrently() -> None:
        try:
            with jobs.use_cron_store(seeded_cron.home):
                jobs.save_job_output(seeded_cron.job["id"], expected)
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=save_concurrently)
    writer.start()
    try:
        assert writer_started.wait(timeout=2.0)
        with pytest.raises(module.CronCaptureRefused):
            with module.open_existing_cron_complete_capture(
                budget=module.CronCaptureBudget(timeout_seconds=2.0)
            ) as capture:
                tuple(capture.iter_records())
    finally:
        release_writer.set()
        writer.join(timeout=2.0)
    assert not writer.is_alive()
    assert not errors


def test_admission_journal_identity_is_opaque_and_profile_local(
    seeded_cron: _SeededCron,
) -> None:
    """Prepared admission state must be captured by its owner identity, not guessed paths."""

    execution_id = seeded_cron.execution["id"]
    journal = scheduler_authority.journal_path(seeded_cron.job["id"], execution_id)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_bytes(
        json.dumps(
            {
                "params": {
                    "job_id": seeded_cron.job["id"],
                    "request_id": execution_id,
                    "extra_prompt": None,
                },
                "receipt": {"session_id": "opaque-session", "admission_id": "opaque-admission"},
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    before = _all_file_bytes(seeded_cron.home)

    with module.open_existing_cron_complete_capture() as capture:
        records = tuple(capture.iter_records())

    admission_records = [
        record for record in records if record.source_id == module.CRON_ADMISSIONS_SOURCE_ID
    ]
    assert any(record.key == journal.name for record in admission_records)
    assert _all_file_bytes(seeded_cron.home) == before
