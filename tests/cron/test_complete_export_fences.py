"""Real SQLite writer-fence and closed-schema contracts for Cron export."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import re
import sqlite3

import pytest

import hermes_cron_complete_export as module
from cron import executions, jobs


@pytest.fixture()
def execution_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
    row = executions.create_execution("opaque-fenced-job", source="synthetic-fence")
    return home, path, row


def _records():
    with module.open_existing_cron_complete_capture() as capture:
        return tuple(capture.iter_records())


def test_complete_capture_holds_actual_sqlite_writer_fence(execution_store) -> None:
    _home, path, row = execution_store
    with module.open_existing_cron_complete_capture() as capture:
        records = tuple(capture.iter_records())
        assert any(record.key == row["id"] for record in records)
        with closing(sqlite3.connect(path, timeout=0, isolation_level=None)) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
                writer.execute("BEGIN IMMEDIATE")
    # The actual producer must recover after the capture releases its fence.
    assert executions.mark_execution_running(row["id"])["status"] == "running"


def test_capture_refuses_an_existing_independent_sqlite_writer(execution_store) -> None:
    _home, path, row = execution_store
    with closing(sqlite3.connect(path, timeout=0, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(module.CronCaptureRefused):
                with module.open_existing_cron_complete_capture(
                    budget=module.CronCaptureBudget(timeout_seconds=2.0)
                ) as capture:
                    tuple(capture.iter_records())
        finally:
            writer.rollback()
    assert any(record.key == row["id"] for record in _records())


@pytest.mark.parametrize("schema_change", ["table-check", "index-predicate"])
def test_capture_refuses_changed_constraints_with_identical_column_shapes(
    execution_store, schema_change: str,
) -> None:
    _home, path, row = execution_store
    # First establish that this exact real-producer fixture is capturable.
    assert any(record.key == row["id"] for record in _records())
    with closing(sqlite3.connect(path)) as connection:
        if schema_change == "table-check":
            original = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='executions'"
            ).fetchone()[0]
            changed, count = re.subn(
                r"CHECK\(status IN\s*\([^)]*\)\)", "", original, count=1
            )
            assert count == 1 and changed != original
            version = connection.execute("PRAGMA schema_version").fetchone()[0]
            connection.execute("PRAGMA writable_schema=ON")
            connection.execute(
                "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='executions'",
                (changed,),
            )
            connection.execute("PRAGMA writable_schema=OFF")
            connection.execute(f"PRAGMA schema_version={version + 1}")
        else:
            original = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name='idx_executions_occurrence'"
            ).fetchone()[0]
            changed = original.replace("status='completed'", "status='failed'")
            assert changed != original
            connection.execute("DROP INDEX idx_executions_occurrence")
            connection.execute(changed)
        connection.commit()
    with pytest.raises(module.CronCaptureRefused):
        _records()


def test_capture_overflow_is_an_explicit_refusal_after_a_valid_positive(execution_store) -> None:
    _home, _path, row = execution_store
    assert any(record.key == row["id"] for record in _records())
    with pytest.raises((module.CronCaptureFailed, module.CronCaptureRefused)):
        with module.open_existing_cron_complete_capture(
            budget=module.CronCaptureBudget(max_records=1, max_bytes=64)
        ) as capture:
            tuple(capture.iter_records())


def test_capture_store_scope_preserves_callers_configuration_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_constants import get_hermes_home

    launch_home = tmp_path / "launch"
    store_home = tmp_path / "store"
    launch_home.mkdir()
    store_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    with jobs.use_cron_store(store_home):
        job = jobs.create_job(prompt="opaque store-owned job", schedule="every 1h")
        assert get_hermes_home() == launch_home
        with module.open_existing_cron_complete_capture() as capture:
            assert get_hermes_home() == launch_home
            records = tuple(capture.iter_records())
            assert any(record.key == job["id"] for record in records)
        assert get_hermes_home() == launch_home
    assert not (launch_home / "cron" / "jobs.json").exists()


def test_complete_capture_preserves_real_uncheckpointed_execution_wal(execution_store) -> None:
    _home, path, row = execution_store
    with closing(sqlite3.connect(path, isolation_level=None)) as reader:
        reader.execute("BEGIN")
        assert reader.execute(
            "SELECT status FROM executions WHERE id=?", (row["id"],)
        ).fetchone()[0] == "claimed"
        assert executions.mark_execution_running(row["id"])["status"] == "running"
        wal = Path(str(path) + "-wal")
        assert wal.is_file() and wal.stat().st_size > 32
        before = (path.read_bytes(), wal.read_bytes())
        records = _records()
        record = next(record for record in records if record.key == row["id"])
        assert next(field.value for field in record.fields if field.name == "status") == "running"
        assert (path.read_bytes(), wal.read_bytes()) == before
        reader.rollback()


@pytest.mark.parametrize("change", ["same-inode-output", "late-unknown-source", "late-known-family"])
def test_consumed_capture_refuses_business_or_namespace_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    import os

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with jobs.use_cron_store(home):
        job = jobs.create_job(prompt="opaque currentness job", schedule="every 1h")
        output = jobs.save_job_output(job["id"], "alpha")
    with pytest.raises(module.CronCaptureFailed):
        with module.open_existing_cron_complete_capture() as capture:
            records = tuple(capture.iter_records())
            assert any(record.key == job["id"] for record in records)
            if change == "same-inode-output":
                before = output.stat()
                with output.open("r+b") as writer:
                    writer.write(b"omega")
                    writer.flush()
                    os.fsync(writer.fileno())
                os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))
                assert output.stat().st_ino == before.st_ino
                assert output.stat().st_size == before.st_size
            elif change == "late-unknown-source":
                (home / "cron" / "new-business-state.bin").write_bytes(b"opaque")
            else:
                with jobs.use_cron_store(home):
                    # Deliberately bypass the admitted writer to model an external actor.
                    (home / "cron" / "ticker_last_error").write_text("opaque late error")


@pytest.mark.parametrize("late_name", ["unknown-business-source.bin", "ticker_last_error"])
def test_capture_refuses_namespace_created_while_entering_sqlite_fence(
    execution_store, monkeypatch: pytest.MonkeyPatch, late_name: str,
) -> None:
    import threading

    import hermes_state_dbfile as dbfile

    home, _path, _row = execution_store
    at_sqlite_fence = threading.Event()
    allow_sqlite_fence = threading.Event()
    callback_timeouts = []
    errors = []
    completed = []
    connect = dbfile._connect_tracked_db

    def connect_with_trace(*args, **kwargs):
        connection = connect(*args, **kwargs)

        def trace(statement):
            # This callback is emitted by the real SQLite engine immediately
            # before executing its actual writer transaction. No SQL result or
            # capture outcome is fabricated by the synchronization seam.
            if statement.strip().upper() == "BEGIN IMMEDIATE":
                at_sqlite_fence.set()
                if not allow_sqlite_fence.wait(timeout=5.0):
                    callback_timeouts.append(True)

        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(dbfile, "_connect_tracked_db", connect_with_trace)

    def consume():
        try:
            completed.append(_records())
        except module.CronCaptureError as exc:
            errors.append(exc)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert at_sqlite_fence.wait(timeout=5.0)
        (home / "cron" / late_name).write_bytes(b"opaque late business state")
    finally:
        allow_sqlite_fence.set()
        worker.join(timeout=10.0)
    assert not worker.is_alive()
    assert not callback_timeouts
    assert not completed, "capture accepted a source absent from its inventory"
    assert errors and isinstance(errors[0], (module.CronCaptureRefused, module.CronCaptureFailed))


def test_capture_deadline_covers_consumer_after_full_record_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with jobs.use_cron_store(home):
        job = jobs.create_job(prompt="opaque deadline job", schedule="every 1h")
    with pytest.raises(module.CronCaptureFailed):
        with module.open_existing_cron_complete_capture(
            budget=module.CronCaptureBudget(timeout_seconds=4.0)
        ) as capture:
            records = tuple(capture.iter_records())
            assert any(record.key == job["id"] for record in records)
            time.sleep(4.1)  # health: allow HX013 -- actual consumer delay is the capture deadline boundary under test


def test_empty_optional_directories_report_no_retained_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("output", "admissions", "bot_chat_pending"):
        (home / "cron" / name).mkdir(parents=True)
    with module.open_existing_cron_complete_capture() as capture:
        assert not tuple(capture.iter_records())
        assert {
            module.CRON_OUTPUTS_SOURCE_ID,
            module.CRON_ADMISSIONS_SOURCE_ID,
            module.CRON_BOT_CHAT_PENDING_SOURCE_ID,
        }.issubset(capture.manifest.absent_source_ids)


def test_bot_pending_filename_must_match_retained_producer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cron import bot_chat_legacy

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    bot_chat_legacy.defer(
        "opaque-original-id", {"id": "opaque-job"}, "opaque content", home,
        for_failure=False,
    )
    # A real valid producer fixture must first be accepted without repair.
    assert any(record.raw_bytes is not None for record in _records())
    original = home / "cron" / "bot_chat_pending" / "opaque-original-id.json"
    renamed = original.with_name("different-id.json")
    before = original.read_bytes()
    original.rename(renamed)
    with pytest.raises(module.CronCaptureRefused):
        _records()
    assert renamed.read_bytes() == before
    assert not original.exists()


@pytest.mark.parametrize("source_name", ["jobs.json", "ticker_last_error", "suggestions.json"])
def test_capture_refuses_same_inode_prepared_file_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_name: str,
) -> None:
    import os

    from cron import suggestions

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(suggestions, "SUGGESTIONS_FILE", home / "cron" / "suggestions.json")
    with jobs.use_cron_store(home):
        jobs.create_job(prompt="alpha", schedule="every 1h")
        jobs.record_ticker_error("alpha")
    suggestions.add_suggestion(
        title="alpha", description="opaque description", source="catalog",
        job_spec={"prompt": "opaque suggested prompt", "schedule": "every 2h"},
        dedup_key="opaque-dedup",
    )
    path = home / "cron" / source_name
    before = path.stat()
    original = path.read_bytes()
    changed = original.replace(b"alpha", b"omega", 1)
    assert changed != original and len(changed) == len(original)
    with pytest.raises(module.CronCaptureFailed):
        with module.open_existing_cron_complete_capture() as capture:
            assert tuple(capture.iter_records())
            with path.open("r+b") as writer:
                writer.write(changed)
                writer.flush()
                os.fsync(writer.fileno())
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert path.stat().st_ino == before.st_ino
            assert path.stat().st_size == before.st_size


def test_small_capture_budget_prevents_materializing_large_producer_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tracemalloc

    from cron import delivery_queue

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    warm = executions.create_execution("opaque-warm-job", source="synthetic-budget")
    delivery_queue.enqueue(warm["id"], {"id": "opaque-warm-job"}, "opaque warmup")
    assert _records()  # warm imports and establish the genuine producer positive
    large = executions.create_execution("opaque-large-job", source="synthetic-budget")
    delivery_queue.enqueue(large["id"], {"id": "opaque-large-job"}, "x" * (8 * 1024 * 1024))

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    baseline = tracemalloc.get_traced_memory()[0]
    try:
        with pytest.raises((module.CronCaptureFailed, module.CronCaptureRefused)):
            with module.open_existing_cron_complete_capture(
                budget=module.CronCaptureBudget(max_bytes=64 * 1024)
            ) as capture:
                tuple(capture.iter_records())
        peak_delta = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert peak_delta < 4 * 1024 * 1024, "capture materialized the oversized business cell"
