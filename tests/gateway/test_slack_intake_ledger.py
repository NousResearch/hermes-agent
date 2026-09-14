"""Public behavior contracts for the durable Slack intake ledger."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import sqlite3
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


async def test_cold_executor_start_failure_retains_bounded_admission(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slack-intake-start-fault")
    permits = threading.BoundedSemaphore(32)
    monkeypatch.setattr(ledger, "_LEDGER_EXECUTOR", executor, raising=False)
    monkeypatch.setattr(ledger, "_LEDGER_PERMITS", permits)
    loop = asyncio.get_running_loop()
    previous_default = getattr(loop, "_default_executor", None)
    loop.set_default_executor(executor)
    original_start = threading.Thread.start

    def fail_start(thread):
        if thread.name.startswith("slack-intake-start-fault"):
            raise RuntimeError("synthetic worker start failure")
        return original_start(thread)

    outcomes = []
    try:
        with monkeypatch.context() as fault:
            fault.setattr(threading.Thread, "start", fail_start)
            for index in range(40):
                outcomes.append(
                    await ledger.record_listener_received_safely(
                        **_kwargs(event_id=f"Ev-start-fault-{index}")
                    )
                )

        assert all(
            not outcome.persisted and outcome.failure_reason == "persistence_failed"
            for outcome in outcomes
        )
        assert executor._work_queue.qsize() <= 32
        assert len(ledger.read_fallback_receipts()) == 40
        assert ledger.persistence_degraded()
        assert not ledger._db_path().exists()
        available = 0
        while permits.acquire(blocking=False):
            available += 1
        for _ in range(available):
            permits.release()
        assert available == 32

        recovered = await ledger.record_listener_received_safely(
            **_kwargs(event_id="Ev-start-recovered")
        )
        assert recovered.persisted
        assert not ledger.persistence_degraded()
        assert ledger._LEDGER_EXECUTOR is not executor
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)
    finally:
        loop._default_executor = previous_default
        if getattr(ledger, "_LEDGER_EXECUTOR", None) is not executor:
            ledger._LEDGER_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        executor.shutdown(wait=True, cancel_futures=True)


def _ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "slack-intake.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    return ledger


def _kwargs(**overrides):
    values = {
        "workspace_id": "T-workspace-private",
        "event_id": "Ev-event-private",
        "transport_id": "Envelope-private",
        "event_type": "message",
        "channel_id": "C-channel-private",
        "thread_id": "1712345.100",
        "message_id": "1712345.200",
        "received_at": 1_700_000_000.0,
    }
    values.update(overrides)
    return values


class _ConnectionProxy:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, parameters=()):
        return self._connection.execute(sql, parameters)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._connection.__exit__(exc_type, exc, traceback)

    def __getattr__(self, name):
        return getattr(self._connection, name)


async def test_receipt_round_trip_contains_only_digests_and_fixed_metadata(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)

    observation = ledger.record_listener_received(**_kwargs())
    rows = ledger.read_receipts()

    assert observation.duplicate is False
    assert len(rows) == 1
    row = rows[0]
    assert row["receipt_id"] == observation.receipt_id
    assert row["event_type"] == "message"
    assert row["terminal_state"] is None
    assert row["receive_count"] == 1
    assert [event["stage"] for event in row["events"]] == ["listener_received"]
    assert all(len(row[field]) == 64 for field in (
        "receipt_id", "workspace_hash", "event_hash", "channel_hash",
        "thread_hash", "message_hash", "message_key_hash",
    ))


async def test_raw_slack_identifiers_never_reach_ledger_files(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    raw = _kwargs()
    secrets = tuple(
        raw[name]
        for name in (
            "workspace_id", "event_id", "transport_id", "channel_id", "thread_id", "message_id"
        )
    )

    receipt = ledger.record_listener_received(**_kwargs())
    ledger.append_stage(receipt.receipt_id, stage="listener_entered")
    ledger.mark_dropped(receipt.receipt_id, reason="ignored_channel")

    stored = b"".join(path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    for value in secrets:
        assert value.encode() not in stored


async def test_duplicate_receipt_increments_count_and_records_duplicate_stage(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)

    first = ledger.record_listener_received(**_kwargs())
    second = ledger.record_listener_received(**_kwargs(received_at=1_700_000_001.0))
    third = ledger.record_listener_received(**_kwargs(received_at=1_699_999_999.0))
    row = ledger.read_receipts()[0]

    assert first.receipt_id == second.receipt_id == third.receipt_id
    assert second.duplicate is third.duplicate is True
    assert row["receive_count"] == 3
    assert row["last_received_at"] == 1_700_000_001.0
    assert [event["stage"] for event in row["events"]] == [
        "listener_received", "listener_duplicate", "listener_duplicate",
    ]


async def test_event_retention_preserves_initial_stage_for_later_duplicate(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt = ledger.record_listener_received(**_kwargs())
    for index in range(ledger._MAX_EVENTS_PER_RECEIPT + 2):
        ledger.append_stage(
            receipt.receipt_id,
            stage="acknowledged",
        )

    duplicate = ledger.record_listener_received(**_kwargs(received_at=1_700_000_001.0))

    assert duplicate.duplicate is True
    row = ledger.read_receipts()[0]
    assert row["events"][0]["stage"] == "listener_received"
    assert len(row["events"]) <= ledger._MAX_EVENTS_PER_RECEIPT


async def test_same_event_id_with_different_metadata_fails_closed(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger.record_listener_received(**_kwargs())

    with pytest.raises(ledger.InvalidIntakeTransition, match="identity metadata changed"):
        ledger.record_listener_received(**_kwargs(channel_id="C-other"))

    assert ledger.read_receipts()[0]["receive_count"] == 1


async def test_progress_stage_is_idempotent_when_repeated_adjacent(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt = ledger.record_listener_received(**_kwargs())

    first = ledger.append_stage(receipt.receipt_id, stage="listener_entered")
    second = ledger.append_stage(receipt.receipt_id, stage="listener_entered")

    assert first.appended is True
    assert second.appended is False
    assert [event["stage"] for event in ledger.read_receipts()[0]["events"]] == [
        "listener_received", "listener_entered",
    ]


async def test_progress_stage_is_idempotent_across_two_connection_race(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt = ledger.record_listener_received(**_kwargs())
    original_open = ledger._open_connection
    first_locked = threading.Event()
    release_first = threading.Event()
    second_begin_attempted = threading.Event()
    outcomes = []
    errors = []

    class _NoProcessLock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _RaceConnection(_ConnectionProxy):
        def __init__(self, connection, owner):
            super().__init__(connection)
            self._owner = owner

        def execute(self, sql, parameters=()):
            if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
                if self._owner == "first":
                    result = super().execute(sql, parameters)
                    first_locked.set()
                    if not release_first.wait(2):
                        raise TimeoutError("first progress writer was not released")
                    return result
                second_begin_attempted.set()
            return super().execute(sql, parameters)

    def open_racing_connection():
        return _RaceConnection(original_open(), threading.current_thread().name)

    def append():
        try:
            outcomes.append(
                ledger.append_stage(receipt.receipt_id, stage="listener_entered")
            )
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(ledger, "_DB_LOCK", _NoProcessLock())
    monkeypatch.setattr(ledger, "_open_connection", open_racing_connection)
    first = threading.Thread(target=append, name="first")
    second = threading.Thread(target=append, name="second")
    first.start()
    assert first_locked.wait(2)
    second.start()
    assert second_begin_attempted.wait(2)
    release_first.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert sorted(outcome.appended for outcome in outcomes) == [False, True]
    assert [event["stage"] for event in ledger.read_receipts()[0]["events"]] == [
        "listener_received", "listener_entered",
    ]


async def test_conflicting_terminal_transition_preserves_first_decision(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt = ledger.record_listener_received(**_kwargs())
    ledger.mark_dropped(receipt.receipt_id, reason="ignored_channel")

    with pytest.raises(ledger.InvalidIntakeTransition, match="already terminal"):
        ledger.mark_accepted(receipt.receipt_id)

    row = ledger.read_receipts()[0]
    assert row["terminal_state"] == "dropped"
    assert row["terminal_reason"] == "ignored_channel"


async def test_first_accepted_semantic_message_suppresses_distinct_event_twin(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-message", event_type="message"))
    twin = ledger.record_listener_received(**_kwargs(event_id="Ev-mention", event_type="app_mention"))

    accepted = ledger.mark_accepted(first.receipt_id)
    duplicate = ledger.mark_accepted(twin.receipt_id)

    assert accepted.state == "accepted"
    assert duplicate.state == "dropped"
    assert duplicate.reason == "duplicate_ts"
    assert duplicate.related_receipt_id == first.receipt_id


async def test_first_accepted_app_mention_suppresses_message_twin(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    mention = ledger.record_listener_received(
        **_kwargs(event_id="Ev-mention-first", event_type="app_mention")
    )
    message = ledger.record_listener_received(
        **_kwargs(event_id="Ev-message-second", event_type="message")
    )

    accepted = ledger.mark_accepted(mention.receipt_id)
    duplicate = ledger.mark_accepted(message.receipt_id)

    assert accepted.state == "accepted"
    assert duplicate.state == "dropped"
    assert duplicate.reason == "duplicate_ts"
    assert duplicate.related_receipt_id == mention.receipt_id


async def test_same_timestamp_in_another_channel_is_not_suppressed(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-one"))
    second = ledger.record_listener_received(
        **_kwargs(event_id="Ev-two", channel_id="C-other")
    )

    assert ledger.mark_accepted(first.receipt_id).state == "accepted"
    assert ledger.mark_accepted(second.receipt_id).state == "accepted"


@pytest.mark.parametrize("value", [None, "", " padded", "trail ", "x\x00y", "x" * 257])
async def test_identifier_validation_rejects_ambiguous_values(monkeypatch, tmp_path, value):
    ledger = _ledger(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        ledger.record_listener_received(**_kwargs(event_id=value))
    assert not ledger._db_path().exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "not-time"])
async def test_timestamp_validation_rejects_nonfinite_values(monkeypatch, tmp_path, value):
    ledger = _ledger(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="must be finite"):
        ledger.record_listener_received(**_kwargs(received_at=value))


@pytest.mark.parametrize("value", [[], {}])
@pytest.mark.parametrize(
    "field",
    [
        "initial_event_type",
        "initial_stage",
        "progress_stage",
        "progress_reason",
        "terminal_state",
        "terminal_reason",
    ],
)
async def test_direct_classification_validation_is_total(monkeypatch, tmp_path, field, value):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt_id = "a" * 64
    if field.startswith("progress_") or field.startswith("terminal_"):
        receipt_id = ledger.record_listener_received(**_kwargs()).receipt_id

    with pytest.raises(ValueError):
        if field == "initial_event_type":
            ledger.record_listener_received(**_kwargs(event_type=value))
        elif field == "initial_stage":
            ledger.record_listener_received(**_kwargs(stage=value))
        elif field == "progress_stage":
            ledger.append_stage(receipt_id, stage=value)
        elif field == "progress_reason":
            ledger.append_stage(
                receipt_id, stage="listener_entered", reason=value
            )
        elif field == "terminal_state":
            ledger._mark_terminal(
                receipt_id, state=value, reason=None, decided_at=1.0
            )
        else:
            ledger.mark_dropped(receipt_id, reason=value)


async def test_correlation_key_is_private_and_stable(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)

    first = ledger.hash_identifier("event", "Ev-one")
    key_path = tmp_path / "correlation.key"
    first_key = key_path.read_bytes()
    ledger._reset_initialization_for_tests()
    second = ledger.hash_identifier("event", "Ev-one")

    assert len(first_key) == 32
    assert first == second
    assert key_path.read_bytes() == first_key
    if os.name == "posix":
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


async def _assert_parent_directory_fsync_after_key_publish(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    events = []
    original_link = ledger.os.link
    original_fsync = ledger.os.fsync

    def track_link(source, destination):
        events.append("publish")
        return original_link(source, destination)

    def track_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            events.append("parent_fsync")
            raise OSError("synthetic parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(ledger.os, "link", track_link)
    monkeypatch.setattr(ledger.os, "fsync", track_fsync)

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert events == ["publish", "parent_fsync"]
    assert not outcome.persisted
    assert outcome.failure_reason == "persistence_failed"
    assert not ledger._db_path().exists()
    assert ledger.persistence_degraded()


async def test_retry_after_key_publish_fsync_failure_revalidates_parent_durability(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    original_fsync = ledger.os.fsync
    directory_fsyncs = 0

    def fail_first_directory_fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("synthetic parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(ledger.os, "fsync", fail_first_directory_fsync)

    first = await ledger.record_listener_received_safely(
        **_kwargs(event_id="Ev-first-fsync")
    )
    second = await ledger.record_listener_received_safely(
        **_kwargs(event_id="Ev-retry-fsync")
    )

    assert not first.persisted
    assert second.persisted
    assert directory_fsyncs == 2
    assert [row["event_type"] for row in ledger.read_receipts()] == ["message"]


@pytest.mark.linux_only
async def test_linux_key_publish_fsyncs_parent_directory(monkeypatch, tmp_path):
    await _assert_parent_directory_fsync_after_key_publish(monkeypatch, tmp_path)


@pytest.mark.macos_only
async def test_macos_key_publish_fsyncs_parent_directory(monkeypatch, tmp_path):
    await _assert_parent_directory_fsync_after_key_publish(monkeypatch, tmp_path)


async def test_cached_correlation_key_fails_closed_if_file_changes(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger.hash_identifier("event", "Ev-one")
    key_path = tmp_path / "correlation.key"
    key_path.write_bytes(b"x" * 32)

    with pytest.raises(OSError, match="changed while in use"):
        ledger.hash_identifier("event", "Ev-two")


async def test_read_does_not_create_a_missing_database(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    assert ledger.read_receipts() == []
    assert not ledger._db_path().exists()


@pytest.mark.asyncio
async def test_newer_schema_fails_closed_without_rewriting_it(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    conn = sqlite3.connect(ledger._db_path())
    conn.execute(f"PRAGMA user_version={ledger._SCHEMA_VERSION + 1}")
    conn.close()

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outcome.failure_reason == "schema_mismatch"
    check = sqlite3.connect(ledger._db_path())
    assert check.execute("PRAGMA user_version").fetchone()[0] == ledger._SCHEMA_VERSION + 1
    check.close()


@pytest.mark.asyncio
async def test_versioned_schema_shape_is_validated_before_any_write(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    conn = sqlite3.connect(ledger._db_path())
    conn.execute("CREATE TABLE slack_intake_receipts (receipt_id TEXT PRIMARY KEY)")
    conn.execute(f"PRAGMA user_version={ledger._SCHEMA_VERSION}")
    conn.commit()
    before = conn.execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    conn.close()

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outcome.failure_reason == "schema_mismatch"
    check = sqlite3.connect(ledger._db_path())
    after = check.execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    check.close()
    assert after == before


@pytest.mark.asyncio
async def test_versioned_schema_rejects_named_index_with_wrong_columns(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger.record_listener_received(**_kwargs(event_id="Ev-existing"))
    conn = sqlite3.connect(ledger._db_path())
    conn.execute("DROP INDEX slack_intake_events_receipt_sequence")
    conn.execute(
        "CREATE INDEX slack_intake_events_receipt_sequence "
        "ON slack_intake_events(sequence, receipt_id)"
    )
    conn.commit()
    before = conn.execute(
        "PRAGMA index_info(slack_intake_events_receipt_sequence)"
    ).fetchall()
    conn.close()
    ledger._reset_initialization_for_tests()

    outcome = await ledger.record_listener_received_safely(
        **_kwargs(event_id="Ev-rejected")
    )

    assert outcome.persisted is False
    assert outcome.failure_reason == "schema_mismatch"
    check = sqlite3.connect(ledger._db_path())
    assert check.execute(
        "PRAGMA index_info(slack_intake_events_receipt_sequence)"
    ).fetchall() == before
    assert check.execute(
        "SELECT COUNT(*) FROM slack_intake_receipts"
    ).fetchone()[0] == 1
    check.close()


@pytest.mark.asyncio
async def test_schema_migration_is_atomic_when_ddl_fails(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    conn = sqlite3.connect(ledger._db_path())

    def deny_second_table(action, name, *_args):
        if action == sqlite3.SQLITE_CREATE_TABLE and name == "slack_message_projections":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(deny_second_table)
    with pytest.raises(sqlite3.DatabaseError):
        ledger._initialize_schema(conn)
    conn.close()

    check = sqlite3.connect(ledger._db_path())
    objects = check.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'slack_%' ORDER BY name"
    ).fetchall()
    version = check.execute("PRAGMA user_version").fetchone()[0]
    check.close()
    assert objects == []
    assert version == 0


@pytest.mark.asyncio
async def test_schema_migration_rechecks_version_after_two_connection_wait(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    first_locked = threading.Event()
    release_first = threading.Event()
    second_begin_attempted = threading.Event()
    second_create_count = 0
    errors = []

    class _MigrationConnection(_ConnectionProxy):
        def __init__(self, connection, owner):
            super().__init__(connection)
            self._owner = owner

        def execute(self, sql, parameters=()):
            nonlocal second_create_count
            normalized = " ".join(sql.split()).upper()
            if normalized == "BEGIN IMMEDIATE":
                if self._owner == "first":
                    result = super().execute(sql, parameters)
                    first_locked.set()
                    if not release_first.wait(2):
                        raise TimeoutError("first migration was not released")
                    return result
                second_begin_attempted.set()
            if self._owner == "second" and normalized.startswith("CREATE "):
                second_create_count += 1
            return super().execute(sql, parameters)

    first_conn = _MigrationConnection(
        sqlite3.connect(ledger._db_path(), timeout=2, check_same_thread=False),
        "first",
    )
    second_conn = _MigrationConnection(
        sqlite3.connect(ledger._db_path(), timeout=2, check_same_thread=False),
        "second",
    )
    for conn in (first_conn, second_conn):
        conn.execute("PRAGMA busy_timeout=2000")

    def migrate(conn):
        try:
            ledger._initialize_schema(conn)
        except Exception as exc:
            errors.append(exc)
        finally:
            conn.close()

    first = threading.Thread(target=migrate, args=(first_conn,))
    second = threading.Thread(target=migrate, args=(second_conn,))
    first.start()
    assert first_locked.wait(2)
    second.start()
    assert second_begin_attempted.wait(2)
    release_first.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert second_create_count == 0
    check = sqlite3.connect(ledger._db_path())
    assert check.execute("PRAGMA user_version").fetchone()[0] == ledger._SCHEMA_VERSION
    check.close()


@pytest.mark.asyncio
async def test_unrecognized_returned_journal_mode_fails_before_schema(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ledger,
        "apply_wal_with_fallback",
        lambda _conn, *, db_label: "memory",
        raising=False,
    )

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outcome.failure_reason == "persistence_failed"
    conn = sqlite3.connect(ledger._db_path())
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'slack_%'"
    ).fetchall() == []
    conn.close()


@pytest.mark.asyncio
async def test_returned_journal_mode_must_match_actual_mode(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)

    monkeypatch.setattr(
        ledger,
        "apply_wal_with_fallback",
        lambda _conn, *, db_label: "wal",
    )

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outcome.failure_reason == "persistence_failed"
    conn = sqlite3.connect(ledger._db_path())
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'slack_%'"
    ).fetchall() == []
    conn.close()


@pytest.mark.asyncio
async def test_vulnerable_sqlite_runtime_keeps_new_ledger_out_of_wal(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    import hermes_state_wal

    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda: True)

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is True
    conn = sqlite3.connect(ledger._db_path())
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    conn.close()


@pytest.mark.asyncio
async def test_wal_refusal_accepts_verified_delete_mode(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)

    def refuse_wal(conn, *, db_label):
        assert db_label == "Slack intake ledger"
        return conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower()

    monkeypatch.setattr(ledger, "apply_wal_with_fallback", refuse_wal)

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is True
    conn = sqlite3.connect(ledger._db_path())
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    conn.close()


@pytest.mark.asyncio
async def test_expensive_initialization_is_cached_by_database_identity(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    original_journal = ledger.apply_wal_with_fallback
    original_schema = ledger._initialize_schema
    calls = []

    def observe_journal(conn, *, db_label):
        calls.append("journal")
        return original_journal(conn, db_label=db_label)

    def observe_schema(conn):
        calls.append("schema")
        return original_schema(conn)

    monkeypatch.setattr(ledger, "apply_wal_with_fallback", observe_journal)
    monkeypatch.setattr(ledger, "_initialize_schema", observe_schema)

    ledger.record_listener_received(**_kwargs(event_id="Ev-cache-one"))
    ledger.record_listener_received(**_kwargs(event_id="Ev-cache-two"))
    ledger.read_receipts()
    assert calls == ["journal", "schema"]

    ledger._db_path().unlink()
    ledger.record_listener_received(**_kwargs(event_id="Ev-cache-recreated"))
    assert calls == ["journal", "schema", "journal", "schema"]


@pytest.mark.asyncio
async def test_real_sqlite_full_preserves_existing_receipt(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-full-existing"))
    original_open = ledger._open_connection

    def open_at_current_page_limit():
        conn = original_open()
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        conn.execute(f"PRAGMA max_page_count={pages}")
        return conn

    failure = None
    with monkeypatch.context() as fault:
        fault.setattr(ledger, "_open_connection", open_at_current_page_limit)
        for index in range(256):
            outcome = await ledger.record_listener_received_safely(
                **_kwargs(event_id=f"Ev-full-{index}")
            )
            if not outcome.persisted:
                failure = outcome
                break

    assert failure is not None
    assert failure.failure_reason == "disk_full"
    assert any(row["receipt_id"] == first.receipt_id for row in ledger.read_receipts())


@pytest.mark.asyncio
async def test_quota_failure_at_database_write_preserves_existing_receipt(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-quota-existing"))
    original_open = ledger._open_connection

    class _QuotaConnection(_ConnectionProxy):
        def execute(self, sql, parameters=()):
            if "INSERT INTO slack_intake_receipts" in sql:
                raise OSError(errno.EDQUOT, "synthetic quota boundary")
            return super().execute(sql, parameters)

    with monkeypatch.context() as fault:
        fault.setattr(ledger, "_open_connection", lambda: _QuotaConnection(original_open()))
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id="Ev-quota-rejected")
        )

    assert outcome.persisted is False
    assert outcome.failure_reason == "quota_exceeded"
    assert [row["receipt_id"] for row in ledger.read_receipts()] == [first.receipt_id]


@pytest.mark.asyncio
async def test_os_disk_full_at_database_write_preserves_existing_receipt(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-disk-existing"))
    original_open = ledger._open_connection

    class _DiskFullConnection(_ConnectionProxy):
        def execute(self, sql, parameters=()):
            if "INSERT INTO slack_intake_receipts" in sql:
                raise OSError(errno.ENOSPC, "synthetic disk-full boundary")
            return super().execute(sql, parameters)

    with monkeypatch.context() as fault:
        fault.setattr(
            ledger,
            "_open_connection",
            lambda: _DiskFullConnection(original_open()),
        )
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id="Ev-disk-rejected")
        )

    assert outcome.persisted is False
    assert outcome.failure_reason == "disk_full"
    assert [row["receipt_id"] for row in ledger.read_receipts()] == [first.receipt_id]


@pytest.mark.asyncio
async def test_sqlite_authorizer_denial_preserves_existing_receipt_and_raw_ids(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-auth-existing"))
    original_open = ledger._open_connection

    def open_with_insert_denied():
        conn = original_open()

        def deny_insert(action, *_args):
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_INSERT
                else sqlite3.SQLITE_OK
            )

        conn.set_authorizer(deny_insert)
        return conn

    rejected_id = "Ev-authorizer-private"
    with monkeypatch.context() as fault:
        fault.setattr(ledger, "_open_connection", open_with_insert_denied)
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id=rejected_id)
        )

    assert outcome.persisted is False
    assert outcome.failure_reason == "persistence_failed"
    assert [row["receipt_id"] for row in ledger.read_receipts()] == [first.receipt_id]
    stored = b"".join(path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    assert rejected_id.encode() not in stored


@pytest.mark.asyncio
async def test_failed_commit_preserves_existing_receipt(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-commit-existing"))
    original_open = ledger._open_connection

    class _FailedCommitConnection(_ConnectionProxy):
        def __exit__(self, exc_type, exc, traceback):
            if exc_type is None:
                self._connection.rollback()
                raise sqlite3.OperationalError("synthetic failed commit")
            return self._connection.__exit__(exc_type, exc, traceback)

    with monkeypatch.context() as fault:
        fault.setattr(
            ledger,
            "_open_connection",
            lambda: _FailedCommitConnection(original_open()),
        )
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id="Ev-commit-rejected")
        )

    assert outcome.persisted is False
    assert outcome.failure_reason == "persistence_failed"
    assert [row["receipt_id"] for row in ledger.read_receipts()] == [first.receipt_id]


@pytest.mark.asyncio
async def test_real_busy_deadline_preserves_existing_receipt(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-busy-existing"))
    blocker = ledger._open_connection()
    blocker.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id="Ev-busy-rejected")
        )
    finally:
        blocker.rollback()
        blocker.close()

    assert time.monotonic() - started >= ledger._BUSY_TIMEOUT_MS / 1000
    assert outcome.persisted is False
    assert outcome.failure_reason == "busy_deadline_exceeded"
    assert [row["receipt_id"] for row in ledger.read_receipts()] == [first.receipt_id]


@pytest.mark.asyncio
async def test_corrupt_database_degrades_without_exposing_payload(
    monkeypatch, tmp_path, caplog
):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger._db_path().write_bytes(b"not sqlite: Ev-private C-private")

    with caplog.at_level(logging.CRITICAL):
        outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outcome.failure_reason == "database_corrupt"
    assert "Ev-event-private" not in caplog.text
    assert str(tmp_path) not in caplog.text


@pytest.mark.macos_only
async def test_database_symlink_is_rejected_before_sqlite_can_follow_it(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.sqlite3"
    sqlite3.connect(outside).close()
    before = outside.read_bytes()
    ledger._db_path().symlink_to(outside)

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is False
    assert outside.read_bytes() == before


async def test_capacity_is_bounded_by_receipt_count(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(ledger, "_MAX_RECEIPTS", 1)
    ledger.record_listener_received(**_kwargs(event_id="Ev-one"))

    with pytest.raises(ledger.IntakeLedgerCapacityError):
        ledger.record_listener_received(**_kwargs(event_id="Ev-two"))

    assert len(ledger.read_receipts()) == 1


async def test_capacity_is_bounded_by_allocated_database_bytes(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    first = ledger.record_listener_received(**_kwargs(event_id="Ev-size-one"))
    assert ledger._db_path().stat().st_size > 1
    monkeypatch.setattr(ledger, "_MAX_DATABASE_BYTES", 1, raising=False)

    outcome = await ledger.record_listener_received_safely(
        **_kwargs(event_id="Ev-size-two")
    )

    assert outcome.persisted is False
    assert outcome.failure_reason == "capacity_exceeded"
    rows = ledger.read_receipts()
    assert [row["receipt_id"] for row in rows] == [first.receipt_id]


async def test_pruned_freelist_pages_do_not_permanently_wedge_capacity(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    for index in range(64):
        receipt = ledger.record_listener_received(
            **_kwargs(event_id=f"Ev-prunable-{index}", received_at=1.0)
        )
        ledger.mark_accepted(receipt.receipt_id, decided_at=1.0)
    physical_before = ledger._db_path().stat().st_size
    monkeypatch.setattr(ledger, "_MAX_DATABASE_BYTES", physical_before)

    outcome = await ledger.record_listener_received_safely(
        **_kwargs(
            event_id="Ev-after-prune",
            received_at=ledger._RETENTION_SECONDS + 2.0,
        )
    )

    assert outcome.persisted is True
    assert ledger._db_path().stat().st_size >= physical_before
    assert len(ledger.read_receipts()) == 1


async def test_stale_open_receipt_is_dead_lettered_before_new_admission(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    old = ledger.record_listener_received(**_kwargs(event_id="Ev-old", received_at=1.0))
    for index in range(ledger._MAX_EVENTS_PER_RECEIPT):
        ledger.append_stage(
            old.receipt_id,
            stage="acknowledged",
            observed_at=2.0 + index,
        )
    ledger.record_listener_received(
        **_kwargs(event_id="Ev-new", received_at=ledger._MAX_OPEN_AGE_SECONDS + 2.0)
    )

    row = next(item for item in ledger.read_receipts() if item["receipt_id"] == old.receipt_id)
    assert row["terminal_state"] == "dead_lettered"
    assert row["terminal_reason"] == "stale_open"
    assert len(row["events"]) <= ledger._MAX_EVENTS_PER_RECEIPT


async def test_concurrent_duplicate_writers_share_one_receipt(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    barrier = threading.Barrier(3)
    outcomes = []

    def write():
        barrier.wait()
        outcomes.append(ledger.record_listener_received(**_kwargs()))

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)

    assert not any(thread.is_alive() for thread in threads)
    assert len({item.receipt_id for item in outcomes}) == 1
    assert ledger.read_receipts()[0]["receive_count"] == 2


async def test_begin_immediate_serializes_two_connections_before_identity_read(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger._open_connection().close()
    first_locked = threading.Event()
    release_first = threading.Event()
    second_begin_attempted = threading.Event()
    second_selected = threading.Event()
    errors = []
    outcomes = []

    class _NoProcessLock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _ObservedConnection:
        def __init__(self, connection, owner):
            self._connection = connection
            self._owner = owner

        def execute(self, sql, parameters=()):
            normalized = " ".join(sql.split()).upper()
            if normalized == "BEGIN IMMEDIATE":
                if self._owner == "first":
                    result = self._connection.execute(sql, parameters)
                    first_locked.set()
                    if not release_first.wait(2):
                        raise TimeoutError("first writer was not released")
                    return result
                second_begin_attempted.set()
            elif self._owner == "second" and normalized.startswith(
                "SELECT WORKSPACE_HASH"
            ):
                second_selected.set()
            return self._connection.execute(sql, parameters)

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, *args):
            return self._connection.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def open_observed_connection():
        connection = sqlite3.connect(ledger._db_path(), timeout=2)
        connection.execute("PRAGMA busy_timeout=2000")
        return _ObservedConnection(connection, threading.current_thread().name)

    def write():
        try:
            outcomes.append(ledger.record_listener_received(**_kwargs()))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(ledger, "_DB_LOCK", _NoProcessLock())
    monkeypatch.setattr(ledger, "_open_connection", open_observed_connection)
    first = threading.Thread(target=write, name="first")
    second = threading.Thread(target=write, name="second")
    first.start()
    assert first_locked.wait(2)
    second.start()
    assert second_begin_attempted.wait(2)
    assert not second_selected.is_set()
    release_first.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len({item.receipt_id for item in outcomes}) == 1
    assert ledger.read_receipts()[0]["receive_count"] == 2


@pytest.mark.asyncio
async def test_safe_async_write_stays_inside_context_local_profile(monkeypatch, tmp_path):
    from gateway import slack_intake_ledger as ledger
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profiles" / "client"
    launch_home.mkdir()
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()

    token = set_hermes_home_override(profile_home)
    try:
        outcome = await ledger.record_listener_received_safely(**_kwargs())
    finally:
        reset_hermes_home_override(token)

    assert outcome.persisted is True
    assert (profile_home / "runtime" / "slack-intake" / "ledger.sqlite3").is_file()
    assert not (launch_home / "runtime" / "slack-intake" / "ledger.sqlite3").exists()


@pytest.mark.asyncio
async def test_safe_async_write_does_not_require_descriptor_relative_filesystem_apis(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    original_open = ledger.os.open

    def reject_descriptor_relative_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None:
            raise NotImplementedError("descriptor-relative open unavailable")
        return original_open(path, flags, mode)

    monkeypatch.setattr(ledger.os, "open", reject_descriptor_relative_open)

    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is True
    assert ledger._db_path().is_file()


@pytest.mark.asyncio
async def test_safe_write_runs_outside_event_loop_thread(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    event_loop_thread = threading.get_ident()
    worker_threads = []
    original = ledger.record_listener_received

    def observe_thread(**kwargs):
        worker_threads.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(ledger, "record_listener_received", observe_thread)
    outcome = await ledger.record_listener_received_safely(**_kwargs())

    assert outcome.persisted is True
    assert worker_threads and worker_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_safe_boundary_preserves_caller_cancellation(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    cancellation = asyncio.CancelledError("original cancellation")

    async def cancel(*_args, **_kwargs):
        raise cancellation

    monkeypatch.setattr(ledger, "_run_ledger_work", cancel)

    with pytest.raises(asyncio.CancelledError) as caught:
        await ledger.record_listener_received_safely(**_kwargs())
    assert caught.value is cancellation
    caught.value.__traceback__ = None
    del caught, cancellation
    assert ledger.read_fallback_receipts() == []


@pytest.mark.asyncio
async def test_persistence_failure_uses_bounded_metadata_only_fallback(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(ledger, "_MAX_FALLBACK_RECEIPTS", 2)

    async def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("secret database path")

    monkeypatch.setattr(ledger, "_run_ledger_work", fail)
    for index in range(3):
        outcome = await ledger.record_listener_received_safely(
            **_kwargs(event_id=f"Ev-private-{index}")
        )
        assert outcome.persisted is False

    fallback = ledger.read_fallback_receipts()
    assert len(fallback) == 2
    assert ledger.fallback_eviction_count() == 1
    assert ledger.persistence_degraded() is True
    assert "Ev-private" not in repr(fallback)


@pytest.mark.asyncio
async def test_http_intake_contract_fails_before_success_when_receipt_is_not_durable(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)

    async def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("PRIVATE-HTTP-PERSISTENCE-DETAIL")

    monkeypatch.setattr(ledger, "_run_ledger_work", fail)

    with pytest.raises(ledger.IntakePersistenceRequired) as caught:
        await ledger.record_http_envelope(**_kwargs())

    assert caught.value.failure_reason == "persistence_failed"
    assert str(caught.value) == "Slack intake receipt is not durable (persistence_failed)"
    assert "PRIVATE-HTTP-PERSISTENCE-DETAIL" not in str(caught.value)
    assert ledger.persistence_degraded()
    assert ledger.read_fallback_receipts() == []


@pytest.mark.asyncio
async def test_http_intake_contract_returns_only_after_receipt_is_readable(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)

    observation = await ledger.record_http_envelope(**_kwargs())

    assert observation.receipt_id in {
        row["receipt_id"] for row in ledger.read_receipts()
    }


@pytest.mark.asyncio
async def test_http_intake_contract_preserves_cancellation_without_fallback(
    monkeypatch, tmp_path
):
    ledger = _ledger(monkeypatch, tmp_path)
    original = asyncio.CancelledError("synthetic HTTP cancellation")

    async def cancel(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(ledger, "_run_ledger_work", cancel)

    with pytest.raises(asyncio.CancelledError) as caught:
        await ledger.record_http_envelope(**_kwargs())

    assert caught.value is original
    assert ledger.read_fallback_receipts() == []


@pytest.mark.parametrize("value", [[], {}])
@pytest.mark.parametrize(
    "field",
    [
        "safe_event_type",
        "safe_stage",
        "unavailable_envelope_event_type",
        "unavailable_stage",
        "unavailable_reason",
    ],
)
async def test_unavailable_fallback_normalizes_unhashable_labels(
    monkeypatch, tmp_path, field, value
):
    ledger = _ledger(monkeypatch, tmp_path)
    if field == "unavailable_envelope_event_type":
        ledger.record_unavailable_envelope(**_kwargs(event_type=value))
    elif field.startswith("safe_"):
        kwargs = _kwargs()
        kwargs[field.removeprefix("safe_")] = value
        outcome = await ledger.record_listener_received_safely(**kwargs)
        assert not outcome.persisted
        assert outcome.failure_reason == "invalid_metadata"
    else:
        kwargs = {
            "stage": "listener_entered",
            "observed_at": 1_700_000_000.0,
            "reason": "ignored",
        }
        kwargs[field.removeprefix("unavailable_")] = value
        ledger.record_unavailable_stage("a" * 64, **kwargs)

    fallback = ledger.read_fallback_receipts()
    assert len(fallback) == 1
    expected_stage = {
        "safe_event_type": "listener_received",
        "safe_stage": "unknown",
        "unavailable_envelope_event_type": "envelope_received",
        "unavailable_stage": "unknown",
        "unavailable_reason": "listener_entered",
    }[field]
    assert fallback[0]["stage"] == expected_stage
    if "reason" in fallback[0]:
        expected_reason = None if field == "unavailable_reason" else "ignored"
        assert fallback[0]["reason"] == expected_reason
    if "event_type" in fallback[0]:
        expected_event_type = (
            "unknown"
            if field in {"safe_event_type", "unavailable_envelope_event_type"}
            else "message"
        )
        assert fallback[0]["event_type"] == expected_event_type
    assert ledger.persistence_degraded()


@pytest.mark.parametrize("value", [[], {}])
@pytest.mark.parametrize(
    "field", ["progress_stage", "progress_reason", "terminal_reason"]
)
async def test_safe_progress_and_terminal_fallbacks_normalize_unhashable_labels(
    monkeypatch, tmp_path, field, value
):
    ledger = _ledger(monkeypatch, tmp_path)
    receipt = ledger.record_listener_received(**_kwargs())
    before = ledger.read_receipts()

    if field.startswith("progress_"):
        kwargs = {
            "stage": "listener_entered",
            "observed_at": 1_700_000_001.0,
            "reason": "ignored",
        }
        kwargs[field.removeprefix("progress_")] = value
        outcome = await ledger.append_stage_safely(receipt.receipt_id, **kwargs)
    else:
        outcome = await ledger.mark_dropped_safely(
            receipt.receipt_id,
            reason=value,
            decided_at=1_700_000_001.0,
        )

    assert not outcome.persisted
    assert outcome.failure_reason == "invalid_metadata"
    assert ledger.read_receipts() == before
    fallback = ledger.read_fallback_receipts()
    assert len(fallback) == 1
    assert fallback[0]["stage"] == {
        "progress_stage": "unknown",
        "progress_reason": "listener_entered",
        "terminal_reason": "dropped",
    }[field]
    assert fallback[0]["reason"] == (
        "ignored" if field == "progress_stage" else None
    )
    assert ledger.persistence_degraded()


@pytest.mark.asyncio
async def test_success_after_outage_clears_degraded_health(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    original = ledger._run_ledger_work

    async def fail(*_args, **_kwargs):
        raise OSError("synthetic outage")

    monkeypatch.setattr(ledger, "_run_ledger_work", fail)
    failed = await ledger.record_listener_received_safely(**_kwargs(event_id="Ev-failed"))
    assert failed.persisted is False
    assert ledger.persistence_degraded() is True

    monkeypatch.setattr(ledger, "_run_ledger_work", original)
    recovered = await ledger.record_listener_received_safely(**_kwargs(event_id="Ev-recovered"))
    assert recovered.persisted is True
    assert ledger.persistence_degraded() is False


@pytest.mark.asyncio
async def test_safe_progress_and_terminal_unknown_receipt_are_total(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    missing = "a" * 64

    progress = await ledger.append_stage_safely(missing, stage="listener_entered")
    terminal = await ledger.mark_dropped_safely(
        missing, reason="invalid_event", decided_at=1.0
    )

    assert progress.persisted is terminal.persisted is False
    assert progress.failure_reason == terminal.failure_reason == "unknown_receipt"


async def test_read_receipts_returns_copies(monkeypatch, tmp_path):
    ledger = _ledger(monkeypatch, tmp_path)
    ledger.record_listener_received(**_kwargs())
    first = ledger.read_receipts()
    first[0]["terminal_state"] = "tampered"
    first[0]["events"].clear()

    second = ledger.read_receipts()
    assert second[0]["terminal_state"] is None
    assert second[0]["events"]
