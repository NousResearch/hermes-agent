from __future__ import annotations

import errno
import json
import logging
import multiprocessing as mp
import os
import sqlite3
from pathlib import Path

import pytest

import hermes_state
import session_fallback_spool as spool
from hermes_state import SessionDB, SessionDBBatchMessage


@pytest.fixture()
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_db = SessionDB(db_path=home / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _bootstrap(
    *,
    session_id: str = "replay-session",
    parent_session_id: str | None = None,
) -> spool.SessionSpoolBootstrap:
    return spool.SessionSpoolBootstrap(
        session_id=session_id,
        source="cli",
        started_at=123.456,
        model="gpt-test",
        model_config={"mode": "replay"},
        system_prompt="persist me exactly",
        parent_session_id=parent_session_id,
        cwd="/tmp/project",
        profile_name="profile-a",
        user_id="user-1",
        session_key="session-key",
        chat_id="chat-1",
        chat_type="group",
        thread_id="thread-1",
    )


def _batch_messages(
    unit_id: str = "unit-1",
    *,
    content: str = "hello replay",
) -> list[SessionDBBatchMessage]:
    return [
        SessionDBBatchMessage(
            persistence_unit_id=unit_id,
            persistence_message_key=f"{unit_id}-key-0",
            persistence_ordinal=0,
            role="user",
            content=content,
            timestamp=100.0,
        )
    ]


def _record(
    unit_id: str = "unit-1",
    *,
    session_id: str = "replay-session",
    content: str = "hello replay",
) -> spool.SessionSpoolRecord:
    return spool.SessionSpoolRecord(
        bootstrap=_bootstrap(session_id=session_id),
        persist_attempt_id="a" * 32,
        persist_attempt_unit_index=0,
        canonical_failure={
            "stage": "append_messages_batch",
            "error_class": "RuntimeError",
            "error_message": "db down",
            "session_row_created": True,
        },
        batch_messages=tuple(_batch_messages(unit_id=unit_id, content=content)),
    )


def _write_sealed_segment(home, sequence: int, *records: spool.SessionSpoolRecord):
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)
    segment_path = sealed / f"{sequence:020d}.spool"
    segment_path.write_bytes(b"".join(spool._frame_bytes_for_record(record) for record in records))
    return segment_path


def _close_runtime_fds(runtime) -> None:
    spool._close_fd_quietly(runtime.lock_fd)
    spool._close_fd_quietly(runtime.root_fd)
    spool._close_fd_quietly(runtime.home_fd)


def _hold_replay_owner_lock(home_path: str, ready_conn, release_conn) -> None:
    os.environ["HERMES_HOME"] = home_path
    runtime = spool._open_locked_runtime()
    owner = None
    try:
        owner = spool._try_acquire_replay_owner(runtime)
        ready_conn.send(owner is not None)
        release_conn.recv()
    finally:
        if owner is not None:
            spool._close_fd_quietly(owner.fd)
        ready_conn.close()
        release_conn.close()
        _close_runtime_fds(runtime)


def _replay_owner_context():
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _backfill_record(
    unit_id: str,
    *,
    content: str,
    parent_session_id: str,
) -> spool.SessionSpoolRecord:
    return spool.SessionSpoolRecord(
        bootstrap=_bootstrap(parent_session_id=parent_session_id),
        persist_attempt_id="b" * 32,
        persist_attempt_unit_index=0,
        canonical_failure={
            "stage": "append_messages_batch",
            "error_class": "RuntimeError",
            "error_message": "db down",
            "session_row_created": True,
        },
        batch_messages=tuple(_batch_messages(unit_id=unit_id, content=content)),
    )


def _fts_message_count(db) -> int:
    with db._lock:
        return int(db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])


def _replay_messages_for_state(caplog, state: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and f"state={state}" in record.getMessage()
    ]


def _assert_existing_row_backfilled(session, *, parent_session_id: str) -> None:
    assert session["source"] == "cli"
    assert session["parent_session_id"] == parent_session_id
    assert session["profile_name"] == "profile-a"
    assert session["user_id"] == "user-1"
    assert session["session_key"] == "session-key"
    assert session["chat_id"] == "chat-1"
    assert session["chat_type"] == "group"
    assert session["thread_id"] == "thread-1"


def test_existing_row_bootstrap_backfill_manual_replay_fills_null_fields_and_duplicate_safe(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db.create_session("real-parent", "cli")
    db.create_session(
        "replay-session",
        "cli",
        chat_type="group",
        thread_id="thread-1",
    )
    record = _backfill_record(
        "existing-row-bootstrap-backfill-manual",
        content="backfill manual transcript",
        parent_session_id="real-parent",
    )
    _write_sealed_segment(home, 1, record)

    result = spool.replay_to_session_db(db, trigger="manual")
    session = db.get_session("replay-session")

    assert result.state is spool.ReplayRunState.REPLAYED
    assert session is not None
    _assert_existing_row_backfilled(session, parent_session_id="real-parent")
    assert session["message_count"] == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "backfill manual transcript"
    ]
    assert _fts_message_count(db) == 1

    duplicate = db.reconcile_bootstrap_and_append_messages_batch(
        record.bootstrap,
        record.batch_messages,
        replay_patience_s=2.0,
    )

    session_after = db.get_session("replay-session")
    assert duplicate.inserted_count == 0
    assert duplicate.duplicate_count == 1
    assert session_after is not None
    _assert_existing_row_backfilled(session_after, parent_session_id="real-parent")
    assert session_after["message_count"] == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "backfill manual transcript"
    ]
    assert _fts_message_count(db) == 1


def test_existing_row_bootstrap_backfill_startup_replay_fills_null_fields_after_restart(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    setup_db = SessionDB(db_path=home / "state.db")
    try:
        setup_db.create_session("real-parent", "cli")
        setup_db.create_session("replay-session", "cli")
    finally:
        setup_db.close()

    record = _backfill_record(
        "existing-row-bootstrap-backfill-startup",
        content="backfill startup transcript",
        parent_session_id="real-parent",
    )
    _write_sealed_segment(home, 1, record)
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)

    startup_db = SessionDB(db_path=home / "state.db")
    try:
        session = startup_db.get_session("replay-session")
        assert session is not None
        _assert_existing_row_backfilled(session, parent_session_id="real-parent")
        assert session["message_count"] == 1
        assert [row["content"] for row in startup_db.get_messages("replay-session")] == [
            "backfill startup transcript"
        ]
        assert _fts_message_count(startup_db) == 1
    finally:
        startup_db.close()

    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)
    restarted = SessionDB(db_path=home / "state.db")
    try:
        session = restarted.get_session("replay-session")
        assert session is not None
        _assert_existing_row_backfilled(session, parent_session_id="real-parent")
        assert session["message_count"] == 1
        assert [row["content"] for row in restarted.get_messages("replay-session")] == [
            "backfill startup transcript"
        ]
        assert _fts_message_count(restarted) == 1
    finally:
        restarted.close()


def test_existing_row_bootstrap_backfill_conflict_rolls_back_metadata_messages_counters_and_fts(
    db,
):
    db.create_session("real-parent", "cli")
    db.create_session(
        "replay-session",
        "cli",
        thread_id="wrong-thread",
    )
    record = _backfill_record(
        "existing-row-bootstrap-backfill-conflict",
        content="backfill conflict transcript",
        parent_session_id="real-parent",
    )

    with pytest.raises(hermes_state.AppendMessagesBatchConflictError, match="thread_id"):
        db.reconcile_bootstrap_and_append_messages_batch(
            record.bootstrap,
            record.batch_messages,
            replay_patience_s=2.0,
        )

    session = db.get_session("replay-session")
    assert session is not None
    assert session["source"] == "cli"
    assert session["thread_id"] == "wrong-thread"
    assert session["parent_session_id"] is None
    assert session["profile_name"] is None
    assert session["user_id"] is None
    assert session["session_key"] is None
    assert session["chat_id"] is None
    assert session["chat_type"] is None
    assert session["message_count"] == 0
    assert db.get_messages("replay-session") == []
    assert _fts_message_count(db) == 0


def test_reconcile_bootstrap_and_append_messages_batch_creates_missing_session_row(db):
    bootstrap = _bootstrap()

    result = db.reconcile_bootstrap_and_append_messages_batch(
        bootstrap,
        _batch_messages(),
        replay_patience_s=2.0,
    )

    assert result.inserted_count == 1
    assert result.duplicate_count == 0
    session = db.get_session("replay-session")
    assert session is not None
    assert session["source"] == "cli"
    assert session["model"] == "gpt-test"
    assert session["system_prompt"] == "persist me exactly"
    assert session["parent_session_id"] is None
    assert session["cwd"] == "/tmp/project"
    assert session["profile_name"] == "profile-a"
    assert session["user_id"] == "user-1"
    assert session["session_key"] == "session-key"
    assert session["chat_id"] == "chat-1"
    assert session["chat_type"] == "group"
    assert session["thread_id"] == "thread-1"
    assert [row["content"] for row in db.get_messages("replay-session")] == ["hello replay"]


def test_reconcile_bootstrap_and_append_messages_batch_rejects_missing_parent(db):
    bootstrap = _bootstrap(parent_session_id="missing-parent")

    with pytest.raises(hermes_state.AppendMessagesBatchConflictError, match="parent"):
        db.reconcile_bootstrap_and_append_messages_batch(
            bootstrap,
            _batch_messages(),
            replay_patience_s=2.0,
        )

    assert db.get_session("replay-session") is None
    assert db.get_messages("replay-session") == []


def test_missing_parent_manual_replay_blocks_and_preserves_later_fifo_head(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(
        home,
        1,
        spool.SessionSpoolRecord(
            bootstrap=_bootstrap(parent_session_id="missing-parent"),
            persist_attempt_id="b" * 32,
            persist_attempt_unit_index=0,
            canonical_failure={
                "stage": "append_messages_batch",
                "error_class": "RuntimeError",
                "error_message": "db down",
                "session_row_created": True,
            },
            batch_messages=tuple(_batch_messages(unit_id="unit-missing", content="alpha")),
        ),
    )
    later = _write_sealed_segment(home, 2, _record(unit_id="unit-later", content="later"))

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert result.first_blocked_segment == 1
    assert result.first_blocked_offset == 0
    assert result.error_class == "AppendMessagesBatchConflictError"
    assert db.get_messages("replay-session") == []
    assert later.exists()


def test_missing_parent_startup_replay_opens_and_preserves_sealed_head(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)
    head = _write_sealed_segment(
        home,
        1,
        spool.SessionSpoolRecord(
            bootstrap=_bootstrap(parent_session_id="missing-parent"),
            persist_attempt_id="c" * 32,
            persist_attempt_unit_index=0,
            canonical_failure={
                "stage": "append_messages_batch",
                "error_class": "RuntimeError",
                "error_message": "db down",
                "session_row_created": True,
            },
            batch_messages=tuple(_batch_messages(unit_id="unit-startup", content="alpha")),
        ),
    )

    startup_db = SessionDB(db_path=home / "state.db")
    try:
        assert startup_db.get_messages("replay-session") == []
        assert head.exists()
    finally:
        startup_db.close()


def test_compression_busy_returns_retry_pending_and_preserves_later_fifo(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))
    later = _write_sealed_segment(home, 2, _record(unit_id="unit-b", content="later"))

    def _busy(*_args, **_kwargs):
        raise hermes_state.CompressionSessionBusyError("busy")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _busy)

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class is not None
    assert result.cooldown_seconds > 0
    assert result.frames_acked == 0
    assert db.get_messages("replay-session") == []
    assert later.exists()


def test_compression_closed_returns_blocked_integrity(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))

    def _closed(*_args, **_kwargs):
        raise hermes_state.CompressionSessionClosedError("replay-session")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _closed)

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert result.first_blocked_segment == 1
    assert result.first_blocked_offset == 0
    assert result.error_class == "CompressionSessionClosedError"


def test_sqlite_locked_returns_retry_pending(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))

    def _locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _locked)

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class is not None
    assert result.cooldown_seconds > 0


def test_sqlite_busy_unrelated_operational_error_propagates(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))

    def _busy(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _busy)
    busy_result = spool.replay_to_session_db(db, trigger="manual")
    assert busy_result.state is spool.ReplayRunState.RETRY_PENDING
    assert busy_result.retry_class is not None

    other_home = tmp_path / ".hermes-other"
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    _write_sealed_segment(other_home, 1, _record(unit_id="unit-b", content="beta"))

    def _other(*_args, **_kwargs):
        raise sqlite3.OperationalError("syntax error")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _other)
    with pytest.raises(sqlite3.OperationalError, match="syntax error"):
        spool.replay_to_session_db(db, trigger="manual")


def test_replay_to_session_db_replays_clean_segment_and_compacts_it(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    segment_path = _write_sealed_segment(home, 1, _record())

    result = spool.replay_to_session_db(db, trigger="startup")

    assert result.state is spool.ReplayRunState.REPLAYED
    assert result.frames_committed == 1
    assert result.frames_duplicated == 0
    assert [row["content"] for row in db.get_messages("replay-session")] == ["hello replay"]
    assert not segment_path.exists()


def test_sessiondb_startup_replay_runs_once_only_for_writable_canonical_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)
    calls = []

    def _fake_replay(session_db, *, trigger):
        calls.append((session_db.db_path, trigger))
        return spool.ReplayRunResult(state=spool.ReplayRunState.EMPTY, trigger=trigger)

    monkeypatch.setattr(spool, "replay_to_session_db", _fake_replay)

    canonical = SessionDB(db_path=home / "state.db")
    canonical.close()
    second = SessionDB(db_path=home / "state.db")
    second.close()
    other = SessionDB(db_path=home / "other.db")
    other.close()
    readonly = SessionDB(db_path=home / "state.db", read_only=True)
    readonly.close()

    assert calls == [(home / "state.db", "startup")]


def test_corrupt_active_public_replay_publishes_evidence_and_blocker(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    spool.append_records((_record(unit_id="corrupt-active-direct"),))

    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    corrupted = bytearray(active_path.read_bytes())
    corrupted[0] = 0
    active_path.write_bytes(bytes(corrupted))

    result = spool.replay_to_session_db(db, trigger="startup")

    blockers = sorted((home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME).glob("*.blocker.json"))
    quarantine_spools = sorted((home / spool.SPOOL_ROOT_NAME / spool.QUARANTINE_DIR_NAME).glob("*.spool"))
    quarantine_sidecars = sorted((home / spool.SPOOL_ROOT_NAME / spool.QUARANTINE_DIR_NAME).glob("*.json"))

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert result.first_blocked_segment == 1
    assert len(blockers) == 1
    assert len(quarantine_spools) == 1
    assert len(quarantine_sidecars) == 1
    assert active_path.read_bytes() == b""


def test_corrupt_active_startup_replay_publishes_evidence_and_keeps_messages_empty(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)

    spool.append_records((_record(unit_id="corrupt-active-startup"),))

    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    corrupted = bytearray(active_path.read_bytes())
    corrupted[0] = 0
    active_path.write_bytes(bytes(corrupted))

    startup_db = SessionDB(db_path=home / "state.db")
    try:
        blockers = sorted((home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME).glob("*.blocker.json"))
        quarantine_spools = sorted((home / spool.SPOOL_ROOT_NAME / spool.QUARANTINE_DIR_NAME).glob("*.spool"))
        quarantine_sidecars = sorted((home / spool.SPOOL_ROOT_NAME / spool.QUARANTINE_DIR_NAME).glob("*.json"))

        assert startup_db.get_messages("replay-session") == []
        assert len(blockers) == 1
        assert len(quarantine_spools) == 1
        assert len(quarantine_sidecars) == 1
        assert active_path.read_bytes() == b""
    finally:
        startup_db.close()


def _assert_metadata_only_replay_evidence(
    home,
    *,
    sequence: int,
    expected_source_kind: str,
    expected_tail_status: str,
    expected_valid_prefix_bytes: int,
    expected_original_size_bytes: int,
):
    quarantine = home / spool.SPOOL_ROOT_NAME / spool.QUARANTINE_DIR_NAME
    evidence_spools = sorted(
        quarantine.glob(f"seq-{sequence:020d}-{expected_tail_status}-vp{expected_valid_prefix_bytes}.spool")
    )
    evidence_sidecars = sorted(
        quarantine.glob(f"seq-{sequence:020d}-{expected_tail_status}-vp{expected_valid_prefix_bytes}.json")
    )
    assert len(evidence_spools) == 1
    assert len(evidence_sidecars) == 1

    payload = json.loads(evidence_sidecars[0].read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "schema_version",
        "segment_sequence",
        "source_kind",
        "tail_status",
        "valid_prefix_bytes",
        "original_size_bytes",
        "evidence_spool_name",
    }
    assert payload == {
        "schema_version": 1,
        "segment_sequence": f"{sequence:020d}",
        "source_kind": expected_source_kind,
        "tail_status": expected_tail_status,
        "valid_prefix_bytes": expected_valid_prefix_bytes,
        "original_size_bytes": expected_original_size_bytes,
        "evidence_spool_name": evidence_spools[0].name,
    }
    return evidence_spools[0], evidence_sidecars[0]


def _write_blocker_backed_prefix_state(
    home,
    *,
    sequence: int,
    source_kind: str,
    prefix_bytes: bytes,
    original_bytes: bytes,
    tail_status: str,
):
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    blockers = sealed / spool.BLOCKERS_DIR_NAME
    quarantine = root / spool.QUARANTINE_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)
    blockers.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)

    prefix_name = f"{sequence:020d}.prefix.spool"
    (sealed / prefix_name).write_bytes(prefix_bytes)
    evidence_base = f"seq-{sequence:020d}-{tail_status}-vp{len(prefix_bytes)}"
    evidence_spool_name = f"{evidence_base}.spool"
    evidence_sidecar_name = f"{evidence_base}.json"
    (quarantine / evidence_spool_name).write_bytes(original_bytes)
    (quarantine / evidence_sidecar_name).write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": f"{sequence:020d}",
                "source_kind": source_kind,
                "tail_status": tail_status,
                "valid_prefix_bytes": len(prefix_bytes),
                "original_size_bytes": len(original_bytes),
                "evidence_spool_name": evidence_spool_name,
            }
        )
    )
    (blockers / f"{sequence:020d}.blocker.json").write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": f"{sequence:020d}",
                "source_kind": source_kind,
                "tail_status": tail_status,
                "valid_prefix_bytes": len(prefix_bytes),
                "acked_prefix_bytes": 0,
                "blocking_offset": len(prefix_bytes),
                "prefix_segment_name": prefix_name,
                "evidence_spool_name": evidence_spool_name,
                "evidence_sidecar_name": evidence_sidecar_name,
                "original_size_bytes": len(original_bytes),
            }
        )
    )
    return sealed / prefix_name


def _build_blocker_crash_state(home, case_name: str):
    first = _record(unit_id=f"{case_name}-a", content="alpha")
    second = _record(unit_id=f"{case_name}-b", content="beta")
    clean_frame = spool._frame_bytes_for_record(first)
    corrupt_frame = bytearray(spool._frame_bytes_for_record(second))
    corrupt_frame[-1] ^= 0x01
    prefix_path = _write_blocker_backed_prefix_state(
        home,
        sequence=1,
        source_kind="sealed",
        prefix_bytes=clean_frame,
        original_bytes=clean_frame + bytes(corrupt_frame),
        tail_status="checksum_mismatch",
    )
    _write_sealed_segment(home, 2, _record(unit_id=f"{case_name}-later", content="gamma"))
    root = home / spool.SPOOL_ROOT_NAME
    quarantine = root / spool.QUARANTINE_DIR_NAME
    evidence_spool = quarantine / f"seq-{1:020d}-checksum_mismatch-vp{len(clean_frame)}.spool"
    evidence_sidecar = quarantine / f"seq-{1:020d}-checksum_mismatch-vp{len(clean_frame)}.json"
    blocker_path = root / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME / f"{1:020d}.blocker.json"

    if case_name == "missing_evidence_sidecar":
        evidence_sidecar.unlink()
        expected_error_class = "missing_replay_evidence_sidecar"
    elif case_name == "missing_evidence_spool":
        evidence_spool.unlink()
        expected_error_class = "missing_replay_evidence_spool"
    elif case_name == "malformed_evidence_sidecar":
        evidence_sidecar.write_text('{"schema_version":1}\n', encoding="utf-8")
        expected_error_class = "invalid_replay_evidence_sidecar"
    elif case_name == "mismatched_evidence_relationship":
        payload = json.loads(evidence_sidecar.read_text(encoding="utf-8"))
        payload["valid_prefix_bytes"] = payload["valid_prefix_bytes"] + 1
        evidence_sidecar.write_bytes(spool._canonical_json_bytes(payload))
        expected_error_class = "invalid_blocker_relationship"
    elif case_name == "missing_prefix_required":
        prefix_path.unlink()
        expected_error_class = "invalid_blocker_relationship"
    elif case_name == "mismatched_prefix_required":
        prefix_path.write_bytes(clean_frame[:-1])
        expected_error_class = "invalid_blocker_relationship"
    else:
        raise AssertionError(f"unknown blocker crash-state case: {case_name}")

    return {
        "blocked_sequence": 1,
        "blocking_offset": len(clean_frame),
        "blocker_path": blocker_path,
        "evidence_spool": evidence_spool,
        "evidence_sidecar": evidence_sidecar,
        "expected_error_class": expected_error_class,
        "prefix_path": prefix_path,
    }


@pytest.mark.parametrize(
    ("case_name", "expected_error_class"),
    [
        ("missing_evidence_sidecar", "missing_replay_evidence_sidecar"),
        ("missing_evidence_spool", "missing_replay_evidence_spool"),
        ("malformed_evidence_sidecar", "invalid_replay_evidence_sidecar"),
        ("mismatched_evidence_relationship", "invalid_blocker_relationship"),
        ("missing_prefix_required", "invalid_blocker_relationship"),
        ("mismatched_prefix_required", "invalid_blocker_relationship"),
    ],
)
def test_blocker_crash_state_outcome_manual_replay_returns_blocked_integrity_and_fd_stable(
    db, tmp_path, monkeypatch, case_name, expected_error_class
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = _build_blocker_crash_state(home, case_name)

    import psutil

    baseline_fds = psutil.Process().num_fds()

    first = spool.replay_to_session_db(db, trigger="manual")
    second = spool.replay_to_session_db(db, trigger="manual")

    assert first.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert first.first_blocked_segment == state["blocked_sequence"]
    assert first.first_blocked_offset == state["blocking_offset"]
    assert first.error_class == expected_error_class
    assert db.get_messages("replay-session") == []
    assert [row["content"] for row in db.get_messages("replay-session") if row["content"] == "gamma"] == []
    assert second.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert second.first_blocked_segment == state["blocked_sequence"]
    assert second.first_blocked_offset == state["blocking_offset"]
    assert second.error_class == expected_error_class
    assert state["blocker_path"].exists()
    assert psutil.Process().num_fds() == baseline_fds


@pytest.mark.parametrize(
    ("case_name", "expected_error_class"),
    [
        ("missing_evidence_sidecar", "missing_replay_evidence_sidecar"),
        ("missing_evidence_spool", "missing_replay_evidence_spool"),
        ("malformed_evidence_sidecar", "invalid_replay_evidence_sidecar"),
        ("mismatched_evidence_relationship", "invalid_blocker_relationship"),
        ("missing_prefix_required", "invalid_blocker_relationship"),
        ("mismatched_prefix_required", "invalid_blocker_relationship"),
    ],
)
def test_blocker_crash_state_outcome_startup_remains_available_and_fail_closed(
    tmp_path, monkeypatch, case_name, expected_error_class
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = _build_blocker_crash_state(home, case_name)
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)

    startup_db = SessionDB(db_path=home / "state.db")
    try:
        assert startup_db.get_messages("replay-session") == []
        assert state["blocker_path"].exists()
    finally:
        startup_db.close()

    replay_db = SessionDB(db_path=home / "state.db")
    try:
        replay_result = spool.replay_to_session_db(replay_db, trigger="startup")
        assert replay_result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
        assert replay_result.first_blocked_segment == state["blocked_sequence"]
        assert replay_result.first_blocked_offset == state["blocking_offset"]
        assert replay_result.error_class == expected_error_class
        assert replay_db.get_messages("replay-session") == []
        assert [row["content"] for row in replay_db.get_messages("replay-session") if row["content"] == "gamma"] == []
    finally:
        replay_db.close()


def test_blocker_backed_valid_prefix_active_manual_replays_prefix_once_then_blocks_restart_duplicate_safe(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    first = _record(unit_id="unit-a", content="alpha")
    second = _record(unit_id="unit-b", content="beta")
    clean_frame = spool._frame_bytes_for_record(first)
    corrupt_frame = bytearray(spool._frame_bytes_for_record(second))
    corrupt_frame[-1] ^= 0x01
    expected_sequence = 1

    spool.append_records((first, second))
    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    active_path.write_bytes(clean_frame + bytes(corrupt_frame))

    first_result = spool.replay_to_session_db(db, trigger="manual")

    blocker_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.BLOCKERS_DIR_NAME
        / f"{expected_sequence:020d}.blocker.json"
    )
    ack_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.ACKS_DIR_NAME
        / f"{expected_sequence:020d}.prefix.spool.ap{len(clean_frame):020d}.json"
    )

    assert first_result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert first_result.first_blocked_segment == expected_sequence
    assert first_result.first_blocked_offset == len(clean_frame)
    assert blocker_path.exists()
    assert ack_path.exists()
    evidence_spool, _evidence_sidecar = _assert_metadata_only_replay_evidence(
        home,
        sequence=expected_sequence,
        expected_source_kind="active",
        expected_tail_status="checksum_mismatch",
        expected_valid_prefix_bytes=len(clean_frame),
        expected_original_size_bytes=len(clean_frame) + len(corrupt_frame),
    )
    assert evidence_spool.read_bytes() == clean_frame + bytes(corrupt_frame)

    restarted = SessionDB(db_path=home / "state.db")
    try:
        second_result = spool.replay_to_session_db(restarted, trigger="manual")
        assert second_result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
        assert [row["content"] for row in restarted.get_messages("replay-session")] == [
            "alpha"
        ]
        assert second_result.first_blocked_segment == expected_sequence
        assert second_result.first_blocked_offset == len(clean_frame)
        assert restarted.get_messages("replay-session") == db.get_messages("replay-session")
        assert blocker_path.exists()
        assert ack_path.exists()
    finally:
        restarted.close()


def test_blocker_backed_valid_prefix_sealed_startup_replays_prefix_once_then_blocks_restart_duplicate_safe(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)

    first = _record(unit_id="unit-a", content="alpha")
    second = _record(unit_id="unit-b", content="beta")
    clean_frame = spool._frame_bytes_for_record(first)
    corrupt_frame = bytearray(spool._frame_bytes_for_record(second))
    corrupt_frame[-1] ^= 0x01
    _write_blocker_backed_prefix_state(
        home,
        sequence=1,
        source_kind="sealed",
        prefix_bytes=clean_frame,
        original_bytes=clean_frame + bytes(corrupt_frame),
        tail_status="checksum_mismatch",
    )
    _write_sealed_segment(home, 2, _record(unit_id="unit-c", content="gamma"))

    startup_db = SessionDB(db_path=home / "state.db")
    try:
        blocker_path = (
            home
            / spool.SPOOL_ROOT_NAME
            / spool.SEALED_DIR_NAME
            / spool.BLOCKERS_DIR_NAME
            / "00000000000000000001.blocker.json"
        )
        ack_path = (
            home
            / spool.SPOOL_ROOT_NAME
            / spool.SEALED_DIR_NAME
            / spool.ACKS_DIR_NAME
            / f"00000000000000000001.prefix.spool.ap{len(clean_frame):020d}.json"
        )
        assert [row["content"] for row in startup_db.get_messages("replay-session")] == [
            "alpha"
        ]
        assert [row["content"] for row in startup_db.get_messages("replay-session") if row["content"] == "gamma"] == []
        assert blocker_path.exists()
        assert ack_path.exists()
        evidence_spool, _evidence_sidecar = _assert_metadata_only_replay_evidence(
            home,
            sequence=1,
            expected_source_kind="sealed",
            expected_tail_status="checksum_mismatch",
            expected_valid_prefix_bytes=len(clean_frame),
            expected_original_size_bytes=len(clean_frame) + len(corrupt_frame),
        )
        assert evidence_spool.read_bytes() == clean_frame + bytes(corrupt_frame)
    finally:
        startup_db.close()

    monkeypatch.setattr(hermes_state, "_SESSION_SPOOL_STARTUP_ONCE", set(), raising=False)
    restarted = SessionDB(db_path=home / "state.db")
    try:
        replay_result = spool.replay_to_session_db(restarted, trigger="startup")
        assert replay_result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
        assert replay_result.first_blocked_segment == 1
        assert replay_result.first_blocked_offset == len(clean_frame)
        assert [row["content"] for row in restarted.get_messages("replay-session")] == [
            "alpha"
        ]
    finally:
        restarted.close()


def test_blocker_backed_valid_prefix_zero_prefix_active_remains_blocked_without_messages(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    spool.append_records((_record(unit_id="zero-prefix-active"),))
    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    corrupted = bytearray(active_path.read_bytes())
    corrupted[0] = 0
    active_path.write_bytes(bytes(corrupted))

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert db.get_messages("replay-session") == []


def test_blocker_backed_valid_prefix_zero_prefix_sealed_remains_blocked_without_messages(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)

    corrupt_frame = bytearray(spool._frame_bytes_for_record(_record(unit_id="unit-bad", content="beta")))
    corrupt_frame[0] = 0
    (sealed / "00000000000000000001.spool").write_bytes(bytes(corrupt_frame))
    (sealed / "00000000000000000002.spool").write_bytes(
        spool._frame_bytes_for_record(_record(unit_id="unit-c", content="gamma"))
    )

    result = spool.replay_to_session_db(db, trigger="manual")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert db.get_messages("replay-session") == []
    assert [row["content"] for row in db.get_messages("replay-session") if row["content"] == "gamma"] == []


def test_replay_stops_at_blocker_and_does_not_advance_later_sequences_after_restart(
    db, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    blockers = sealed / spool.BLOCKERS_DIR_NAME
    quarantine = root / spool.QUARANTINE_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)
    blockers.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)

    clean_frame = spool._frame_bytes_for_record(_record(unit_id="unit-a", content="alpha"))
    corrupt_frame = bytearray(
        spool._frame_bytes_for_record(_record(unit_id="unit-b", content="beta"))
    )
    corrupt_frame[-1] ^= 0x01
    (sealed / "00000000000000000001.spool").write_bytes(clean_frame + bytes(corrupt_frame))
    (sealed / "00000000000000000002.spool").write_bytes(
        spool._frame_bytes_for_record(_record(unit_id="unit-c", content="gamma"))
    )

    first = spool.replay_to_session_db(db, trigger="startup")

    assert first.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]

    restarted = SessionDB(db_path=home / "state.db")
    try:
        second = spool.replay_to_session_db(restarted, trigger="startup")
        assert second.state is spool.ReplayRunState.BLOCKED_INTEGRITY
        assert [row["content"] for row in restarted.get_messages("replay-session")] == [
            "alpha"
        ]
        assert restarted.get_messages("replay-session") == db.get_messages("replay-session")
    finally:
        restarted.close()


def test_corrupt_sealed_segment_replays_only_prefix_then_blocks_fifo(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)

    clean_frame = spool._frame_bytes_for_record(_record(unit_id="unit-a", content="alpha"))
    corrupt_frame = bytearray(
        spool._frame_bytes_for_record(_record(unit_id="unit-b", content="beta"))
    )
    corrupt_frame[-1] ^= 0x01
    (sealed / "00000000000000000001.spool").write_bytes(clean_frame + bytes(corrupt_frame))
    (sealed / "00000000000000000002.spool").write_bytes(
        spool._frame_bytes_for_record(_record(unit_id="unit-c", content="gamma"))
    )

    result = spool.replay_to_session_db(db, trigger="startup")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert result.first_blocked_segment == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert [row["content"] for row in db.get_messages("replay-session") if row["content"] == "gamma"] == []


def _write_ack_sidecar(home, segment_name: str, acked_prefix_bytes: int, valid_prefix_bytes: int, *, sequence: int = 1):
    acks = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.ACKS_DIR_NAME
    acks.mkdir(parents=True, exist_ok=True)
    ack_name = f"{segment_name}.ap{acked_prefix_bytes:020d}.json"
    payload = {
        "schema_version": 1,
        "segment_sequence": f"{sequence:020d}",
        "segment_name": segment_name,
        "segment_kind": "prefix" if segment_name.endswith(".prefix.spool") else "clean",
        "segment_size_bytes": valid_prefix_bytes,
        "acked_prefix_bytes": acked_prefix_bytes,
        "valid_prefix_bytes": valid_prefix_bytes,
        "tail_status": "clean",
        "last_frame_offset": 0,
        "last_frame_length": acked_prefix_bytes,
        "last_frame_checksum_hex": "1" * 32,
    }
    (acks / ack_name).write_bytes(spool._canonical_json_bytes(payload))
    return acks / ack_name


def test_full_ack_tombstone_suppresses_replay_but_partial_ack_tombstone_blocks(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)

    full = _write_ack_sidecar(home, "00000000000000000001.spool", 5, 5, sequence=1)
    empty_result = spool.replay_to_session_db(db, trigger="startup")

    assert empty_result.state is spool.ReplayRunState.EMPTY
    assert db.get_messages("replay-session") == []

    full.unlink()
    _write_ack_sidecar(home, "00000000000000000001.spool", 3, 5, sequence=1)
    blocked = spool.replay_to_session_db(db, trigger="startup")

    assert blocked.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert blocked.first_blocked_segment == 1


def test_blocker_held_full_ack_tombstone_keeps_fifo_stopped_after_restart(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    blockers = sealed / spool.BLOCKERS_DIR_NAME
    sealed.mkdir(parents=True, exist_ok=True)
    blockers.mkdir(parents=True, exist_ok=True)
    _write_ack_sidecar(home, "00000000000000000001.prefix.spool", 5, 5, sequence=1)
    (blockers / "00000000000000000001.blocker.json").write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": "00000000000000000001",
                "source_kind": "sealed",
                "tail_status": "checksum_mismatch",
                "valid_prefix_bytes": 5,
                "acked_prefix_bytes": 5,
                "blocking_offset": 5,
                "prefix_segment_name": "00000000000000000001.prefix.spool",
                "evidence_spool_name": "seq-00000000000000000001-checksum_mismatch-vp5.spool",
                "evidence_sidecar_name": "seq-00000000000000000001-checksum_mismatch-vp5.json",
                "original_size_bytes": 10,
            }
        )
    )
    (sealed / "00000000000000000002.spool").write_bytes(
        spool._frame_bytes_for_record(_record(unit_id="unit-c", content="gamma"))
    )

    first = spool.replay_to_session_db(db, trigger="startup")
    assert first.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert db.get_messages("replay-session") == []

    restarted = SessionDB(db_path=home / "state.db")
    try:
        second = spool.replay_to_session_db(restarted, trigger="startup")
        assert second.state is spool.ReplayRunState.BLOCKED_INTEGRITY
        assert restarted.get_messages("replay-session") == []
    finally:
        restarted.close()


def test_replay_respects_startup_pre_persist_and_manual_budgets(tmp_path, monkeypatch):
    def _run(trigger: str):
        home = tmp_path / trigger
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=home / "state.db")
        try:
            for idx in range(20):
                _write_sealed_segment(
                    home,
                    idx + 1,
                    _record(unit_id=f"unit-{idx}", content=f"msg-{idx}"),
                )
            result = spool.replay_to_session_db(db, trigger=trigger)
            return db, result
        except Exception:
            db.close()
            raise

    startup_db, startup = _run("startup")
    try:
        assert startup.state is spool.ReplayRunState.REPLAYED
        assert startup.frames_committed == 20
    finally:
        startup_db.close()

    pre_db, pre = _run("pre_persist")
    try:
        assert pre.state is spool.ReplayRunState.PARTIALLY_REPLAYED
        assert pre.frames_committed == 16
        assert pre.pending_bytes_after > 0
        assert len(pre_db.get_messages("replay-session")) == 16
    finally:
        pre_db.close()

    manual_db, manual = _run("manual")
    try:
        assert manual.state is spool.ReplayRunState.REPLAYED
        assert manual.frames_committed == 20
    finally:
        manual_db.close()


def test_retryable_ack_cleanup_returns_retry_pending_with_ack_pending_true(
    tmp_path, monkeypatch
):
    home = tmp_path / "ack-cleanup-retry"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    try:
        _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))
        _write_ack_sidecar(home, "00000000000000000001.spool", 3, 5, sequence=1)

        def _busy(**_kwargs):
            raise OSError(16, "busy")

        monkeypatch.setattr(spool, "_cleanup_stale_lower_ack_sidecars", _busy)

        result = spool.replay_to_session_db(db, trigger="manual")

        assert result.state is spool.ReplayRunState.RETRY_PENDING
        assert result.ack_pending is True
        assert result.frames_committed == 1
        assert result.frames_acked == 0
        assert result.retry_class == "ack_cleanup_busy"
        assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    finally:
        db.close()


def test_retry_cooldown_skips_early_trigger_and_allows_later_takeover(
    tmp_path, monkeypatch
):
    home = tmp_path / "retry-cooldown"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    try:
        _write_sealed_segment(home, 1, _record(unit_id="unit-a", content="alpha"))
        original = spool._publish_ack_sidecar_strict
        calls = {"count": 0}
        clock = {"now": 100.0}

        monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])

        def _flaky(runtime, *, segment_sequence, segment_path, ack_payload):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError(16, "busy")
            return original(
                runtime,
                segment_sequence=segment_sequence,
                segment_path=segment_path,
                ack_payload=ack_payload,
            )

        monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _flaky)

        first = spool.replay_to_session_db(db, trigger="manual")
        assert first.state is spool.ReplayRunState.RETRY_PENDING
        assert first.ack_pending is True
        assert calls["count"] == 1

        second = spool.replay_to_session_db(db, trigger="manual")
        assert second.state is spool.ReplayRunState.RETRY_PENDING
        assert second.cooldown_seconds > 0
        assert calls["count"] == 1

        clock["now"] += second.cooldown_seconds + 0.01
        third = spool.replay_to_session_db(db, trigger="manual")
        assert third.state is spool.ReplayRunState.REPLAYED
        assert calls["count"] == 2
    finally:
        db.close()


def test_owner_busy_returns_truthful_backlog_without_mutating_or_taking_over_owner(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    append = spool.append_records((_record(unit_id="owner-busy", content="alpha"),))
    active_path = Path(append.unit_results[0].receipt.path)
    active_bytes = active_path.read_bytes()
    sealed_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME
    context = _replay_owner_context()
    ready_parent, ready_child = context.Pipe()
    release_parent, release_child = context.Pipe()
    proc = context.Process(
        target=_hold_replay_owner_lock,
        args=(str(home), ready_child, release_child),
    )
    proc.start()
    ready_child.close()
    release_child.close()
    assert ready_parent.recv() is True
    try:
        result = spool.replay_to_session_db(object(), trigger="manual")

        assert proc.is_alive()
        assert result.state is spool.ReplayRunState.OWNER_BUSY
        assert result.pending_bytes_after == len(active_bytes)
        assert result.pending_frames_after == 1
        assert result.ack_pending is False
        assert result.first_blocked_segment is None
        assert result.first_blocked_offset is None
        assert active_path.read_bytes() == active_bytes
        assert not sealed_dir.exists()

        runtime = spool._open_locked_runtime()
        try:
            assert spool._try_acquire_replay_owner(runtime) is None
        finally:
            _close_runtime_fds(runtime)
    finally:
        release_parent.send(True)
        proc.join(5)
        ready_parent.close()
        release_parent.close()

    assert proc.exitcode == 0


def test_owner_busy_snapshot_failure_returns_unknown_metrics_without_mutation_or_takeover(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    append = spool.append_records((_record(unit_id="owner-busy-unknown", content="alpha"),))
    active_path = Path(append.unit_results[0].receipt.path)
    active_bytes = active_path.read_bytes()
    sealed_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME
    context = _replay_owner_context()
    ready_parent, ready_child = context.Pipe()
    release_parent, release_child = context.Pipe()
    proc = context.Process(
        target=_hold_replay_owner_lock,
        args=(str(home), ready_child, release_child),
    )
    proc.start()
    ready_child.close()
    release_child.close()
    assert ready_parent.recv() is True

    def _snapshot_boom(_runtime):
        raise OSError(errno.EIO, f"secondary snapshot race {active_path} owner-busy-unknown")

    monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)
    try:
        result = spool.replay_to_session_db(object(), trigger="manual")

        assert proc.is_alive()
        assert result.state is spool.ReplayRunState.OWNER_BUSY
        assert result.pending_bytes_after == -1
        assert result.pending_frames_after == -1
        assert result.ack_pending is False
        assert result.first_blocked_segment is None
        assert result.first_blocked_offset is None
        assert active_path.read_bytes() == active_bytes
        assert not sealed_dir.exists()

        runtime = spool._open_locked_runtime()
        try:
            assert spool._try_acquire_replay_owner(runtime) is None
        finally:
            _close_runtime_fds(runtime)
    finally:
        release_parent.send(True)
        proc.join(5)
        ready_parent.close()
        release_parent.close()

    assert proc.exitcode == 0


def test_retry_cooldown_returns_truthful_backlog_without_reacquiring_owner(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    db = SessionDB(db_path=home / "state.db")
    caplog.set_level(logging.INFO)
    segment_path = _write_sealed_segment(
        home,
        1,
        _record(unit_id="cooldown-truth", content="alpha"),
    )
    segment_bytes = segment_path.read_bytes()
    ack_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.ACKS_DIR_NAME
    original_publish = spool._publish_ack_sidecar_strict
    original_try_owner = spool._try_acquire_replay_owner
    calls = {"ack": 0, "owner": 0}

    def _count_owner(runtime):
        calls["owner"] += 1
        return original_try_owner(runtime)

    def _busy_once(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["ack"] += 1
        if calls["ack"] == 1:
            raise OSError(
                errno.EBUSY,
                f"busy {segment_path} cooldown-truth-key-0 alpha",
            )
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    monkeypatch.setattr(spool, "_try_acquire_replay_owner", _count_owner)
    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _busy_once)
    try:
        first = spool.replay_to_session_db(db, trigger="manual")
        second = spool.replay_to_session_db(db, trigger="manual")
    finally:
        db.close()

    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "ack_publish_busy"
    assert first.ack_pending is True
    assert first.cooldown_seconds > 0
    assert first.pending_bytes_after == len(segment_bytes)
    assert first.pending_frames_after == 1
    assert second.state is spool.ReplayRunState.RETRY_PENDING
    assert second.retry_class == "ack_publish_busy"
    assert second.ack_pending is True
    assert second.cooldown_seconds > 0
    assert second.pending_bytes_after == len(segment_bytes)
    assert second.pending_frames_after == 1
    assert segment_path.read_bytes() == segment_bytes
    assert ack_dir.exists()
    assert sorted(ack_dir.glob("*.json")) == []
    assert calls["ack"] == 1
    assert calls["owner"] == 1
    assert len(messages) == 1
    assert "retry_class=ack_publish_busy" in messages[0]
    assert "ack_pending=True" in messages[0]
    assert f"pending_bytes={len(segment_bytes)}" in messages[0]
    assert "pending_frames=1" in messages[0]


def test_retry_cooldown_snapshot_failure_returns_unknown_metrics_without_reacquiring_owner(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    db = SessionDB(db_path=home / "state.db")
    caplog.set_level(logging.INFO)
    segment_path = _write_sealed_segment(
        home,
        1,
        _record(unit_id="cooldown-unknown", content="alpha"),
    )
    segment_bytes = segment_path.read_bytes()
    ack_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.ACKS_DIR_NAME
    original_publish = spool._publish_ack_sidecar_strict
    original_try_owner = spool._try_acquire_replay_owner
    calls = {"ack": 0, "owner": 0}

    def _count_owner(runtime):
        calls["owner"] += 1
        return original_try_owner(runtime)

    def _busy_once(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["ack"] += 1
        if calls["ack"] == 1:
            raise OSError(
                errno.EBUSY,
                f"busy {segment_path} cooldown-unknown-key-0 alpha",
            )
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    def _snapshot_boom(_runtime):
        raise OSError(
            errno.EIO,
            f"secondary snapshot race {segment_path} cooldown-unknown-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_try_acquire_replay_owner", _count_owner)
    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _busy_once)
    try:
        first = spool.replay_to_session_db(db, trigger="manual")
        spool._REPLAY_LOG_STATE.clear()
        monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)
        second = spool.replay_to_session_db(db, trigger="manual")
    finally:
        db.close()

    messages = _replay_messages_for_state(caplog, "retry_pending")
    joined = "\n".join(messages)

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "ack_publish_busy"
    assert first.pending_bytes_after == len(segment_bytes)
    assert first.pending_frames_after == 1
    assert second.state is spool.ReplayRunState.RETRY_PENDING
    assert second.retry_class == "ack_publish_busy"
    assert second.ack_pending is True
    assert second.cooldown_seconds > 0
    assert second.pending_bytes_after == -1
    assert second.pending_frames_after == -1
    assert segment_path.read_bytes() == segment_bytes
    assert ack_dir.exists()
    assert sorted(ack_dir.glob("*.json")) == []
    assert calls["ack"] == 1
    assert calls["owner"] == 1
    assert len(messages) == 2
    assert "pending_bytes=-1" in messages[-1]
    assert "pending_frames=-1" in messages[-1]
    assert "secondary snapshot race" not in joined
    assert str(segment_path) not in joined
    assert "cooldown-unknown-key-0" not in joined
    assert "alpha" not in joined


def test_replay_to_session_db_seals_clean_active_spool_before_replaying(db, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = home / spool.SPOOL_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    active_path = root / spool.ACTIVE_SPOOL_NAME
    active_path.write_bytes(spool._frame_bytes_for_record(_record()))

    result = spool.replay_to_session_db(db, trigger="startup")

    assert result.state is spool.ReplayRunState.REPLAYED
    assert result.frames_committed == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["hello replay"]
    sealed_entries = sorted((root / spool.SEALED_DIR_NAME).glob("*.spool"))
    assert sealed_entries == []
    assert active_path.exists()
    assert active_path.read_bytes() == b""


def test_ack_publish_enospc_returns_not_durable_and_next_retry_is_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    segment_path = _write_sealed_segment(home, 1, _record(unit_id="unit-enospc", content="alpha"))
    segment_bytes = segment_path.read_bytes()
    caplog.set_level(logging.INFO)

    def _enospc(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, f"disk full {segment_path} unit-enospc-key-0 alpha")

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _enospc)

    first = spool.replay_to_session_db(db, trigger="manual")
    first_log = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )

    assert first.state is spool.ReplayRunState.NOT_DURABLE
    assert first.error_class == "errno_enospc"
    assert segment_path.exists()
    assert first.pending_bytes_after == len(segment_bytes)
    assert spool._pending_frames_for_log(first) == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert "pending_frames=1" in first_log
    assert f"pending_bytes={len(segment_bytes)}" in first_log
    assert "disk full" not in first_log
    assert str(segment_path) not in first_log
    assert "unit-enospc-key-0" not in first_log
    assert "alpha" not in first_log

    monkeypatch.undo()
    monkeypatch.setenv("HERMES_HOME", str(home))
    second = spool.replay_to_session_db(db, trigger="manual")

    assert second.state is spool.ReplayRunState.REPLAYED
    assert second.frames_committed == 0
    assert second.frames_duplicated == 1
    assert second.frames_acked == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert not segment_path.exists()


def test_second_ack_publish_enospc_reports_only_unacked_frame_and_retry_stays_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    segment_path = _write_sealed_segment(
        home,
        1,
        _record(unit_id="second-ack-enospc-a", content="alpha"),
        _record(unit_id="second-ack-enospc-b", content="second-is-longer"),
    )
    decoded = spool.decode_spool_segment(segment_path)
    first_frame, second_frame = decoded.prefix_frames
    caplog.set_level(logging.INFO)
    original_publish = spool._publish_ack_sidecar_strict
    calls = {"count": 0}

    def _fail_second(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError(
                errno.ENOSPC,
                f"disk full {segment_path} second-ack-enospc-b-key-0 second-is-longer",
            )
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _fail_second)

    first = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "not_durable")
    ack_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.ACKS_DIR_NAME

    assert first.state is spool.ReplayRunState.NOT_DURABLE
    assert first.error_class == "errno_enospc"
    assert first.frames_acked == 1
    assert first.pending_frames_after == 1
    assert first.pending_bytes_after == second_frame.frame_length
    assert first.ack_pending is True
    assert sorted(path.name for path in ack_dir.glob("*.json")) == [
        f"{segment_path.name}.ap{first_frame.frame_length:020d}.json"
    ]
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "alpha",
        "second-is-longer",
    ]
    assert _fts_message_count(db) == 2
    assert len(messages) == 1
    assert "pending_frames=1" in messages[0]
    assert f"pending_bytes={second_frame.frame_length}" in messages[0]
    assert "disk full" not in messages[0]
    assert str(segment_path) not in messages[0]
    assert "second-ack-enospc-b-key-0" not in messages[0]
    assert "second-is-longer" not in messages[0]

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", original_publish)

    second = spool.replay_to_session_db(db, trigger="manual")

    assert second.state is spool.ReplayRunState.REPLAYED
    assert second.frames_committed == 0
    assert second.frames_duplicated == 2
    assert second.frames_acked == 2
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "alpha",
        "second-is-longer",
    ]
    assert _fts_message_count(db) == 2
    assert not segment_path.exists()


@pytest.mark.parametrize(
    ("raised", "expected_retry_class"),
    [
        (OSError(errno.EBUSY, "busy"), "ack_publish_busy"),
        (spool.SpoolRetryableReplayError("ack_cleanup_busy", ack_pending=True), "ack_cleanup_busy"),
    ],
)
def test_second_ack_publish_retry_pending_reports_exact_unacked_frame_metadata(
    db, tmp_path, monkeypatch, caplog, raised, expected_retry_class
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    segment_path = _write_sealed_segment(
        home,
        1,
        _record(unit_id="second-ack-retry-a", content="alpha"),
        _record(unit_id="second-ack-retry-b", content="second-is-longer"),
    )
    decoded = spool.decode_spool_segment(segment_path)
    first_frame, second_frame = decoded.prefix_frames
    caplog.set_level(logging.INFO)
    original_publish = spool._publish_ack_sidecar_strict
    calls = {"count": 0}

    def _fail_second(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["count"] += 1
        if calls["count"] == 2:
            raise raised
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _fail_second)

    result = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class == expected_retry_class
    assert result.pending_frames_after == 1
    assert result.pending_bytes_after == second_frame.frame_length
    assert result.ack_pending is True
    assert result.cooldown_seconds > 0
    assert len(messages) == 1
    assert f"retry_class={expected_retry_class}" in messages[0]
    assert "ack_pending=True" in messages[0]
    assert "pending_frames=1" in messages[0]
    assert f"pending_bytes={second_frame.frame_length}" in messages[0]
    assert str(segment_path) not in messages[0]
    assert "second-is-longer" not in messages[0]


def test_blocker_prefix_retry_metrics_exclude_durable_acked_prefix(db, tmp_path, monkeypatch, caplog):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    first = _record(unit_id="blocker-prefix-a", content="alpha")
    second = _record(unit_id="blocker-prefix-b", content="beta-longer")
    third = _record(unit_id="blocker-prefix-c", content="gamma")
    first_frame = spool._frame_bytes_for_record(first)
    second_frame = spool._frame_bytes_for_record(second)
    third_frame = bytearray(spool._frame_bytes_for_record(third))
    third_frame[-1] ^= 0x01
    prefix_path = _write_blocker_backed_prefix_state(
        home,
        sequence=1,
        source_kind="sealed",
        prefix_bytes=first_frame + second_frame,
        original_bytes=first_frame + second_frame + bytes(third_frame),
        tail_status="checksum_mismatch",
    )
    blocker_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.BLOCKERS_DIR_NAME
        / "00000000000000000001.blocker.json"
    )
    blocker_payload = json.loads(blocker_path.read_text(encoding="utf-8"))
    blocker_payload["acked_prefix_bytes"] = len(first_frame)
    blocker_path.write_bytes(spool._canonical_json_bytes(blocker_payload))
    db.reconcile_bootstrap_and_append_messages_batch(
        first.bootstrap,
        first.batch_messages,
        replay_patience_s=2.0,
    )
    caplog.set_level(logging.INFO)

    def _blocked(*_args, **_kwargs):
        raise hermes_state.CompressionSessionClosedError("replay-session")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _blocked)

    result = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "blocked_integrity")

    assert result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert result.error_class == "CompressionSessionClosedError"
    assert result.first_blocked_segment == 1
    assert result.first_blocked_offset == len(first_frame)
    assert result.pending_frames_after == 1
    assert result.pending_bytes_after == len(second_frame)
    assert result.ack_pending is True
    assert prefix_path.exists()
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert len(messages) == 1
    assert "pending_frames=1" in messages[0]
    assert f"pending_bytes={len(second_frame)}" in messages[0]
    assert "replay-session" not in messages[0]
    assert str(prefix_path) not in messages[0]
    assert "beta-longer" not in messages[0]


def test_blocker_prefix_ack_publish_enospc_returns_not_durable_until_repaired(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()

    first = _record(unit_id="blocker-prefix-ack-enospc-a", content="alpha")
    second = _record(unit_id="blocker-prefix-ack-enospc-b", content="beta-longer")
    later = _record(unit_id="blocker-prefix-ack-enospc-c", content="gamma")
    corrupt_tail = bytearray(
        spool._frame_bytes_for_record(
            _record(unit_id="blocker-prefix-ack-enospc-tail", content="corrupt-tail")
        )
    )
    corrupt_tail[-1] ^= 0x01
    first_frame = spool._frame_bytes_for_record(first)
    second_frame = spool._frame_bytes_for_record(second)
    prefix_path = _write_blocker_backed_prefix_state(
        home,
        sequence=1,
        source_kind="sealed",
        prefix_bytes=first_frame + second_frame,
        original_bytes=first_frame + second_frame + bytes(corrupt_tail),
        tail_status="checksum_mismatch",
    )
    blocker_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.BLOCKERS_DIR_NAME
        / "00000000000000000001.blocker.json"
    )
    ack_dir = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.ACKS_DIR_NAME
    full_ack_path = ack_dir / f"00000000000000000001.prefix.spool.ap{len(first_frame) + len(second_frame):020d}.json"
    evidence_spool, evidence_sidecar = _assert_metadata_only_replay_evidence(
        home,
        sequence=1,
        expected_source_kind="sealed",
        expected_tail_status="checksum_mismatch",
        expected_valid_prefix_bytes=len(first_frame) + len(second_frame),
        expected_original_size_bytes=len(first_frame) + len(second_frame) + len(corrupt_tail),
    )
    caplog.set_level(logging.INFO)
    original_publish = spool._publish_ack_sidecar_strict
    calls = {"count": 0}

    def _fail_first(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(
                errno.ENOSPC,
                f"injected {prefix_path} blocker-prefix-ack-enospc-a-key-0 alpha",
            )
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _fail_first)

    first_result = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "not_durable")

    assert first_result.state is spool.ReplayRunState.NOT_DURABLE
    assert first_result.error_class == "errno_enospc"
    assert first_result.frames_decoded == 1
    assert first_result.frames_committed == 1
    assert first_result.frames_duplicated == 0
    assert first_result.frames_acked == 0
    assert first_result.bytes_decoded == len(first_frame)
    assert first_result.bytes_acked == 0
    assert first_result.pending_frames_after == 2
    assert first_result.pending_bytes_after == len(first_frame) + len(second_frame)
    assert first_result.ack_pending is True
    assert prefix_path.exists()
    assert blocker_path.exists()
    assert evidence_spool.exists()
    assert evidence_sidecar.exists()
    assert prefix_path.read_bytes() == first_frame + second_frame
    assert evidence_spool.read_bytes() == first_frame + second_frame + bytes(corrupt_tail)
    assert sorted(path.name for path in ack_dir.glob("*.json")) == []
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert [row["content"] for row in db.get_messages("replay-session") if row["content"] == "gamma"] == []
    assert len(messages) == 1
    assert "error_class=errno_enospc" in messages[0]
    assert "pending_frames=2" in messages[0]
    assert f"pending_bytes={len(first_frame) + len(second_frame)}" in messages[0]
    assert "injected" not in messages[0]
    assert "OSError" not in messages[0]
    assert str(prefix_path) not in messages[0]
    assert str(blocker_path) not in messages[0]
    assert str(evidence_spool) not in messages[0]
    assert "blocker-prefix-ack-enospc-a-key-0" not in messages[0]
    assert "alpha" not in messages[0]
    assert "beta-longer" not in messages[0]

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", original_publish)

    second_result = spool.replay_to_session_db(db, trigger="manual")

    assert second_result.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert second_result.error_class == "checksum_mismatch"
    assert second_result.first_blocked_segment == 1
    assert second_result.first_blocked_offset == len(first_frame) + len(second_frame)
    assert second_result.frames_committed == 1
    assert second_result.frames_duplicated == 1
    assert second_result.frames_acked == 2
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "alpha",
        "beta-longer",
    ]
    assert [row["content"] for row in db.get_messages("replay-session") if row["content"] == "gamma"] == []
    assert prefix_path.exists()
    assert blocker_path.exists()
    assert evidence_spool.exists()
    assert evidence_sidecar.exists()
    assert full_ack_path.exists()

    blocker_path.unlink()
    evidence_spool.unlink()
    evidence_sidecar.unlink()
    later_segment = _write_sealed_segment(home, 2, later)

    third_result = spool.replay_to_session_db(db, trigger="manual")

    assert third_result.state is spool.ReplayRunState.REPLAYED
    assert third_result.frames_committed == 1
    assert third_result.frames_duplicated == 2
    assert third_result.frames_acked == 3
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "alpha",
        "beta-longer",
        "gamma",
    ]
    assert not prefix_path.exists()
    assert not later_segment.exists()


def test_second_ack_publish_snapshot_failure_returns_unknown_metrics_without_leaking_context(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    segment_path = _write_sealed_segment(
        home,
        1,
        _record(unit_id="snapshot-unknown-a", content="alpha"),
        _record(unit_id="snapshot-unknown-b", content="second-is-longer"),
    )
    caplog.set_level(logging.INFO)
    original_publish = spool._publish_ack_sidecar_strict
    calls = {"count": 0}

    def _fail_second(runtime, *, segment_sequence, segment_path, ack_payload):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError(errno.EBUSY, f"busy {segment_path} snapshot-unknown-b-key-0 second-is-longer")
        return original_publish(
            runtime,
            segment_sequence=segment_sequence,
            segment_path=segment_path,
            ack_payload=ack_payload,
        )

    def _snapshot_boom(_runtime):
        raise OSError(errno.EIO, f"snapshot blew up {segment_path} second-is-longer")

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _fail_second)
    monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)

    result = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class == "ack_publish_busy"
    assert result.ack_pending is True
    assert result.cooldown_seconds > 0
    assert result.pending_bytes_after == -1
    assert result.pending_frames_after == -1
    assert len(messages) == 1
    assert "retry_class=ack_publish_busy" in messages[0]
    assert "ack_pending=True" in messages[0]
    assert "pending_bytes=-1" in messages[0]
    assert "pending_frames=-1" in messages[0]
    assert "snapshot blew up" not in messages[0]
    assert str(segment_path) not in messages[0]
    assert "snapshot-unknown-b-key-0" not in messages[0]
    assert "second-is-longer" not in messages[0]


def test_fully_acked_cleanup_busy_returns_retry_pending_with_zero_backlog_and_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    record = _record(unit_id="ack-cleanup-busy", content="alpha")
    frame = spool._frame_bytes_for_record(record)
    segment_path = _write_sealed_segment(home, 1, record)
    ack_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.ACKS_DIR_NAME
        / f"{segment_path.name}.ap{len(frame):020d}.json"
    )
    caplog.set_level(logging.INFO)
    original_delete = spool._delete_fully_acked_segment

    def _busy(*_args, **_kwargs):
        raise OSError(
            errno.EBUSY,
            f"delete failed {segment_path} ack-cleanup-busy-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_delete_fully_acked_segment", _busy)

    first = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "ack_cleanup_busy"
    assert first.ack_pending is True
    assert first.cooldown_seconds > 0
    assert first.frames_committed == 1
    assert first.frames_duplicated == 0
    assert first.frames_acked == 1
    assert first.pending_bytes_after == 0
    assert first.pending_frames_after == 0
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert segment_path.exists()
    assert ack_path.exists()
    assert len(messages) == 1
    assert "retry_class=ack_cleanup_busy" in messages[0]
    assert "ack_pending=True" in messages[0]
    assert "pending_bytes=0" in messages[0]
    assert "pending_frames=0" in messages[0]
    assert "delete failed" not in messages[0]
    assert str(segment_path) not in messages[0]
    assert "ack-cleanup-busy-key-0" not in messages[0]
    assert "alpha" not in messages[0]

    monkeypatch.setattr(spool, "_delete_fully_acked_segment", original_delete)
    clock["now"] += first.cooldown_seconds + 0.01

    second = spool.replay_to_session_db(db, trigger="manual")

    assert second.state is spool.ReplayRunState.REPLAYED
    assert second.frames_committed == 0
    assert second.frames_duplicated == 1
    assert second.frames_acked == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert not segment_path.exists()
    assert not ack_path.exists()


def test_fully_acked_cleanup_enospc_returns_not_durable_with_zero_backlog_and_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    record = _record(unit_id="ack-cleanup-enospc", content="alpha")
    frame = spool._frame_bytes_for_record(record)
    segment_path = _write_sealed_segment(home, 1, record)
    ack_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.ACKS_DIR_NAME
        / f"{segment_path.name}.ap{len(frame):020d}.json"
    )
    caplog.set_level(logging.INFO)
    original_delete = spool._delete_fully_acked_segment

    def _enospc(*_args, **_kwargs):
        raise OSError(
            errno.ENOSPC,
            f"delete failed {segment_path} ack-cleanup-enospc-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_delete_fully_acked_segment", _enospc)

    first = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "not_durable")

    assert first.state is spool.ReplayRunState.NOT_DURABLE
    assert first.error_class == "errno_enospc"
    assert first.ack_pending is True
    assert first.frames_committed == 1
    assert first.frames_duplicated == 0
    assert first.frames_acked == 1
    assert first.pending_bytes_after == 0
    assert first.pending_frames_after == 0
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert segment_path.exists()
    assert ack_path.exists()
    assert len(messages) == 1
    assert "error_class=errno_enospc" in messages[0]
    assert "pending_bytes=0" in messages[0]
    assert "pending_frames=0" in messages[0]
    assert "delete failed" not in messages[0]
    assert str(segment_path) not in messages[0]
    assert "ack-cleanup-enospc-key-0" not in messages[0]
    assert "alpha" not in messages[0]

    monkeypatch.setattr(spool, "_delete_fully_acked_segment", original_delete)

    second = spool.replay_to_session_db(db, trigger="manual")

    assert second.state is spool.ReplayRunState.REPLAYED
    assert second.frames_committed == 0
    assert second.frames_duplicated == 1
    assert second.frames_acked == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert not segment_path.exists()
    assert not ack_path.exists()


@pytest.mark.parametrize(
    ("raised", "expected_state", "expected_retry_class", "expected_error_class"),
    [
        (
            OSError(errno.EBUSY, "busy"),
            spool.ReplayRunState.RETRY_PENDING,
            "ack_cleanup_busy",
            None,
        ),
        (
            OSError(errno.ENOSPC, "disk full"),
            spool.ReplayRunState.NOT_DURABLE,
            None,
            "errno_enospc",
        ),
    ],
)
def test_fully_acked_cleanup_snapshot_failure_returns_unknown_metrics_without_masking_outcome(
    db,
    tmp_path,
    monkeypatch,
    caplog,
    raised,
    expected_state,
    expected_retry_class,
    expected_error_class,
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    record = _record(unit_id="ack-cleanup-snapshot", content="alpha")
    segment_path = _write_sealed_segment(home, 1, record)
    caplog.set_level(logging.INFO)

    def _fail_delete(*_args, **_kwargs):
        raise OSError(
            raised.errno,
            f"cleanup boom {segment_path} ack-cleanup-snapshot-key-0 alpha",
        )

    def _snapshot_boom(_runtime):
        raise OSError(
            errno.EIO,
            f"snapshot blew up {segment_path} ack-cleanup-snapshot-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_delete_fully_acked_segment", _fail_delete)
    monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)

    result = spool.replay_to_session_db(db, trigger="manual")
    state_messages = _replay_messages_for_state(caplog, expected_state.value)
    joined = "\n".join(state_messages)

    assert result.state is expected_state
    assert result.retry_class == expected_retry_class
    assert result.error_class == expected_error_class
    assert result.ack_pending is True
    assert result.pending_bytes_after == -1
    assert result.pending_frames_after == -1
    if expected_state is spool.ReplayRunState.RETRY_PENDING:
        assert result.cooldown_seconds > 0
    assert len(state_messages) == 1
    assert "snapshot blew up" not in joined
    assert "cleanup boom" not in joined
    assert str(segment_path) not in joined
    assert "ack-cleanup-snapshot-key-0" not in joined
    assert "alpha" not in joined


def test_reconcile_active_enospc_returns_not_durable_with_truthful_pending_backlog(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    record = _record(unit_id="active-not-durable", content="alpha")
    append = spool.append_records((record,))
    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    surviving_bytes = active_path.read_bytes()
    caplog.set_level(logging.INFO)

    def _enospc(_runtime):
        raise OSError(
            errno.ENOSPC,
            f"disk full {active_path} active-not-durable-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _enospc)

    result = spool.replay_to_session_db(object(), trigger="startup")
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )

    assert Path(append.unit_results[0].receipt.path) == active_path
    assert result.state is spool.ReplayRunState.NOT_DURABLE
    assert result.error_class == "errno_enospc"
    assert active_path.read_bytes() == surviving_bytes
    assert result.pending_bytes_after == len(surviving_bytes)
    assert spool._pending_frames_for_log(result) == 1
    assert f"pending_bytes={len(surviving_bytes)}" in message
    assert "pending_frames=1" in message
    assert "disk full" not in message
    assert str(active_path) not in message
    assert "active-not-durable-key-0" not in message
    assert "alpha" not in message


def test_post_rename_active_recreate_enospc_returns_not_durable_with_truthful_sealed_backlog(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    record = _record(
        unit_id="post-rename-active-create-failure",
        content="post-rename-active-create-failure",
    )
    append = spool.append_records((record,))
    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    caplog.set_level(logging.INFO)
    original_open_file_at = spool._open_file_at

    def _fail_recreate(parent_fd, name, **kwargs):
        if (
            name == spool.ACTIVE_SPOOL_NAME
            and kwargs.get("create")
            and not active_path.exists()
        ):
            raise OSError(
                errno.ENOSPC,
                f"disk full {active_path} post-rename-active-create-failure-key-0 post-rename-active-create-failure",
            )
        return original_open_file_at(parent_fd, name, **kwargs)

    monkeypatch.setattr(spool, "_open_file_at", _fail_recreate)

    first = spool.replay_to_session_db(db, trigger="manual")
    first_log = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )
    sealed = sorted((home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME).glob("*.spool"))
    surviving_bytes = sum(path.stat().st_size for path in sealed)

    assert Path(append.unit_results[0].receipt.path) == active_path
    assert first.state is spool.ReplayRunState.NOT_DURABLE
    assert first.error_class == "errno_enospc"
    assert not active_path.exists()
    assert len(sealed) == 1
    assert surviving_bytes > 0
    assert first.pending_bytes_after == surviving_bytes
    assert spool._pending_frames_for_log(first) == 1
    assert db.get_messages("replay-session") == []
    assert f"pending_bytes={surviving_bytes}" in first_log
    assert "pending_frames=1" in first_log
    assert "disk full" not in first_log
    assert str(active_path) not in first_log
    assert "post-rename-active-create-failure-key-0" not in first_log
    assert "post-rename-active-create-failure" not in first_log

    monkeypatch.setattr(spool, "_open_file_at", original_open_file_at)
    second = spool.replay_to_session_db(db, trigger="manual")
    third = spool.replay_to_session_db(db, trigger="manual")

    assert second.state is spool.ReplayRunState.REPLAYED
    assert second.frames_committed == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == [
        "post-rename-active-create-failure"
    ]
    assert third.state is spool.ReplayRunState.EMPTY


def test_early_setup_not_durable_snapshot_counts_preexisting_sealed_partial_ack_backlog(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    first_record = _record(unit_id="sealed-partial-a", content="alpha")
    second_record = _record(unit_id="sealed-partial-b", content="beta")
    segment_path = _write_sealed_segment(home, 1, first_record, second_record)
    first_frame = spool._frame_bytes_for_record(first_record)
    second_frame = spool._frame_bytes_for_record(second_record)
    _write_ack_sidecar(
        home,
        segment_path.name,
        len(first_frame),
        segment_path.stat().st_size,
        sequence=1,
    )
    (home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME).write_bytes(b"")
    caplog.set_level(logging.INFO)

    def _enospc(_runtime):
        raise OSError(
            errno.ENOSPC,
            f"disk full {segment_path} sealed-partial-a-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _enospc)

    result = spool.replay_to_session_db(object(), trigger="startup")
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )

    assert result.state is spool.ReplayRunState.NOT_DURABLE
    assert result.error_class == "errno_enospc"
    assert result.pending_bytes_after == len(second_frame)
    assert spool._pending_frames_for_log(result) == 1
    assert result.ack_pending is True
    assert f"pending_bytes={len(second_frame)}" in message
    assert "pending_frames=1" in message
    assert "disk full" not in message
    assert str(segment_path) not in message
    assert "sealed-partial-a-key-0" not in message
    assert "alpha" not in message


def test_not_durable_pending_snapshot_unknown_preserves_original_error_and_logs_known_unknown_transitions(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    monkeypatch.setattr(spool.time, "monotonic", lambda: 100.0)
    record = _record(unit_id="snapshot-unknown", content="alpha")
    append = spool.append_records((record,))
    active_path = Path(append.unit_results[0].receipt.path)
    caplog.set_level(logging.INFO)

    def _enospc(_runtime):
        raise OSError(
            errno.ENOSPC,
            f"disk full {active_path} snapshot-unknown-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _enospc)
    original_snapshot = getattr(spool, "_snapshot_pending_backlog", None)

    first = spool.replay_to_session_db(object(), trigger="startup")

    def _snapshot_boom(_runtime):
        raise OSError(errno.EIO, f"secondary snapshot race {active_path} alpha")

    monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)
    second = spool.replay_to_session_db(object(), trigger="startup")
    monkeypatch.setattr(spool, "_snapshot_pending_backlog", original_snapshot)
    third = spool.replay_to_session_db(object(), trigger="startup")

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    ]
    joined = "\n".join(messages)

    assert first.state is spool.ReplayRunState.NOT_DURABLE
    assert first.error_class == "errno_enospc"
    assert first.pending_bytes_after == active_path.stat().st_size
    assert first.pending_frames_after == 1

    assert second.state is spool.ReplayRunState.NOT_DURABLE
    assert second.error_class == "errno_enospc"
    assert second.pending_bytes_after == -1
    assert second.pending_frames_after == -1

    assert third.state is spool.ReplayRunState.NOT_DURABLE
    assert third.error_class == "errno_enospc"
    assert third.pending_bytes_after == active_path.stat().st_size
    assert third.pending_frames_after == 1

    assert len(messages) == 3
    assert f"pending_bytes={active_path.stat().st_size}" in messages[0]
    assert "pending_frames=1" in messages[0]
    assert "pending_bytes=-1" in messages[1]
    assert "pending_frames=-1" in messages[1]
    assert f"pending_bytes={active_path.stat().st_size}" in messages[2]
    assert "pending_frames=1" in messages[2]
    assert "secondary snapshot race" not in joined
    assert str(active_path) not in joined
    assert "snapshot-unknown-key-0" not in joined
    assert "alpha" not in joined


def test_not_durable_active_replacement_during_snapshot_returns_unknown_backlog(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    record = _record(unit_id="snapshot-race", content="alpha")
    append = spool.append_records((record,))
    active_path = Path(append.unit_results[0].receipt.path)
    replacement_frame = spool._frame_bytes_for_record(
        _record(
            unit_id="snapshot-race-replacement",
            content="replacement-is-longer",
        )
    )
    parked_path = active_path.with_name("held-active.spool")
    caplog.set_level(logging.INFO)

    def _enospc(_runtime):
        raise OSError(
            errno.ENOSPC,
            f"disk full {active_path} snapshot-race-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _enospc)
    original_scan = spool._scan_fd
    swapped = {"done": False}

    def _swap(fd: int):
        if not swapped["done"]:
            swapped["done"] = True
            os.replace(active_path, parked_path)
            active_path.write_bytes(replacement_frame)
        return original_scan(fd)

    monkeypatch.setattr(spool, "_scan_fd", _swap)

    result = spool.replay_to_session_db(object(), trigger="startup")
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )

    assert result.state is spool.ReplayRunState.NOT_DURABLE
    assert result.error_class == "errno_enospc"
    assert result.pending_bytes_after == -1
    assert result.pending_frames_after == -1
    assert active_path.read_bytes() == replacement_frame
    assert "error_class=errno_enospc" in message
    assert "pending_bytes=-1" in message
    assert "pending_frames=-1" in message
    assert "disk full" not in message
    assert str(active_path) not in message
    assert "snapshot-race-key-0" not in message
    assert "alpha" not in message


def test_prepare_busy_returns_retry_pending_with_truthful_active_backlog_and_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    record = _record(unit_id="prepare-busy", content="alpha")
    append = spool.append_records((record,))
    active_path = Path(append.unit_results[0].receipt.path)
    active_bytes = active_path.read_bytes()
    caplog.set_level(logging.INFO)
    original_reconcile = spool._reconcile_active_spool_for_replay
    calls = {"count": 0}

    def _busy(_runtime):
        calls["count"] += 1
        raise OSError(errno.EBUSY, f"busy {active_path} prepare-busy-key-0 alpha")

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _busy)

    first = spool.replay_to_session_db(db, trigger="startup")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "spool_prepare_busy"
    assert first.ack_pending is False
    assert first.cooldown_seconds > 0
    assert first.pending_bytes_after == len(active_bytes)
    assert first.pending_frames_after == 1
    assert first.first_blocked_segment is None
    assert first.first_blocked_offset is None
    assert active_path.read_bytes() == active_bytes
    assert calls["count"] == 1
    assert len(messages) == 1
    assert "retry_class=spool_prepare_busy" in messages[0]
    assert "ack_pending=False" in messages[0]
    assert f"pending_bytes={len(active_bytes)}" in messages[0]
    assert "pending_frames=1" in messages[0]
    assert f"busy {active_path}" not in messages[0]
    assert str(active_path) not in messages[0]
    assert "prepare-busy-key-0" not in messages[0]
    assert "alpha" not in messages[0]

    second = spool.replay_to_session_db(db, trigger="startup")

    assert second.state is spool.ReplayRunState.RETRY_PENDING
    assert second.retry_class == "spool_prepare_busy"
    assert second.ack_pending is False
    assert second.cooldown_seconds > 0
    assert calls["count"] == 1

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", original_reconcile)
    clock["now"] += first.cooldown_seconds + 0.01

    third = spool.replay_to_session_db(db, trigger="startup")
    fourth = spool.replay_to_session_db(db, trigger="startup")

    assert third.state is spool.ReplayRunState.REPLAYED
    assert third.frames_committed == 1
    assert third.frames_duplicated == 0
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert active_path.exists()
    assert active_path.read_bytes() == b""
    assert fourth.state is spool.ReplayRunState.EMPTY


def test_prepare_busy_snapshot_failure_returns_unknown_metrics_without_masking_retry_class(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    monkeypatch.setattr(spool.time, "monotonic", lambda: 100.0)
    record = _record(unit_id="prepare-busy-unknown", content="alpha")
    append = spool.append_records((record,))
    active_path = Path(append.unit_results[0].receipt.path)
    active_bytes = active_path.read_bytes()
    caplog.set_level(logging.INFO)

    def _busy(_runtime):
        raise OSError(
            errno.EBUSY,
            f"busy {active_path} prepare-busy-unknown-key-0 alpha",
        )

    def _snapshot_boom(_runtime):
        raise OSError(
            errno.EIO,
            f"secondary snapshot race {active_path} prepare-busy-unknown-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_reconcile_active_spool_for_replay", _busy)
    monkeypatch.setattr(spool, "_snapshot_pending_backlog", _snapshot_boom)

    result = spool.replay_to_session_db(object(), trigger="startup")
    messages = _replay_messages_for_state(caplog, "retry_pending")
    joined = "\n".join(messages)

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class == "spool_prepare_busy"
    assert result.ack_pending is False
    assert result.cooldown_seconds > 0
    assert result.pending_bytes_after == -1
    assert result.pending_frames_after == -1
    assert result.first_blocked_segment is None
    assert result.first_blocked_offset is None
    assert active_path.read_bytes() == active_bytes
    assert len(messages) == 1
    assert "retry_class=spool_prepare_busy" in messages[0]
    assert "ack_pending=False" in messages[0]
    assert "pending_bytes=-1" in messages[0]
    assert "pending_frames=-1" in messages[0]
    assert f"busy {active_path}" not in joined
    assert "secondary snapshot race" not in joined
    assert str(active_path) not in joined
    assert "prepare-busy-unknown-key-0" not in joined
    assert "alpha" not in joined


def test_corrupt_active_evidence_enospc_returns_not_durable_and_leaves_active_bytes_truthful(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool.append_records((_record(unit_id="corrupt-active-enospc"),))

    active_path = home / spool.SPOOL_ROOT_NAME / spool.ACTIVE_SPOOL_NAME
    corrupted = bytearray(active_path.read_bytes())
    corrupted[0] = 0
    active_path.write_bytes(bytes(corrupted))
    caplog.set_level(logging.INFO)

    def _enospc(*_args, **_kwargs):
        raise OSError(
            errno.ENOSPC,
            f"disk full {active_path} corrupt-active-enospc-key-0 hello replay",
        )

    monkeypatch.setattr(spool, "_write_sidecar_json", _enospc)

    result = spool.replay_to_session_db(db, trigger="startup")
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )

    assert result.state is spool.ReplayRunState.NOT_DURABLE
    assert result.error_class == "errno_enospc"
    assert active_path.read_bytes() == bytes(corrupted)
    assert result.pending_bytes_after == len(corrupted)
    assert spool._pending_frames_for_log(result) == 0
    blockers = sorted((home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME).glob("*.blocker.json"))
    assert blockers == []
    assert f"pending_bytes={len(corrupted)}" in message
    assert "pending_frames=0" in message
    assert "disk full" not in message
    assert str(active_path) not in message
    assert "corrupt-active-enospc-key-0" not in message
    assert "hello replay" not in message


def test_corrupt_sealed_publish_enospc_returns_not_durable_with_truthful_backlog(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    first_segment = _write_sealed_segment(home, 1, _record(unit_id="corrupt-sealed-a", content="alpha"))
    corrupted = bytearray(first_segment.read_bytes())
    corrupted[0] = 0
    first_segment.write_bytes(bytes(corrupted))
    later_segment = _write_sealed_segment(
        home,
        2,
        _record(unit_id="corrupt-sealed-b", content="beta"),
    )
    caplog.set_level(logging.INFO)

    def _enospc(*_args, **_kwargs):
        raise OSError(
            errno.ENOSPC,
            f"disk full {first_segment} corrupt-sealed-a-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_publish_corrupt_sealed_segment_state", _enospc)

    result = spool.replay_to_session_db(object(), trigger="manual")
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__ and "state=not_durable" in record.getMessage()
    )
    expected_bytes = first_segment.stat().st_size + later_segment.stat().st_size

    assert result.state is spool.ReplayRunState.NOT_DURABLE
    assert result.error_class == "errno_enospc"
    assert first_segment.read_bytes() == bytes(corrupted)
    assert later_segment.exists()
    assert result.pending_bytes_after == expected_bytes
    assert spool._pending_frames_for_log(result) == 1
    assert f"pending_bytes={expected_bytes}" in message
    assert "pending_frames=1" in message
    assert "disk full" not in message
    assert str(first_segment) not in message
    assert "corrupt-sealed-a-key-0" not in message
    assert "alpha" not in message


def test_corrupt_sealed_publish_busy_returns_retry_pending_with_truthful_prefix_backlog_and_duplicate_safe(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    clean_frame = spool._frame_bytes_for_record(
        _record(unit_id="corrupt-publish-busy", content="alpha")
    )
    corrupt_frame = bytearray(
        spool._frame_bytes_for_record(_record(unit_id="corrupt-publish-busy-tail", content="beta"))
    )
    corrupt_frame[-1] ^= 0x01
    segment_path = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / "00000000000000000001.spool"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    segment_bytes = clean_frame + bytes(corrupt_frame)
    segment_path.write_bytes(segment_bytes)
    caplog.set_level(logging.INFO)
    original_publish = spool._publish_corrupt_sealed_segment_state

    def _busy(*_args, **_kwargs):
        raise OSError(
            errno.EBUSY,
            f"busy {segment_path} corrupt-publish-busy-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_publish_corrupt_sealed_segment_state", _busy)

    first = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "corrupt_publish_busy"
    assert first.ack_pending is False
    assert first.cooldown_seconds > 0
    assert first.frames_committed == 1
    assert first.frames_duplicated == 0
    assert first.frames_acked == 0
    assert first.pending_bytes_after == len(clean_frame)
    assert first.pending_frames_after == 1
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert segment_path.read_bytes() == segment_bytes
    assert len(messages) == 1
    assert "retry_class=corrupt_publish_busy" in messages[0]
    assert "ack_pending=False" in messages[0]
    assert f"pending_bytes={len(clean_frame)}" in messages[0]
    assert "pending_frames=1" in messages[0]
    assert f"busy {segment_path}" not in messages[0]
    assert str(segment_path) not in messages[0]
    assert "corrupt-publish-busy-key-0" not in messages[0]
    assert "alpha" not in messages[0]

    monkeypatch.setattr(spool, "_publish_corrupt_sealed_segment_state", original_publish)
    clock["now"] += first.cooldown_seconds + 0.01

    second = spool.replay_to_session_db(db, trigger="manual")
    blocker_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / spool.BLOCKERS_DIR_NAME
        / "00000000000000000001.blocker.json"
    )
    prefix_path = (
        home
        / spool.SPOOL_ROOT_NAME
        / spool.SEALED_DIR_NAME
        / "00000000000000000001.prefix.spool"
    )
    evidence_spool, evidence_sidecar = _assert_metadata_only_replay_evidence(
        home,
        sequence=1,
        expected_source_kind="sealed",
        expected_tail_status="checksum_mismatch",
        expected_valid_prefix_bytes=len(clean_frame),
        expected_original_size_bytes=len(segment_bytes),
    )

    assert second.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert second.error_class == "checksum_mismatch"
    assert second.first_blocked_segment == 1
    assert second.first_blocked_offset == len(clean_frame)
    assert second.frames_committed == 0
    assert second.frames_duplicated == 1
    assert second.frames_acked == 0
    assert [row["content"] for row in db.get_messages("replay-session")] == ["alpha"]
    assert _fts_message_count(db) == 1
    assert not segment_path.exists()
    assert prefix_path.exists()
    assert blocker_path.exists()
    assert evidence_spool.read_bytes() == segment_bytes
    assert evidence_sidecar.exists()


def test_corrupt_sealed_publish_busy_snapshot_failure_returns_unknown_metrics_without_masking_retry_class(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    spool._REPLAY_COOLDOWNS.clear()
    spool._REPLAY_LOG_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    clean_frame = spool._frame_bytes_for_record(
        _record(unit_id="corrupt-publish-unknown-a", content="alpha")
    )
    corrupt_frame = bytearray(
        spool._frame_bytes_for_record(_record(unit_id="corrupt-publish-unknown-b", content="beta"))
    )
    corrupt_frame[-1] ^= 0x01
    first_segment = home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME / "00000000000000000001.spool"
    first_segment.parent.mkdir(parents=True, exist_ok=True)
    first_segment.write_bytes(clean_frame + bytes(corrupt_frame))
    _write_sealed_segment(home, 2, _record(unit_id="corrupt-publish-unknown-c", content="gamma"))
    caplog.set_level(logging.INFO)

    def _busy(*_args, **_kwargs):
        raise OSError(
            errno.EBUSY,
            f"busy {first_segment} corrupt-publish-unknown-a-key-0 alpha",
        )

    def _snapshot_boom(_fd, **_kwargs):
        raise OSError(
            errno.EIO,
            f"secondary snapshot race {first_segment} corrupt-publish-unknown-a-key-0 alpha",
        )

    monkeypatch.setattr(spool, "_publish_corrupt_sealed_segment_state", _busy)
    monkeypatch.setattr(spool, "_scan_fd", _snapshot_boom)

    result = spool.replay_to_session_db(db, trigger="manual")
    messages = _replay_messages_for_state(caplog, "retry_pending")

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class == "corrupt_publish_busy"
    assert result.ack_pending is False
    assert result.cooldown_seconds > 0
    assert result.pending_bytes_after == -1
    assert result.pending_frames_after == -1
    assert len(messages) == 1
    assert "retry_class=corrupt_publish_busy" in messages[0]
    assert "pending_bytes=-1" in messages[0]
    assert "pending_frames=-1" in messages[0]
    assert "secondary snapshot race" not in messages[0]
    assert str(first_segment) not in messages[0]
    assert "corrupt-publish-unknown-a-key-0" not in messages[0]
    assert "alpha" not in messages[0]


def test_replay_degraded_logs_are_deduped_and_recovery_logs_once(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_sealed_segment(home, 1, _record(unit_id="unit-log", content="alpha"))
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    caplog.set_level(logging.INFO)

    def _busy(*_args, **_kwargs):
        raise hermes_state.CompressionSessionBusyError("busy")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _busy)
    first = spool.replay_to_session_db(db, trigger="manual")
    second = spool.replay_to_session_db(db, trigger="manual")

    clock["now"] += second.cooldown_seconds + 0.01

    def _closed(*_args, **_kwargs):
        raise hermes_state.CompressionSessionClosedError("replay-session")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _closed)
    third = spool.replay_to_session_db(db, trigger="manual")

    monkeypatch.setattr(
        db,
        "reconcile_bootstrap_and_append_messages_batch",
        lambda *_args, **_kwargs: hermes_state.AppendMessagesBatchResult(inserted_count=1, duplicate_count=0),
    )
    fourth = spool.replay_to_session_db(db, trigger="manual")

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == spool.__name__
    ]
    joined = "\n".join(messages)

    assert first.state is spool.ReplayRunState.RETRY_PENDING
    assert first.retry_class == "compression_busy"
    assert second.state is spool.ReplayRunState.RETRY_PENDING
    assert third.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert third.error_class == "CompressionSessionClosedError"
    assert fourth.state is spool.ReplayRunState.REPLAYED
    assert len(messages) == 3
    assert "state=retry_pending" in messages[0]
    assert "retry_class=compression_busy" in messages[0]
    assert "state=blocked_integrity" in messages[1]
    assert "error_class=CompressionSessionClosedError" in messages[1]
    assert "state=replayed" in messages[2]
    assert "alpha" not in joined
    assert "replay-session" not in joined


def test_replay_silent_blocked_branch_logs_and_empty_recovery_clears_state(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    caplog.set_level(logging.INFO)

    saved = {
        name: getattr(spool, name)
        for name in (
            "_reconcile_active_spool_for_replay",
            "_seal_clean_active_spool_for_replay",
            "_first_blocker_sequence",
            "_load_blocker_backed_prefix_replay_state",
            "_ordered_segment_entries",
        )
    }
    spool._reconcile_active_spool_for_replay = lambda _runtime: None
    spool._seal_clean_active_spool_for_replay = lambda _runtime: None
    spool._first_blocker_sequence = lambda **_kwargs: 1

    def _blocked(**_kwargs):
        raise spool.SpoolBlockedReplayError("compression_closed", frame_offset=0)

    spool._load_blocker_backed_prefix_replay_state = _blocked
    spool._ordered_segment_entries = lambda **_kwargs: []

    try:
        first = spool.replay_to_session_db(object(), trigger="manual")
        spool._first_blocker_sequence = lambda **_kwargs: None
        second = spool.replay_to_session_db(object(), trigger="manual")
    finally:
        for name, value in saved.items():
            setattr(spool, name, value)

    messages = [record.getMessage() for record in caplog.records if record.name == spool.__name__]
    joined = "\n".join(messages)

    assert first.state is spool.ReplayRunState.BLOCKED_INTEGRITY
    assert first.error_class == "compression_closed"
    assert second.state is spool.ReplayRunState.EMPTY
    assert len(messages) == 2
    assert "state=blocked_integrity" in messages[0]
    assert "error_class=compression_closed" in messages[0]
    assert "state=empty" in messages[1]
    assert "replay-session" not in joined
    assert "unit-" not in joined


def test_replay_silent_retry_branch_logs_truthful_pending_metadata(
    db, tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    first = _record(unit_id="blocked-prefix-a", content="alpha")
    second = _record(unit_id="blocked-prefix-b", content="beta")
    clean_frame = spool._frame_bytes_for_record(first)
    corrupt_frame = bytearray(spool._frame_bytes_for_record(second))
    corrupt_frame[-1] ^= 0x01
    _write_blocker_backed_prefix_state(
        home,
        sequence=1,
        source_kind="sealed",
        prefix_bytes=clean_frame,
        original_bytes=clean_frame + bytes(corrupt_frame),
        tail_status="checksum_mismatch",
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(spool.time, "monotonic", lambda: clock["now"])
    caplog.set_level(logging.INFO)

    def _busy(*_args, **_kwargs):
        raise hermes_state.CompressionSessionBusyError("busy")

    monkeypatch.setattr(db, "reconcile_bootstrap_and_append_messages_batch", _busy)

    result = spool.replay_to_session_db(db, trigger="manual")

    messages = [record.getMessage() for record in caplog.records if record.name == spool.__name__]
    joined = "\n".join(messages)

    assert result.state is spool.ReplayRunState.RETRY_PENDING
    assert result.retry_class == "compression_busy"
    assert len(messages) == 1
    assert "state=retry_pending" in messages[0]
    assert "retry_class=compression_busy" in messages[0]
    assert "pending_frames=1" in messages[0]
    assert f"pending_bytes={len(clean_frame)}" in messages[0]
    assert "alpha" not in joined
    assert "replay-session" not in joined