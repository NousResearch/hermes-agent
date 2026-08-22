from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import FTS_STORAGE_VERSION, SCHEMA_VERSION, SessionDB
from hermes_cli import session_recovery
from hermes_cli.session_recovery import (
    SessionRecoverySafetyError,
    SessionRecoverySourceError,
    inspect_session_database,
    recover_session_database,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_source(path: Path) -> dict[str, int]:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(3):
            session_id = f"recovery-session-{session_number}"
            db.create_session(
                session_id,
                "cli",
                cwd=f"/tmp/recovery-{session_number}",
            )
            db.set_session_title(session_id, f"Recovery {session_number}")
            for message_number in range(7):
                db.append_message(
                    session_id,
                    "user" if message_number % 2 == 0 else "assistant",
                    f"recoverable payload {session_number} {message_number}",
                )

        db.set_meta("goal:recovery-session-0", '{"status":"active"}')
        db.apply_telegram_topic_migration()
        db._conn.execute(
            """
            INSERT INTO telegram_dm_topic_mode (
                chat_id, user_id, enabled, activated_at, updated_at
            ) VALUES (?, ?, 1, ?, ?)
            """,
            ("chat-1", "user-1", 1.0, 2.0),
        )
        db._conn.execute(
            """
            INSERT INTO telegram_dm_topic_bindings (
                chat_id, thread_id, user_id, session_key, session_id,
                managed_mode, linked_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chat-1",
                "thread-1",
                "user-1",
                "telegram:user-1:chat-1",
                "recovery-session-0",
                "auto",
                1.0,
                2.0,
            ),
        )
        db._conn.execute(
            """
            INSERT INTO gateway_routing (
                scope, session_key, entry_json, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("telegram", "telegram:user-1:chat-1", "{}", 2.0),
        )
        db._conn.execute(
            """
            INSERT INTO async_delegations (
                delegation_id, origin_session, state, dispatched_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("delegation-1", "recovery-session-0", "completed", 1.0, 2.0),
        )
        # These are derived transition markers and must not reach the new DB.
        db.set_meta("fts_rebuild_high_water", "999")
        db.set_meta("fts_rebuild_progress", "500")
    finally:
        db.close()
    return {"sessions": 3, "messages": 21}


def _orphan_fts_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE type='table' "
            "AND name IN ('messages_fts', 'messages_fts_trigram')"
        )
        conn.execute("PRAGMA writable_schema=OFF")
    finally:
        conn.close()
def _make_page_spanning_source(
    path: Path,
    message_count: int = 320,
) -> tuple[int, int | None]:
    db = SessionDB(db_path=path)
    try:
        db.create_session(
            "partial-recovery-session",
            "cli",
            cwd="/tmp/partial-recovery",
        )
        for message_number in range(message_count):
            db.append_message(
                "partial-recovery-session",
                "user" if message_number % 2 == 0 else "assistant",
                (
                    f"partial recovery payload {message_number:04d} "
                    + chr(65 + message_number % 26) * 1_500
                ),
            )
    finally:
        db.close()

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM messages"
            ).fetchall()
        )
        count_index = next(
            (
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = 'messages'"
                ).fetchall()
                if plan.endswith(str(row[0]))
            ),
            None,
        )
        names = ["messages"]
        if count_index is not None:
            names.append(count_index)
        placeholders = ", ".join("?" for _ in names)
        roots = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT name, rootpage FROM sqlite_master "
                f"WHERE name IN ({placeholders})",
                tuple(names),
            ).fetchall()
        }
        return roots["messages"], (
            roots[count_index] if count_index is not None else None
        )
    finally:
        conn.close()


def _make_many_sessions_source(
    path: Path,
    session_count: int = 180,
) -> int:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(session_count):
            session_id = f"partial-session-{session_number:04d}"
            db.create_session(
                session_id,
                "cli",
                cwd=f"/tmp/partial-session-{session_number:04d}",
                system_prompt=(
                    f"session payload {session_number:04d} "
                    + chr(65 + session_number % 26) * 1_500
                ),
            )
            db.append_message(session_id, "user", f"message {session_number}")
    finally:
        db.close()

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        row = conn.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


def _btree_leaf_pages(path: Path, root_page: int) -> tuple[int, list[int]]:
    data = path.read_bytes()
    page_size = int.from_bytes(data[16:18], "big")
    if page_size == 1:
        page_size = 65_536
    leaf_pages: list[int] = []
    visited: set[int] = set()

    def visit(page_number: int) -> None:
        if page_number in visited:
            return
        visited.add(page_number)
        page_start = (page_number - 1) * page_size
        header_offset = page_start + (100 if page_number == 1 else 0)
        page_type = data[header_offset]
        cell_count = int.from_bytes(
            data[header_offset + 3 : header_offset + 5],
            "big",
        )
        if page_type in {0x0A, 0x0D}:
            leaf_pages.append(page_number)
            return
        assert page_type in {0x02, 0x05}, (
            f"unexpected table b-tree page type {page_type:#x} "
            f"on page {page_number}"
        )

        pointer_array = header_offset + 12
        for cell_number in range(cell_count):
            pointer_offset = pointer_array + cell_number * 2
            cell_offset = int.from_bytes(
                data[pointer_offset : pointer_offset + 2],
                "big",
            )
            child_offset = page_start + cell_offset
            child_page = int.from_bytes(
                data[child_offset : child_offset + 4],
                "big",
            )
            visit(child_page)
        rightmost_page = int.from_bytes(
            data[header_offset + 8 : header_offset + 12],
            "big",
        )
        visit(rightmost_page)

    visit(root_page)
    return page_size, leaf_pages


def _corrupt_middle_table_leaf(
    path: Path,
    root_page: int,
    *,
    require_interior: bool = True,
) -> int:
    page_size, leaf_pages = _btree_leaf_pages(path, root_page)
    assert leaf_pages
    if require_interior:
        assert len(leaf_pages) >= 3
    leaf_page = leaf_pages[len(leaf_pages) // 2]
    page_start = (leaf_page - 1) * page_size
    header_offset = page_start + (100 if leaf_page == 1 else 0)

    data = bytearray(path.read_bytes())
    assert data[header_offset] in {0x0A, 0x0D}
    # An impossible cell count damages this one middle leaf while preserving
    # the table root and leaves on both sides. This is a physical SQLite page
    # failure, not a mocked cursor exception.
    data[header_offset + 3 : header_offset + 5] = b"\xff\xff"
    path.write_bytes(data)
    return leaf_page


def _corrupt_table_root(path: Path, root_page: int) -> None:
    data = bytearray(path.read_bytes())
    page_size = int.from_bytes(data[16:18], "big")
    if page_size == 1:
        page_size = 65_536
    page_start = (root_page - 1) * page_size
    header_offset = page_start + (100 if root_page == 1 else 0)
    assert data[header_offset] in {0x02, 0x05, 0x0A, 0x0D}
    # Damage the root enough that no rowid bounds can be read. This reproduces
    # a fully failed sessions copy while leaving the messages b-tree intact.
    data[header_offset + 3 : header_offset + 5] = b"\xff\xff"
    path.write_bytes(data)


def test_snapshot_blocks_connections_opened_during_the_copy(
    tmp_path: Path,
) -> None:
    """A connection must not be able to open while raw copy descriptors exist.

    Checking has_live_connection() and then copying leaves a window: a
    connection can open between the two, and the copy's close() cancels its
    POSIX advisory locks. The guard must hold the lifecycle lock across the
    whole bundle copy.

    Runs the copy in a worker thread and pauses it inside the patched copy, so
    the assertion is about lock ordering rather than which thread the
    scheduler happens to resume first: while the copy is parked, a
    connect_tracked() attempt must NOT complete; once released, it must.
    """
    import threading

    from hermes_cli import session_recovery as recovery_module
    from hermes_cli.sqlite_safe_read import connect_tracked

    source = tmp_path / "racy-state.db"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _make_source(source)

    inside_copy = threading.Event()
    release_copy = threading.Event()
    connect_attempted = threading.Event()
    connection_opened = threading.Event()
    errors: list[str] = []
    real_copy2 = recovery_module.shutil.copy2

    def slow_copy2(src, dst, *args, **kwargs):
        result = real_copy2(src, dst, *args, **kwargs)
        if str(src).endswith("racy-state.db"):
            inside_copy.set()
            release_copy.wait(30)
        return result

    def do_copy():
        try:
            recovery_module._copy_source_bundle(source, snapshot_dir)
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(f"copy failed: {exc}")

    def do_connect():
        # Signal immediately before the blocking call so a timed "still
        # blocked" assertion cannot pass merely because this thread had not
        # been scheduled yet.
        connect_attempted.set()
        try:
            conn = connect_tracked(source, isolation_level=None, timeout=30.0)
            connection_opened.set()
            conn.close()
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(f"connect failed: {exc}")

    recovery_module.shutil.copy2 = slow_copy2
    copier = threading.Thread(target=do_copy, daemon=True)
    connector = threading.Thread(target=do_connect, daemon=True)
    try:
        copier.start()
        assert inside_copy.wait(30), "copy never reached the patched operation"

        connector.start()
        assert connect_attempted.wait(30), "connector thread never started"
        # The connector is at the lock. While the copy holds it, the
        # connection must not open.
        assert not connection_opened.wait(1.0), (
            "connect_tracked() completed while raw copy descriptors were open "
            "— the guard is not holding the lifecycle lock across the copy"
        )

        release_copy.set()
        # Once the copy finishes and releases the lock, it must open promptly.
        assert connection_opened.wait(30), (
            "connect_tracked() never completed after the copy released the lock"
        )
    finally:
        release_copy.set()
        recovery_module.shutil.copy2 = real_copy2
        copier.join(30)
        connector.join(30)

    assert not errors, errors[0]


def test_partial_recovery_keeps_messages_when_sessions_are_unsalvageable(
    tmp_path: Path,
) -> None:
    """Salvaged messages must survive even when NO session row is recoverable.

    Reported July 2026: a user's recovery copied 20,817 of 20,824 messages,
    then orphan cleanup deleted every one of them because the sessions b-tree
    was damaged worse than the messages b-tree. The output had 0 sessions and
    0 messages — the salvage worked and then threw the result away, which is
    the exact opposite of what --allow-partial is for.

    Messages must be retained under reconstructed placeholder sessions, and
    the placeholder-ness must be reported as loss rather than passed off as a
    clean recovery.
    """
    source = tmp_path / "sessions-destroyed.db"
    output = tmp_path / "sessions-destroyed-recovered.db"

    messages_per_session = {
        "doomed-session-a": 40,
        "doomed-session-b": 35,
        "doomed-session-c": 45,
    }
    db = SessionDB(db_path=source)
    try:
        for session_id, message_count in messages_per_session.items():
            db.create_session(session_id, "cli", cwd=f"/tmp/{session_id}")
            for index in range(message_count):
                db.append_message(
                    session_id,
                    "user",
                    f"irreplaceable {session_id} {index}",
                )
    finally:
        db.close()

    # sessions unrecoverable, messages intact — the reported shape.
    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        conn.execute("DELETE FROM sessions")
    finally:
        conn.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=16,
        allow_partial=True,
    )

    cleanup = report["orphan_cleanup"]
    assert cleanup["messages_removed"] == 0, (
        "salvaged messages were deleted for lack of a session row"
    )
    assert cleanup["sessions_reconstructed"] == len(messages_per_session)
    assert cleanup["messages_retained"] == 120

    with sqlite3.connect(str(output)) as verify:
        recovered_sessions = verify.execute(
            "SELECT id, source, title, message_count FROM sessions ORDER BY id"
        ).fetchall()
        messages = verify.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert messages == 120, f"expected all 120 messages retained, got {messages}"
    assert len(recovered_sessions) == len(messages_per_session)

    # Fabricated sessions must be identifiable and carry collision-safe titles.
    assert {row[0] for row in recovered_sessions} == set(messages_per_session)
    assert {row[1] for row in recovered_sessions} == {"recovered"}
    recovered_titles = [str(row[2]) for row in recovered_sessions]
    assert all(title.startswith("[recovered ") for title in recovered_titles)
    assert len(set(recovered_titles)) == len(recovered_titles)
    assert {
        str(row[0]): int(row[3]) for row in recovered_sessions
    } == messages_per_session

    # Retaining the data is still a lossy outcome and must say so.
    assert report["verification"]["loss_detected"] is True
    assert report["partial"] is True
    assert report["complete"] is False
    assert any(
        "reconstructed as placeholders" in warning
        for warning in report["verification"]["warnings"]
    ), report["verification"]["warnings"]

    # The output must remain structurally sound.
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verified"] is True
    assert report["installed"] is False










def test_cli_allow_partial_salvages_rows_across_a_corrupt_leaf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt-state.db"
    rejected_output = tmp_path / "rejected.db"
    output = tmp_path / "partial-recovered.db"
    message_count = 320
    messages_root, count_index_root = _make_page_spanning_source(
        source,
        message_count,
    )
    corrupt_page = _corrupt_middle_table_leaf(source, messages_root)
    if count_index_root is not None:
        _corrupt_middle_table_leaf(
            source,
            count_index_root,
            require_interior=False,
        )
    source_hash = _sha256(source)

    inspection = inspect_session_database(source, work_dir=tmp_path)
    assert inspection["recoverable"] is False
    assert inspection["tables"]["messages"]["rows"] is None
    with pytest.raises(SessionRecoverySourceError, match="messages"):
        recover_session_database(
            source,
            rejected_output,
            work_dir=tmp_path,
        )
    assert not rejected_output.exists()

    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "isolated-hermes-home")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "sessions",
            "recover",
            "--source",
            str(source),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path),
            "--chunk-size",
            "8",
            "--allow-partial",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Partial recovery output verified" in result.stdout
    assert "active session database was not changed" in result.stdout
    assert _sha256(source) == source_hash

    report_path = output.with_name(output.name + ".recovery.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["allow_partial"] is True
    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["installed"] is False
    assert report["source_unchanged"] is True
    assert report["verification"]["healthy"] is True
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verification"]["table_counts"]["sessions"] == 1

    copied_messages = report["copy"]["messages"]
    assert copied_messages["status"] == "partial"
    assert copied_messages["copied_rows"] < message_count
    assert copied_messages["copied_rows"] > 0
    assert copied_messages["skipped_rowid_ranges"]
    assert any(
        item["low"] <= message_count and item["high"] >= 1
        for item in copied_messages["skipped_rowid_ranges"]
    )
    assert copied_messages["query_limit_reached"] is False

    conn = sqlite3.connect(str(output))
    try:
        recovered_ids = {
            int(row[0]) for row in conn.execute("SELECT id FROM messages")
        }
        assert 1 in recovered_ids
        assert message_count in recovered_ids
        assert len(recovered_ids) == copied_messages["copied_rows"]
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        conn.close()

    # Prove the helper damaged an interior data leaf, so successful recovery of
    # the first and last message IDs really crossed the corrupted region.
    assert corrupt_page not in {
        min(_btree_leaf_pages(source, messages_root)[1]),
        max(_btree_leaf_pages(source, messages_root)[1]),
    }


def _surviving_output_bundle(output: Path) -> list[str]:
    """Every path ``_validate_paths`` would refuse a retry on."""

    return [
        str(session_recovery._sidecar_path(output, suffix))
        for suffix in session_recovery._SIDECAR_SUFFIXES
        if os.path.lexists(session_recovery._sidecar_path(output, suffix))
    ]


def test_failed_recovery_leaves_no_output_to_block_a_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-run failure must not lock the user out of retrying.

    ``_validate_paths`` refuses to start when the output *or any of its
    journal sidecars* already exists, so a half-written output turns the one
    command a damaged database has left into a permanent refusal naming a
    file the user never created.
    """

    source = tmp_path / "state.db"
    _make_source(source)
    output = tmp_path / "recovered.db"

    def _disk_error(destination: sqlite3.Connection) -> dict[str, object]:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(session_recovery, "_finalize_derived_metadata", _disk_error)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        recover_session_database(source, output, work_dir=tmp_path)

    assert _surviving_output_bundle(output) == []

    # The retry is the point: it must get past the overwrite guard entirely.
    monkeypatch.undo()
    report = recover_session_database(source, output, work_dir=tmp_path)
    assert report["complete"] is True
    assert report["installed"] is False


def test_keyboard_interrupt_during_recovery_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C is the reported case, so the cleanup must catch ``BaseException``.

    An ``except Exception`` cleanup passes the disk-error test above and still
    leaves the stub behind for the interrupt that people actually hit.
    """

    source = tmp_path / "state.db"
    _make_source(source)
    output = tmp_path / "interrupted.db"

    def _interrupt(destination: sqlite3.Connection) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(session_recovery, "_finalize_derived_metadata", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        recover_session_database(source, output, work_dir=tmp_path)

    assert _surviving_output_bundle(output) == []


def test_failed_output_initialization_leaves_no_journal_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output can fail *while* it is being created, sidecars and all.

    ``SessionDB`` creates the database before it finishes initializing it, so a
    constructor that raises part way leaves a live WAL behind and never hands
    back a handle anything can close. ``_validate_paths`` refuses on every
    entry of ``_SIDECAR_SUFFIXES``, so removing the database on its own still
    leaves ``recovered.db-wal`` blocking the retry.
    """

    source = tmp_path / "state.db"
    _make_source(source)
    output = tmp_path / "half-initialized.db"

    real_session_db = session_recovery.SessionDB
    leaked: list[SessionDB] = []

    def _fail_after_creating(*args: object, **kwargs: object) -> SessionDB:
        # Deliberately left open: an unflushed WAL is what makes this the
        # sidecar case rather than a plain leftover database file.
        leaked.append(real_session_db(*args, **kwargs))
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(session_recovery, "SessionDB", _fail_after_creating)
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            recover_session_database(source, output, work_dir=tmp_path)

        assert leaked, "the stub never reached the real constructor"
        assert _surviving_output_bundle(output) == []
    finally:
        for database in leaked:
            database.close()


def test_verification_failure_keeps_the_output_for_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary: cleanup belongs on the raise path only.

    ``_verify_recovered_database`` collects errors and returns, and the CLI
    tells the user to review a report for an output that failed verification,
    so that output must survive.
    """

    source = tmp_path / "state.db"
    _make_source(source)
    output = tmp_path / "unverified.db"

    def _failed_verification(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "errors": ["forced verification failure"],
            "warnings": [],
            "table_counts": {},
            "integrity_check": ["ok"],
            "foreign_key_check": [],
            "complete": False,
            "healthy": False,
            "loss_detected": False,
        }

    monkeypatch.setattr(
        session_recovery,
        "_verify_recovered_database",
        _failed_verification,
    )
    report = recover_session_database(source, output, work_dir=tmp_path)

    assert report["complete"] is False
    assert report["verified"] is False
    assert output.exists()


def test_failed_report_write_leaves_no_report_to_block_a_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same lockout, one artifact along.

    ``write_recovery_report`` opens with ``"x"`` and ``hermes sessions
    recover`` refuses the whole command when the report path already exists,
    so a ``json.dump`` that dies part way blocks the next run before recovery
    even starts.
    """

    destination = tmp_path / "recovered.db.recovery.json"

    def _no_space(*args: object, **kwargs: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(session_recovery.json, "dump", _no_space)
    with pytest.raises(OSError, match="No space left on device"):
        session_recovery.write_recovery_report(destination, {"operation": "recover"})

    assert not os.path.lexists(destination)

    monkeypatch.undo()
    written = session_recovery.write_recovery_report(
        destination,
        {"operation": "recover"},
    )
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "operation": "recover"
    }


def test_report_write_never_removes_a_file_it_did_not_create(
    tmp_path: Path,
) -> None:
    """The refusal itself must stay non-destructive.

    ``open("x")`` raising ``FileExistsError`` means the report was written by
    an earlier run, so the failure path must leave it exactly as it was.
    """

    destination = tmp_path / "recovered.db.recovery.json"
    destination.write_text("earlier report", encoding="utf-8")

    with pytest.raises(FileExistsError):
        session_recovery.write_recovery_report(destination, {"operation": "recover"})

    assert destination.read_text(encoding="utf-8") == "earlier report"


def test_partial_recovery_clears_only_unreadable_system_prompt_refs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt-system-prompts.db"
    output = tmp_path / "partial-system-prompts.db"
    session_count = 180
    _make_many_sessions_source(source, session_count)

    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        row = conn.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE type = 'table' AND name = 'system_prompts'"
        ).fetchone()
        assert row is not None
        prompt_root = int(row[0])
    finally:
        conn.close()
    _corrupt_middle_table_leaf(source, prompt_root)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    assert report["verified"] is True
    assert report["partial"] is True
    assert report["copy"]["sessions"]["status"] == "complete"
    assert report["copy"]["messages"]["status"] == "complete"
    assert report["copy"]["system_prompts"]["status"] == "partial"
    cleared = report["orphan_cleanup"]["session_prompt_refs_cleared"]
    assert 0 < cleared < session_count
    assert report["verification"]["foreign_key_check"] == []

    conn = sqlite3.connect(str(output))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == session_count
        retained = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE system_prompt_hash IS NOT NULL"
        ).fetchone()[0]
        assert retained == session_count - cleared
        assert (
            conn.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]
            == retained
        )
    finally:
        conn.close()


class _AutoRollbackConnection:
    """A destination whose transaction SQLite has already ended by itself.

    SQLite rolls the active transaction back on its own for SQLITE_FULL,
    SQLITE_IOERR, SQLITE_BUSY and SQLITE_NOMEM. The explicit ``ROLLBACK`` a
    cleanup handler then issues raises ``cannot rollback - no transaction is
    active``, which is exactly the sequence that used to replace the real
    failure. This stands in for it at every write site.
    """

    def __init__(self, connection: sqlite3.Connection, make_failure):
        self._connection = connection
        self._make_failure = make_failure
        self.rollback_attempts = 0

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def execute(self, sql: str, *parameters):
        statement = sql.strip().upper()
        if statement.startswith("ROLLBACK"):
            self.rollback_attempts += 1
            # Undo any transaction still open underneath so the connection
            # stays usable, then fail the way SQLite does once it has already
            # unwound the transaction itself. The simulated failure is raised
            # unconditionally so the stub never depends on the real
            # connection's transaction state.
            try:
                self._connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise sqlite3.OperationalError(
                "cannot rollback - no transaction is active"
            )
        if statement.startswith("COMMIT"):
            raise self._make_failure()
        return self._connection.execute(sql, *parameters)

    def executemany(self, sql: str, parameters):
        return self._connection.executemany(sql, parameters)


def _recovery_pair(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """A populated source plus a fresh current-schema destination."""

    source_path = tmp_path / "rollback-source.db"
    _make_source(source_path)
    destination_path = tmp_path / "rollback-destination.db"
    destination_db = SessionDB(db_path=destination_path)
    try:
        destination_db.apply_telegram_topic_migration()
    finally:
        destination_db.close()
    return (
        sqlite3.connect(str(source_path), isolation_level=None),
        sqlite3.connect(str(destination_path), isolation_level=None),
    )


# Every site in session_recovery.py that wraps a write in
# ``except BaseException: ROLLBACK; raise``.
_ROLLBACK_SITES = {
    "_cleanup_partial_orphans": lambda source, destination: (
        session_recovery._cleanup_partial_orphans(destination)
    ),
    "_copy_state_meta": lambda source, destination: (
        session_recovery._copy_state_meta(
            source,
            destination,
            chunk_size=1_000,
            progress_cb=None,
            source_rows=None,
        )
    ),
    "_copy_table": lambda source, destination: session_recovery._copy_table(
        source,
        destination,
        "messages",
        chunk_size=1_000,
        progress_cb=None,
        source_rows=21,
    ),
    "_copy_table_salvage": lambda source, destination: (
        session_recovery._copy_table_salvage(
            source,
            destination,
            "messages",
            chunk_size=1_000,
            progress_cb=None,
            source_rows=21,
        )
    ),
    "_finalize_derived_metadata": lambda source, destination: (
        session_recovery._finalize_derived_metadata(destination)
    ),
}


@pytest.mark.parametrize("site", sorted(_ROLLBACK_SITES))
def test_cleanup_rollback_never_replaces_the_real_write_failure(
    site: str,
    tmp_path: Path,
) -> None:
    source, destination = _recovery_pair(tmp_path)
    guarded = _AutoRollbackConnection(
        destination,
        lambda: sqlite3.OperationalError("database or disk is full"),
    )
    try:
        try:
            surfaced = json.dumps(
                _ROLLBACK_SITES[site](source, guarded), default=str
            )
        except sqlite3.Error as exc:
            # The two cleanup sites propagate instead of reporting a dict.
            surfaced = str(exc)
    finally:
        source.close()
        destination.close()

    # The salvage site retries narrower rowid ranges, so it rolls back more
    # than once before it gives up and records the failure.
    assert guarded.rollback_attempts >= 1
    assert "database or disk is full" in surfaced
    assert "cannot rollback" not in surfaced


@pytest.mark.parametrize("site", sorted(_ROLLBACK_SITES))
def test_interrupt_during_a_write_still_aborts_the_recovery(
    site: str,
    tmp_path: Path,
) -> None:
    """``except BaseException`` must roll back and then re-raise the interrupt.

    A masked rollback turns Ctrl-C into a ``DatabaseError``, which the copy
    sites treat as an ordinary damaged-source error — so the run continues
    instead of stopping.
    """

    source, destination = _recovery_pair(tmp_path)
    guarded = _AutoRollbackConnection(destination, KeyboardInterrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            _ROLLBACK_SITES[site](source, guarded)
    finally:
        source.close()
        destination.close()

    assert guarded.rollback_attempts == 1


def test_copy_table_reports_a_full_destination_not_a_rollback_failure(
    tmp_path: Path,
) -> None:
    """End-to-end against real SQLite, with no stubbing of the failure."""

    source_path = tmp_path / "page-spanning-source.db"
    _make_page_spanning_source(source_path)
    destination_path = tmp_path / "capped-destination.db"
    destination_db = SessionDB(db_path=destination_path)
    destination_db.close()

    source = sqlite3.connect(str(source_path), isolation_level=None)
    destination = sqlite3.connect(str(destination_path), isolation_level=None)
    try:
        source_rows = int(
            source.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        # Leave room for the copy to begin and then run out part way through,
        # which is what a destination filesystem filling up mid-run looks like.
        pages = int(destination.execute("PRAGMA page_count").fetchone()[0])
        destination.execute(f"PRAGMA max_page_count = {pages + 20}")

        result = session_recovery._copy_table(
            source,
            destination,
            "messages",
            chunk_size=1_000,
            progress_cb=None,
            source_rows=source_rows,
        )
    finally:
        source.close()
        destination.close()

    assert result["status"] == "failed"
    assert "database or disk is full" in result["error"]
    assert "cannot rollback" not in result["error"]



