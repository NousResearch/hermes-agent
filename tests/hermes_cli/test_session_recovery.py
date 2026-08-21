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
    write_recovery_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


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


def _make_auxiliary_omission_source(
    path: Path,
    *,
    delivery_states: tuple[str, ...],
    origin_session_id: str | None = None,
) -> None:
    db = SessionDB(db_path=path)
    try:
        db.create_session("omission-session", "cli", cwd="/tmp/omission")
        db.append_message("omission-session", "user", "synthetic fixture")
    finally:
        db.close()

    from gateway.delivery_ledger import _initialize_schema as init_delivery_schema
    from tools.async_delegation import _initialize_schema as init_delegation_schema

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        init_delivery_schema(conn)
        init_delegation_schema(conn)
        for index, state in enumerate(delivery_states):
            conn.execute(
                """INSERT INTO delivery_obligations (
                    obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"obligation-{index}",
                    "telegram:chat-1",
                    "telegram",
                    "chat-1",
                    None,
                    "synthetic final response",
                    state,
                    0,
                    1.0,
                    1.0,
                    999_999,
                    1,
                    None,
                ),
            )
        if origin_session_id is not None:
            conn.execute(
                """INSERT INTO async_delegations (
                    delegation_id, origin_session, state, dispatched_at,
                    updated_at, origin_session_id
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "delegation-omission",
                    "omission-session",
                    "completed",
                    1.0,
                    1.0,
                    origin_session_id,
                ),
            )
    finally:
        conn.close()


def _malform_table_schema(path: Path, table: str) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        conn.execute("PRAGMA writable_schema=ON")
        cursor = conn.execute(
            "UPDATE sqlite_master SET sql = ? "
            "WHERE type = 'table' AND name = ?",
            (f"CREATE TABLE {table}(", table),
        )
        assert cursor.rowcount == 1
        conn.execute(f"PRAGMA schema_version = {schema_version + 1}")
    finally:
        conn.close()


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


def test_recovery_reports_non_terminal_and_lazy_schema_omissions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "auxiliary-source.db"
    output = tmp_path / "auxiliary-recovered.db"
    _make_auxiliary_omission_source(
        source,
        delivery_states=("pending", "delivered", "unexpected-private-state"),
        origin_session_id="raw-origin-session-id",
    )

    report = recover_session_database(source, output, work_dir=tmp_path)
    report_path = tmp_path / "auxiliary-recovery.json"
    write_recovery_report(report_path, report)
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "delivery_obligations" in persisted_report["omissions"]["tables"]

    omissions = report["omissions"]
    delivery = omissions["tables"]["delivery_obligations"]
    assert delivery["state_counts"] == {
        "pending": 1,
        "delivered": 1,
        "<UNKNOWN>": 1,
    }
    assert delivery["non_terminal_rows"] == 2
    delegation = omissions["columns"]["async_delegations"]
    assert delegation["omitted_source_columns"] == ["origin_session_id"]
    assert delegation["columns_with_non_default_values"] == ["origin_session_id"]
    assert omissions["loss_detected"] is True
    assert report["verification"]["loss_detected"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["verified"] is True
    assert any(
        "pending final replies may be lost" in warning
        for warning in report["verification"]["warnings"]
    )

    conn = sqlite3.connect(str(output), isolation_level=None)
    try:
        assert _table_column_names(conn, "delivery_obligations") == []
        assert "origin_session_id" not in _table_column_names(
            conn,
            "async_delegations",
        )
    finally:
        conn.close()


def test_terminal_delivery_omission_is_reported_without_claiming_data_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "terminal-delivery-source.db"
    output = tmp_path / "terminal-delivery-recovered.db"
    _make_auxiliary_omission_source(
        source,
        delivery_states=("delivered", "abandoned"),
        origin_session_id="",
    )
    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        conn.execute(
            """INSERT INTO async_delegations (
                delegation_id, origin_session, state, dispatched_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)""",
            ("delegation-null-origin", "omission-session", "completed", 1.0, 1.0),
        )
    finally:
        conn.close()

    report = recover_session_database(source, output, work_dir=tmp_path)

    delivery = report["omissions"]["tables"]["delivery_obligations"]
    assert delivery["state_counts"] == {"delivered": 1, "abandoned": 1}
    assert delivery["non_terminal_rows"] == 0
    delegation = report["omissions"]["columns"]["async_delegations"]
    assert delegation["columns_with_non_default_values"] == []
    assert report["omissions"]["loss_detected"] is False
    assert report["verification"]["loss_detected"] is False
    assert report["complete"] is True
    assert report["partial"] is False
    assert any(
        "terminal delivery obligation" in warning
        for warning in report["verification"]["warnings"]
    )


def test_unreadable_omitted_table_schema_is_reported_as_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "malformed-auxiliary-schema.db"
    output = tmp_path / "malformed-auxiliary-recovered.db"
    _make_auxiliary_omission_source(
        source,
        delivery_states=("pending",),
    )
    _malform_table_schema(source, "delivery_obligations")

    report = recover_session_database(source, output, work_dir=tmp_path)

    delivery = report["omissions"]["tables"]["delivery_obligations"]
    assert delivery["omitted"] is True
    assert delivery["inspection_error"] == "source table schema is unreadable"
    assert report["omissions"]["loss_detected"] is True
    assert report["verification"]["loss_detected"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["verified"] is True


def test_source_only_literal_default_is_reported_without_claiming_data_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "defaulted-column-source.db"
    output = tmp_path / "defaulted-column-recovered.db"
    _make_source(source)
    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        conn.execute(
            "ALTER TABLE async_delegations "
            "ADD COLUMN future_marker TEXT DEFAULT 'unset'"
        )
    finally:
        conn.close()

    report = recover_session_database(source, output, work_dir=tmp_path)

    delegation = report["omissions"]["columns"]["async_delegations"]
    assert delegation["omitted_source_columns"] == ["future_marker"]
    assert delegation["columns_with_non_default_values"] == []
    assert report["omissions"]["loss_detected"] is False
    assert report["complete"] is True
    assert report["partial"] is False


@pytest.mark.parametrize(
    ("delivery_states", "expected_complete", "expected_loss", "expected_message"),
    [
        (("delivered",), True, False, "Recovered database verified at"),
        (
            ("pending",),
            False,
            True,
            "Recovery output verified with known omissions at",
        ),
    ],
)
def test_cli_distinguishes_verified_auxiliary_omissions(
    tmp_path: Path,
    delivery_states: tuple[str, ...],
    expected_complete: bool,
    expected_loss: bool,
    expected_message: str,
) -> None:
    state_label = delivery_states[0]
    source = tmp_path / f"cli-{state_label}-source.db"
    output = tmp_path / f"cli-{state_label}-recovered.db"
    _make_auxiliary_omission_source(
        source,
        delivery_states=delivery_states,
        origin_session_id="",
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / f"cli-{state_label}-home")
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
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_message in result.stdout
    assert "did not pass every verification check" not in result.stdout
    report_path = output.with_name(output.name + ".recovery.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is expected_complete
    assert report["verified"] is True
    assert report["omissions"]["loss_detected"] is expected_loss
    if expected_loss:
        assert "pending final replies may be lost" in result.stdout


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



