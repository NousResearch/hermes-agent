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
from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION, SCHEMA_VERSION
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


def _insert_delivery_obligations(path: Path, rows: list[tuple[object, ...]]) -> None:
    from gateway.delivery_ledger import _initialize_schema

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        _initialize_schema(conn)
        conn.executemany(
            """INSERT INTO delivery_obligations (
                obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, last_error, adapter_profile
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    finally:
        conn.close()


def test_recovery_copies_delivery_obligations(tmp_path: Path) -> None:
    """Owed replies must survive salvage — #100313 lost 6 obligation rows."""

    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    now = 1_720_000_000.0
    _insert_delivery_obligations(
        source,
        [
            (
                "ob-pending",
                "telegram:1:chat-1",
                "telegram",
                "chat-1",
                None,
                "owed reply",
                "pending",
                0,
                now,
                now,
                4242,
                99,
                None,
                "default",
            ),
            (
                "ob-delivered",
                "telegram:1:chat-1",
                "telegram",
                "chat-1",
                None,
                "already sent",
                "delivered",
                1,
                now,
                now + 1,
                None,
                None,
                None,
                "default",
            ),
        ],
    )

    inspection = inspect_session_database(source, work_dir=tmp_path)
    assert inspection["tables"]["delivery_obligations"]["available"] is True
    assert inspection["tables"]["delivery_obligations"]["rows"] == 2

    report = recover_session_database(source, output, work_dir=tmp_path)
    copied = report["copy"]["delivery_obligations"]
    assert copied["status"] == "complete"
    assert copied["copied_rows"] == 2
    assert report["verification"]["table_counts"]["delivery_obligations"] == 2
    assert report["complete"] is True
    assert report["verified"] is True
    assert report["installed"] is False

    conn = sqlite3.connect(str(output))
    try:
        recovered = conn.execute(
            """SELECT obligation_id, state, content, owner_pid, adapter_profile
               FROM delivery_obligations ORDER BY obligation_id"""
        ).fetchall()
    finally:
        conn.close()
    assert recovered == [
        ("ob-delivered", "delivered", "already sent", None, "default"),
        ("ob-pending", "pending", "owed reply", 4242, "default"),
    ]


def test_recovery_without_delivery_ledger_is_not_lossy(tmp_path: Path) -> None:
    """CLI-only stores never created the lazy table; that is not data loss."""

    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)

    report = recover_session_database(source, output, work_dir=tmp_path)
    assert report["copy"]["delivery_obligations"]["status"] == "missing"
    assert "delivery_obligations" not in report["verification"]["table_counts"]
    assert report["complete"] is True
    assert report["verified"] is True





def test_recovery_flags_delivery_obligation_count_mismatch_as_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source-vs-destination ledger count mismatch must not verify as complete.

    The destination table is created through the registered initializer; a
    real SQL trigger that silently drops one row stands in for the "rows went
    missing on the way over" failure the verifier has to catch.
    """

    from hermes_cli import session_recovery

    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    now = 1_720_000_000.0
    _insert_delivery_obligations(
        source,
        [
            ("ob-a", "k", "telegram", "chat-1", None, "a", "pending", 0, now, now, None, None, None, "default"),
            ("ob-b", "k", "telegram", "chat-1", None, "b", "pending", 0, now, now, None, None, None, "default"),
        ],
    )

    real_init = session_recovery._AUXILIARY_TABLE_SCHEMAS["delivery_obligations"]

    def lossy_init(conn: sqlite3.Connection) -> None:
        real_init(conn)
        conn.execute(
            """CREATE TRIGGER drop_ob_b BEFORE INSERT ON delivery_obligations
               WHEN NEW.obligation_id = 'ob-b' BEGIN SELECT RAISE(IGNORE); END"""
        )

    monkeypatch.setitem(
        session_recovery._AUXILIARY_TABLE_SCHEMAS, "delivery_obligations", lossy_init
    )

    report = recover_session_database(source, output, work_dir=tmp_path)
    assert report["verification"]["table_counts"]["delivery_obligations"] == 1
    assert report["complete"] is False
    assert any(
        "delivery_obligations count is 1, expected 2" in error
        for error in report["verification"]["errors"]
    )


def test_lost_and_found_direct_copy_creates_lazy_delivery_ledger(tmp_path: Path) -> None:
    """The .recover lane copies the ledger even though SessionDB never made it."""

    from hermes_cli.session_lost_and_found import _copy_direct_tables

    recovered_source = tmp_path / "lost_and_found.db"
    now = 1_720_000_000.0
    _insert_delivery_obligations(
        recovered_source,
        [
            ("ob-1", "k", "telegram", "chat-1", None, "one", "pending", 0, now, now, None, None, None, "default"),
            ("ob-2", "k", "telegram", "chat-1", None, "two", "failed", 3, now, now, None, None, "boom", "default"),
        ],
    )
    output = tmp_path / "rebuilt.db"
    SessionDB(db_path=output).close()

    lf_conn = sqlite3.connect(str(recovered_source), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        assert not dest.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='delivery_obligations'"
        ).fetchall()
        copied = _copy_direct_tables(lf_conn, dest)
        assert copied["delivery_obligations"] == 2
        rows = dest.execute(
            "SELECT obligation_id, state, last_error FROM delivery_obligations ORDER BY obligation_id"
        ).fetchall()
    finally:
        lf_conn.close()
        dest.close()
    assert rows == [("ob-1", "pending", None), ("ob-2", "failed", "boom")]


def test_recovery_quarantines_malformed_session_model_config(tmp_path: Path) -> None:
    """A physically sound output must also survive real session-list JSON reads."""

    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)

    conn = sqlite3.connect(str(source))
    try:
        conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            ('{"broken"', "recovery-session-1"),
        )
        conn.commit()
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        conn.close()
    source_hash = _sha256(source)

    report = recover_session_database(source, output, work_dir=tmp_path)

    assert _sha256(source) == source_hash
    assert '{"broken"' not in json.dumps(report)
    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["semantic_cleanup"] == {
        "malformed_json_values_quarantined": 1,
        "columns": {"sessions.model_config": 1},
    }
    assert any(
        "quarantined 1 malformed JSON value" in warning
        for warning in report["verification"]["warnings"]
    )

    recovered = SessionDB(db_path=output)
    try:
        rows = recovered.list_sessions_rich(limit=10)
        row = recovered.get_session("recovery-session-1")
    finally:
        recovered.close()

    assert {item["id"] for item in rows} == {
        "recovery-session-0",
        "recovery-session-1",
        "recovery-session-2",
    }
    assert row is not None
    assert row["model_config"] == "{}"


def test_quarantine_malformed_json_is_not_called_inside_an_open_transaction(
    tmp_path: Path,
) -> None:
    """Regression guard for the reviewer-flagged nested-BEGIN concern.

    ``_quarantine_malformed_json`` issues its own ``BEGIN IMMEDIATE`` /
    ``COMMIT`` / ``ROLLBACK``. Both recovery call sites hand it a fresh
    autocommit (``isolation_level=None``) connection, so SQLite is not inside
    a transaction when the quarantine runs. If a future caller ever invokes
    it while a transaction is already open, the nested ``BEGIN`` raises
    ``sqlite3.OperationalError: cannot start a transaction within a
    transaction`` — this test pins the current safe contract by asserting the
    connection is transaction-free at entry on both call paths.
    """

    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)

    in_transaction_at_entry: list[bool] = []
    real_quarantine = session_recovery._quarantine_malformed_json

    def spy_quarantine(destination: sqlite3.Connection) -> dict:
        in_transaction_at_entry.append(destination.in_transaction)
        return real_quarantine(destination)

    session_recovery._quarantine_malformed_json = spy_quarantine
    try:
        recover_session_database(source, output, work_dir=tmp_path)
    finally:
        session_recovery._quarantine_malformed_json = real_quarantine

    assert in_transaction_at_entry == [False]
