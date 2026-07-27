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


def test_partial_recovery_reports_damaged_state_meta_as_loss(
    tmp_path: Path,
) -> None:
    """A damaged optional state_meta must degrade AND be reported as loss.

    state_meta can lose ``key`` while ``value`` stays readable. Indexing
    ``key`` unconditionally raised ValueError and killed the whole partial
    recovery. Reporting it as ``missing`` instead is just as wrong in the
    other direction: verification only escalates ``failed``/``partial``, so
    the run would claim ``complete=True`` after silently dropping real
    metadata. A present-but-unusable table must be ``failed``, surface a
    warning, and mark the output partial — while sessions and messages still
    recover.
    """
    source = tmp_path / "meta-damaged.db"
    output = tmp_path / "meta-recovered.db"
    expected = _make_source(source)

    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        conn.execute("DROP TABLE IF EXISTS state_meta")
        conn.execute("CREATE TABLE state_meta(value TEXT)")
        conn.execute("INSERT INTO state_meta(value) VALUES ('orphaned')")
    finally:
        conn.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=4,
        allow_partial=True,
    )

    # The table existed and could not be salvaged: that is loss, not absence.
    assert report["copy"]["state_meta"]["status"] == "failed"
    assert any(
        "state_meta" in warning for warning in report["verification"]["warnings"]
    ), report["verification"]["warnings"]
    assert report["verification"]["loss_detected"] is True
    assert report["partial"] is True
    assert report["complete"] is False

    # Structurally sound, so still installable-with-review.
    assert report["verified"] is True
    assert report["verification"]["healthy"] is True
    assert report["installed"] is False

    # The canonical data still recovers.
    assert report["verification"]["table_counts"]["sessions"] == expected["sessions"]
    assert report["verification"]["table_counts"]["messages"] == expected["messages"]
    assert report["verification"]["integrity_check"] == ["ok"]


def test_partial_recovery_treats_absent_state_meta_as_no_loss(
    tmp_path: Path,
) -> None:
    """A genuinely absent state_meta is not data loss and must not warn."""
    source = tmp_path / "meta-absent.db"
    output = tmp_path / "meta-absent-recovered.db"
    _make_source(source)

    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        conn.execute("DROP TABLE IF EXISTS state_meta")
    finally:
        conn.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=4,
        allow_partial=True,
    )

    assert report["copy"]["state_meta"]["status"] == "missing"
    assert not any(
        "state_meta" in warning for warning in report["verification"]["warnings"]
    ), report["verification"]["warnings"]
    assert report["verification"]["integrity_check"] == ["ok"]


def test_state_meta_salvage_distinguishes_absent_from_unusable(
    tmp_path: Path,
) -> None:
    """Unit-level: the salvage helper must not conflate absent with damaged.

    ``recover_session_database`` short-circuits on the inspection result when
    state_meta is entirely absent, so the helper's own absent-branch is never
    reached through the public path — a regression there would be invisible
    end-to-end. Exercise it directly so both statuses are pinned:
    absent -> ``missing`` (no loss), present-but-unusable -> ``failed``
    (loss, which verification escalates into a warning).
    """
    destination_path = tmp_path / "dest.db"
    destination = sqlite3.connect(str(destination_path), isolation_level=None)
    destination.execute("CREATE TABLE state_meta(key TEXT PRIMARY KEY, value TEXT)")

    absent_path = tmp_path / "absent.db"
    absent = sqlite3.connect(str(absent_path), isolation_level=None)
    absent.execute("CREATE TABLE unrelated(x)")

    damaged_path = tmp_path / "damaged.db"
    damaged = sqlite3.connect(str(damaged_path), isolation_level=None)
    damaged.execute("CREATE TABLE state_meta(value TEXT)")
    damaged.execute("INSERT INTO state_meta(value) VALUES ('orphaned')")

    try:
        absent_result = session_recovery._copy_state_meta_salvage(
            absent, destination, chunk_size=4, progress_cb=None, source_rows=0
        )
        assert absent_result["status"] == "missing", (
            "a table that was never there is not data loss"
        )

        damaged_result = session_recovery._copy_state_meta_salvage(
            damaged, destination, chunk_size=4, progress_cb=None, source_rows=1
        )
        assert damaged_result["status"] == "failed", (
            "a present-but-unusable table is data loss; reporting 'missing' "
            "would let verification claim complete=True after dropping it"
        )
    finally:
        destination.close()
        absent.close()
        damaged.close()


def test_recovery_refuses_to_snapshot_a_live_database(tmp_path: Path) -> None:
    """Snapshotting must refuse while a connection to the source is live.

    Copying a database file is an open()/close() on it, and close() cancels
    every POSIX advisory lock the process holds on that file (see
    hermes_cli.sqlite_safe_read). Recovery normally runs against an offline
    file, but the check must exist so this path cannot drift away from
    hermes_state._backup_db_file, which refuses the same situation.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    source = tmp_path / "live-state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)

    live = connect_tracked(source, isolation_level=None)
    try:
        with pytest.raises(SessionRecoverySafetyError, match="still open"):
            recover_session_database(source, output, work_dir=tmp_path)
    finally:
        live.close()

    assert not output.exists()

    # With the connection closed the same call proceeds normally.
    report = recover_session_database(source, output, work_dir=tmp_path)
    assert report["verification"]["integrity_check"] == ["ok"]


def test_recovery_rebuilds_canonical_data_without_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "damaged-state.db"
    output = tmp_path / "recovered-state.db"
    expected = _make_source(source)
    _orphan_fts_schema(source)

    source_hash = _sha256(source)
    source_stat = source.stat()

    # Exercise the vulnerable-runtime fallback: a fresh recovered DB must be
    # born in DELETE mode instead of enabling WAL (#70055, retained).
    monkeypatch.setattr(
        hermes_state,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: True,
    )

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=4,
    )

    assert report["complete"] is True
    assert report["installed"] is False
    assert report["source_unchanged"] is True
    assert report["verification"]["journal_mode"] == "delete"
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verification"]["schema_version"] == SCHEMA_VERSION
    assert report["verification"]["pending_fts_keys"] == []
    assert report["verification"]["table_counts"]["sessions"] == expected["sessions"]
    assert report["verification"]["table_counts"]["messages"] == expected["messages"]
    assert report["copy"]["messages"]["copied_rows"] == expected["messages"]
    disk_space = report["disk_space"]
    assert disk_space["estimated_output_bytes"] == disk_space["source_bundle_bytes"]
    assert disk_space["work_dir_required_bytes"] == (
        disk_space["source_bundle_bytes"]
        + disk_space["estimated_output_bytes"]
        + disk_space["headroom_bytes"]
    )

    assert _sha256(source) == source_hash
    assert source.stat().st_size == source_stat.st_size
    assert source.stat().st_mtime_ns == source_stat.st_mtime_ns

    conn = sqlite3.connect(str(output))
    try:
        assert (
            conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                ("goal:recovery-session-0",),
            ).fetchone()[0]
            == '{"status":"active"}'
        )
        assert conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_version'"
        ).fetchone()[0] == str(FTS_STORAGE_VERSION)
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM state_meta "
                "WHERE key IN ('fts_rebuild_high_water', 'fts_rebuild_progress')"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM telegram_dm_topic_mode").fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM telegram_dm_topic_bindings").fetchone()[
                0
            ]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM gateway_routing").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts "
                "WHERE messages_fts MATCH 'recoverable'"
            ).fetchone()[0]
            == expected["messages"]
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts_trigram "
                "WHERE messages_fts_trigram MATCH 'cover'"
            ).fetchone()[0]
            == expected["messages"]
        )
    finally:
        conn.close()

    reopened = SessionDB(db_path=output)
    try:
        message_id = reopened.append_message(
            "recovery-session-0",
            "user",
            "post recovery write",
        )
        assert message_id > expected["messages"]
        assert reopened.search_messages("post recovery write")
    finally:
        reopened.close()


def test_recovery_refuses_overwrite_and_source_alias(tmp_path: Path) -> None:
    source = tmp_path / "state.db"
    _make_source(source)

    output = tmp_path / "existing.db"
    output.write_bytes(b"keep me")
    with pytest.raises(SessionRecoverySafetyError, match="overwrite"):
        recover_session_database(source, output, work_dir=tmp_path)
    assert output.read_bytes() == b"keep me"

    with pytest.raises(SessionRecoverySafetyError, match="must not be the source"):
        recover_session_database(source, source, work_dir=tmp_path)


def test_recovery_refuses_before_writing_when_disk_space_is_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    monkeypatch.setattr(
        session_recovery.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(SessionRecoverySafetyError, match="Not enough free disk space"):
        recover_session_database(source, output, work_dir=tmp_path)
    assert not output.exists()
    assert not list(tmp_path.glob("hermes-session-recovery-*"))


def test_recovery_requires_readable_sessions_and_messages(tmp_path: Path) -> None:
    source = tmp_path / "missing-messages.db"
    output = tmp_path / "recovered.db"
    conn = sqlite3.connect(str(source))
    try:
        conn.execute(
            "CREATE TABLE sessions "
            "(id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO sessions(id, source, started_at) VALUES ('s1', 'cli', 1)"
        )
        conn.commit()
    finally:
        conn.close()

    inspection = inspect_session_database(source, work_dir=tmp_path)
    assert inspection["recoverable"] is False
    with pytest.raises(SessionRecoverySourceError, match="messages"):
        recover_session_database(source, output, work_dir=tmp_path)
    assert not output.exists()


def test_allow_partial_still_reports_a_complete_healthy_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "healthy-state.db"
    output = tmp_path / "healthy-recovered.db"
    expected = _make_source(source)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=4,
        allow_partial=True,
    )

    assert report["verified"] is True
    assert report["complete"] is True
    assert report["partial"] is False
    assert report["verification"]["warnings"] == []
    assert report["verification"]["table_counts"]["sessions"] == expected["sessions"]
    assert report["verification"]["table_counts"]["messages"] == expected["messages"]
    assert report["orphan_cleanup"]["total_removed_or_relinked"] == 0


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


def test_partial_recovery_reconstructs_unreadable_sessions_without_message_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt-sessions.db"
    output = tmp_path / "partial-sessions.db"
    session_count = 180
    sessions_root = _make_many_sessions_source(
        source,
        session_count,
    )
    _corrupt_middle_table_leaf(source, sessions_root)
    source_hash = _sha256(source)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["source_unchanged"] is True
    assert _sha256(source) == source_hash
    assert report["copy"]["sessions"]["status"] == "partial"
    assert report["copy"]["messages"]["status"] == "complete"
    copied_sessions = int(report["copy"]["sessions"]["copied_rows"])
    expected_reconstructed = session_count - copied_sessions
    cleanup = report["orphan_cleanup"]
    assert expected_reconstructed > 1
    assert cleanup["sessions_reconstructed"] == expected_reconstructed
    assert cleanup["messages_retained"] == expected_reconstructed
    assert cleanup["messages_removed"] == 0
    assert report["verification"]["foreign_key_check"] == []
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["table_counts"]["sessions"] == session_count
    assert report["verification"]["table_counts"]["messages"] == session_count

    conn = sqlite3.connect(str(output))
    try:
        recovered_sessions = conn.execute(
            "SELECT id, source, title FROM sessions ORDER BY id"
        ).fetchall()
        recovered_ids = {str(row[0]) for row in recovered_sessions}
        assert recovered_ids == {
            f"partial-session-{session_number:04d}"
            for session_number in range(session_count)
        }
        placeholder_rows = [
            row for row in recovered_sessions if str(row[1]) == "recovered"
        ]
        assert len(placeholder_rows) == expected_reconstructed
        placeholder_titles = [str(row[2]) for row in placeholder_rows]
        assert len(set(placeholder_titles)) == expected_reconstructed
        assert all(
            title.startswith("[recovered ") for title in placeholder_titles
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages AS message "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM sessions "
                "WHERE sessions.id = message.session_id)"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            == session_count
        )
    finally:
        conn.close()


def test_partial_recovery_verifies_fully_reconstructed_sessions(
    tmp_path: Path,
) -> None:
    """A failed sessions copy is recoverable when every parent is rebuilt.

    Reported July 2026: all 194 original session rows were unreadable, but
    partial recovery reconstructed 194 placeholders, retained 20,817 readable
    messages, removed none, and produced clean integrity, FK, and FTS checks.
    The verifier still treated ``sessions copy status is failed`` as fatal and
    printed "Do not install it" for a structurally healthy partial output.
    """
    source = tmp_path / "sessions-root-destroyed.db"
    output = tmp_path / "sessions-root-destroyed-recovered.db"
    session_count = 60
    sessions_root = _make_many_sessions_source(source, session_count)
    _corrupt_table_root(source, sessions_root)
    source_hash = _sha256(source)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    assert report["copy"]["sessions"]["status"] == "failed"
    assert report["copy"]["messages"]["status"] == "complete"
    cleanup = report["orphan_cleanup"]
    assert cleanup["sessions_reconstructed"] == session_count
    assert cleanup["messages_retained"] == session_count
    assert cleanup["messages_removed"] == 0

    verification = report["verification"]
    assert verification["errors"] == []
    assert "sessions copy status is failed" in verification["warnings"]
    assert verification["integrity_check"] == ["ok"]
    assert verification["foreign_key_check"] == []
    assert verification["table_counts"]["sessions"] == session_count
    assert verification["table_counts"]["messages"] == session_count
    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["source_unchanged"] is True
    assert _sha256(source) == source_hash

    with sqlite3.connect(str(output)) as conn:
        recovered = conn.execute(
            "SELECT source, COUNT(*) FROM sessions GROUP BY source"
        ).fetchall()
    assert recovered == [("recovered", session_count)]


def test_cli_recover_writes_verified_report_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "state.db"
    output = tmp_path / "recovered.db"
    expected = _make_source(source)
    source_hash = _sha256(source)
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
            "5",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "The active session database was not changed." in result.stdout
    assert _sha256(source) == source_hash
    report_path = output.with_name(output.name + ".recovery.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["verification"]["table_counts"]["sessions"] == expected["sessions"]
    assert report["verification"]["table_counts"]["messages"] == expected["messages"]


def test_failed_repair_points_the_user_at_offline_recovery(tmp_path: Path) -> None:
    """A failed in-place repair must name the offline recovery command.

    This is the reported dead end: `hermes sessions repair` exhausts its
    in-place ladder, prints "keep the files", and stops -- leaving the user
    with no idea that a non-destructive recovery path exists. The failure
    branch has to hand them the next command, inspection first, seeded with
    the preserved backup it just made.
    """
    hermes_home = tmp_path / "isolated-hermes-home"
    hermes_home.mkdir()
    # Not a database at all -> every in-place repair strategy fails.
    (hermes_home / "state.db").write_bytes(b"SQLite format 3\x00" + b"\x7f" * 2048)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "sessions", "repair"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "Repair failed" in combined, combined
    # The pointer, and specifically the read-only step first.
    assert "sessions recover" in combined, combined
    assert "--inspect-only" in combined, combined
    # It must offer the source it actually preserved, not a placeholder.
    assert "--source" in combined, combined
    assert "malformed-backup" in combined, combined
