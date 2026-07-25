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
    # born in DELETE mode instead of enabling WAL.
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
