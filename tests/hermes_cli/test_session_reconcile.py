"""Operational safety tests for the Telegram orphan reconciliation service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import active_sessions
from hermes_cli.session_reconcile import reconcile_telegram_group_orphans
from hermes_state import SessionDB


def _open_group(db: SessionDB, session_id: str) -> None:
    db.create_session(
        session_id,
        "telegram",
        chat_type="group",
        chat_id=f"chat-{session_id}",
        session_key=f"key-{session_id}",
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - 7200, session_id),
    )
    db._conn.commit()


def test_apply_holds_live_lease_protection_and_creates_verified_backup(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: int(pid) == os.getpid())
    monkeypatch.setattr(active_sessions, "_process_start_time", lambda _pid: 100.0)

    db = SessionDB(home / "state.db")
    lease = None
    try:
        for session_id in ("orphan", "leased", "legacy-routed"):
            _open_group(db, session_id)
        (sessions_dir / "sessions.json").write_text(
            json.dumps(
                {
                    "telegram:legacy": {
                        "session_id": "legacy-routed",
                        "session_key": "telegram:legacy",
                    }
                }
            ),
            encoding="utf-8",
        )
        lease, message = active_sessions.try_acquire_active_session(
            session_id="leased",
            surface="gateway:telegram",
            config={"max_concurrent_sessions": 1},
        )
        assert message is None

        report = reconcile_telegram_group_orphans(
            db,
            sessions_dir=sessions_dir,
            min_age_seconds=3600,
            apply=True,
        )

        assert report["finalized_ids"] == ["orphan"]
        assert report["backup_path"]
        actual_backup = Path(report["backup_path"])
        assert actual_backup.exists()
        assert db.get_session("leased")["ended_at"] is None
        assert db.get_session("legacy-routed")["ended_at"] is None

        backup = SessionDB(actual_backup, read_only=True)
        try:
            assert backup.get_session("orphan")["ended_at"] is None
        finally:
            backup.close()
    finally:
        if lease is not None:
            lease.release()
        db.close()


def test_dry_run_does_not_create_backup_or_modify_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    try:
        _open_group(db, "orphan")
        report = reconcile_telegram_group_orphans(
            db,
            sessions_dir=sessions_dir,
            min_age_seconds=3600,
            apply=False,
        )
        assert report["candidate_ids"] == ["orphan"]
        assert report["backup_path"] is None
        assert db.get_session("orphan")["ended_at"] is None
        assert list(home.glob("state.db.backup-*")) == []
    finally:
        db.close()


def test_corrupt_live_lease_registry_fails_closed_before_backup(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    sessions_dir = home / "sessions"
    runtime = home / "runtime"
    sessions_dir.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (runtime / "active_sessions.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    try:
        _open_group(db, "orphan")
        with pytest.raises(RuntimeError, match="active session registry"):
            reconcile_telegram_group_orphans(
                db,
                sessions_dir=sessions_dir,
                min_age_seconds=3600,
                apply=True,
            )
        assert db.get_session("orphan")["ended_at"] is None
        assert list(home.glob("state.db.backup-*")) == []
    finally:
        db.close()


def test_cli_defaults_to_dry_run_then_applies_with_backup(tmp_path):
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    db = SessionDB(home / "state.db")
    try:
        _open_group(db, "cli-orphan")
    finally:
        db.close()

    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    command = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "sessions",
        "reconcile-telegram-orphans",
        "--min-age-hours",
        "1",
    ]
    dry_run = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "Candidate Telegram group sessions: 1" in dry_run.stdout
    assert "Dry run" in dry_run.stdout
    assert list(home.glob("state.db.backup-*")) == []

    applied = subprocess.run(
        [*command, "--apply"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "Verified backup:" in applied.stdout
    assert "Finalized 1 session(s)" in applied.stdout
    assert len(list(home.glob("state.db.backup-*"))) == 1

    verified = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "No orphaned Telegram group sessions found." in verified.stdout
    reopened = SessionDB(home / "state.db")
    try:
        assert reopened.get_session("cli-orphan")["end_reason"] == "orphan_reconcile"
    finally:
        reopened.close()
