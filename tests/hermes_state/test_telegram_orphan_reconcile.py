"""Safety contracts for transactional Telegram group orphan reconciliation."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_state import SessionDB


def _open_group(db: SessionDB, session_id: str, *, source: str = "telegram", age: float = 7200) -> None:
    db.create_session(
        session_id,
        source,
        session_key=f"route:{session_id}",
        chat_id=f"chat:{session_id}",
        chat_type="group",
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - age, session_id),
    )
    db._conn.commit()


def test_reconcile_only_finalizes_old_unrouted_unprotected_telegram_groups(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for session_id in ("orphan", "routed", "leased", "fresh", "discord"):
            _open_group(
                db,
                session_id,
                source="discord" if session_id == "discord" else "telegram",
                age=10 if session_id == "fresh" else 7200,
            )
        db.create_session("dm", "telegram", chat_type="dm", chat_id="dm-chat")
        db.save_gateway_routing_entry(
            "telegram:group:routed",
            json.dumps({"session_id": "routed"}),
            scope="/profile/sessions",
        )

        preview = db.reconcile_orphaned_telegram_group_sessions(
            protected_session_ids={"leased"},
            min_age_seconds=3600,
            apply=False,
        )
        assert preview == {
            "candidate_ids": ["orphan"],
            "finalized_ids": [],
            "dry_run": True,
        }
        assert db.get_session("orphan")["ended_at"] is None

        applied = db.reconcile_orphaned_telegram_group_sessions(
            protected_session_ids={"leased"},
            min_age_seconds=3600,
            apply=True,
        )
        assert applied == {
            "candidate_ids": ["orphan"],
            "finalized_ids": ["orphan"],
            "dry_run": False,
        }
        assert db.get_session("orphan")["end_reason"] == "orphan_reconcile"
        for protected in ("routed", "leased", "fresh", "discord", "dm"):
            assert db.get_session(protected)["ended_at"] is None

        second = db.reconcile_orphaned_telegram_group_sessions(
            protected_session_ids={"leased"},
            min_age_seconds=3600,
            apply=True,
        )
        assert second["candidate_ids"] == []
        assert second["finalized_ids"] == []
    finally:
        db.close()


def test_reconcile_fails_closed_on_malformed_route_and_rolls_back(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _open_group(db, "orphan-a")
        _open_group(db, "orphan-b")
        db.save_gateway_routing_entry("broken", "{not-json", scope="scope")

        with pytest.raises(ValueError, match="malformed gateway routing"):
            db.reconcile_orphaned_telegram_group_sessions(
                protected_session_ids=set(), min_age_seconds=3600, apply=True
            )
        assert db.get_session("orphan-a")["ended_at"] is None
        assert db.get_session("orphan-b")["ended_at"] is None

        db._conn.execute("DELETE FROM gateway_routing")
        db._conn.execute(
            """CREATE TRIGGER abort_orphan_b
               BEFORE UPDATE OF ended_at ON sessions
               WHEN NEW.id = 'orphan-b' AND NEW.ended_at IS NOT NULL
               BEGIN SELECT RAISE(ABORT, 'test rollback'); END"""
        )
        db._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="test rollback"):
            db.reconcile_orphaned_telegram_group_sessions(
                protected_session_ids=set(), min_age_seconds=3600, apply=True
            )
        assert db.get_session("orphan-a")["ended_at"] is None
        assert db.get_session("orphan-b")["ended_at"] is None
    finally:
        db.close()


def test_online_backup_is_consistent_and_restorable(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _open_group(db, "orphan")
        backup_path = db.create_verified_backup("telegram-orphan-reconcile")
        assert backup_path.exists()
        assert backup_path.stat().st_mode & 0o777 == 0o600

        restored = SessionDB(backup_path, read_only=True)
        try:
            assert restored.get_session("orphan")["ended_at"] is None
            assert restored._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            restored.close()
    finally:
        db.close()
