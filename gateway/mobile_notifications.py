"""SQLite-backed mobile notification inbox for Hermes WebUI clients."""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OPEN_STATUSES = ("unread", "read")
CLOSED_STATUSES = ("closed", "expired")
VALID_NOTIFICATION_STATUSES = OPEN_STATUSES + CLOSED_STATUSES
VALID_ACTION_STATUSES = ("pending", "done", "rejected")


class MobileNotificationStore:
    """Persist mobile notification inbox items and approval-style actions."""

    def __init__(self, db_path: Optional[str] = None):
        if not db_path:
            try:
                from hermes_cli.config import get_hermes_home

                db_path = str(get_hermes_home() / "mobile_notifications.db")
            except Exception:
                db_path = ":memory:"

        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(self._conn, db_label="mobile_notifications.db")
        self._init_schema()
        self._tighten_file_permissions()

    def _init_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS mobile_notifications (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                detail_ref TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'unread',
                expires_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS mobile_notification_actions (
                id TEXT PRIMARY KEY,
                notification_id TEXT NOT NULL,
                group_key TEXT NOT NULL,
                label TEXT NOT NULL,
                value TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(notification_id)
                    REFERENCES mobile_notifications(id)
                    ON DELETE CASCADE
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mobile_notifications_status "
            "ON mobile_notifications(status, expires_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mobile_notification_actions_parent "
            "ON mobile_notification_actions(notification_id, group_key, status)"
        )
        self._conn.commit()

    def _tighten_file_permissions(self) -> None:
        if not self._db_path:
            return
        for candidate in (
            Path(self._db_path),
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                pass

    def upsert_notification(
        self,
        notification: Dict[str, Any],
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        notification_id = self._clean_text(
            notification.get("id") or f"mn_{uuid.uuid4().hex}", "id", 128
        )
        kind = self._clean_text(notification.get("kind") or "general", "kind", 64)
        title = self._clean_text(notification.get("title"), "title", 200)
        body = self._clean_text(notification.get("body"), "body", 1000)
        detail_ref = self._clean_text(
            notification.get("detail_ref") or "", "detail_ref", 1000, required=False
        )
        status = notification.get("status") or "unread"
        if status not in VALID_NOTIFICATION_STATUSES:
            raise ValueError("Invalid notification status")
        expires_at = notification.get("expires_at")
        if expires_at is not None:
            expires_at = float(expires_at)
        created_at = float(notification.get("created_at") or now)

        self._conn.execute(
            """INSERT INTO mobile_notifications (
                id, kind, title, body, detail_ref, status, expires_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind,
                title=excluded.title,
                body=excluded.body,
                detail_ref=excluded.detail_ref,
                status=excluded.status,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at""",
            (
                notification_id,
                kind,
                title,
                body,
                detail_ref,
                status,
                expires_at,
                created_at,
                now,
            ),
        )

        if actions is not None:
            self._conn.execute(
                "DELETE FROM mobile_notification_actions WHERE notification_id = ?",
                (notification_id,),
            )
            for raw_action in actions:
                action_id = self._clean_text(
                    raw_action.get("id") or f"act_{uuid.uuid4().hex}", "action_id", 128
                )
                group_key = self._clean_text(
                    raw_action.get("group_key") or "default", "group_key", 128
                )
                label = self._clean_text(raw_action.get("label"), "label", 100)
                value = self._clean_text(raw_action.get("value"), "value", 500)
                action_status = raw_action.get("status") or "pending"
                if action_status not in VALID_ACTION_STATUSES:
                    raise ValueError("Invalid action status")
                self._conn.execute(
                    """INSERT INTO mobile_notification_actions (
                        id, notification_id, group_key, label, value, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        action_id,
                        notification_id,
                        group_key,
                        label,
                        value,
                        action_status,
                        now,
                        now,
                    ),
                )

        self._conn.commit()
        self._tighten_file_permissions()
        return self.get_notification(notification_id) or {}

    def list_notifications(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        self.expire_due()
        if status == "open":
            rows = self._conn.execute(
                """SELECT * FROM mobile_notifications
                WHERE status IN ('unread', 'read')
                ORDER BY created_at DESC
                LIMIT ?""",
                (limit,),
            ).fetchall()
        elif status == "all":
            rows = self._conn.execute(
                """SELECT * FROM mobile_notifications
                ORDER BY created_at DESC
                LIMIT ?""",
                (limit,),
            ).fetchall()
        else:
            raise ValueError("Invalid status filter")
        return [self._row_to_notification(row) for row in rows]

    def get_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        self.expire_due()
        row = self._conn.execute(
            "SELECT * FROM mobile_notifications WHERE id = ?",
            (notification_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_notification(row)

    def mark_read(self, notification_id: str) -> Optional[Dict[str, Any]]:
        self.expire_due()
        row = self._conn.execute(
            "SELECT status FROM mobile_notifications WHERE id = ?",
            (notification_id,),
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "unread":
            self._conn.execute(
                "UPDATE mobile_notifications SET status = 'read', updated_at = ? WHERE id = ?",
                (time.time(), notification_id),
            )
            self._conn.commit()
        return self.get_notification(notification_id)

    def resolve_action(
        self,
        notification_id: str,
        action_id: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        self.expire_due()
        notification = self.get_notification(notification_id)
        if notification is None:
            return "not_found", None
        if notification["status"] == "expired":
            return "expired", notification

        action_row = self._conn.execute(
            """SELECT * FROM mobile_notification_actions
            WHERE notification_id = ? AND id = ?""",
            (notification_id, action_id),
        ).fetchone()
        if action_row is None:
            return "not_found", notification
        if action_row["status"] == "done":
            return "ok", notification
        if action_row["status"] != "pending":
            return "conflict", notification

        done_sibling = self._conn.execute(
            """SELECT id FROM mobile_notification_actions
            WHERE notification_id = ? AND group_key = ? AND status = 'done'
            LIMIT 1""",
            (notification_id, action_row["group_key"]),
        ).fetchone()
        if done_sibling is not None:
            return "conflict", notification

        now = time.time()
        self._conn.execute(
            """UPDATE mobile_notification_actions
            SET status = 'done', updated_at = ?
            WHERE notification_id = ? AND id = ?""",
            (now, notification_id, action_id),
        )
        self._conn.execute(
            """UPDATE mobile_notification_actions
            SET status = 'rejected', updated_at = ?
            WHERE notification_id = ? AND group_key = ? AND id != ? AND status = 'pending'""",
            (now, notification_id, action_row["group_key"], action_id),
        )

        pending = self._conn.execute(
            """SELECT COUNT(*) FROM mobile_notification_actions
            WHERE notification_id = ? AND status = 'pending'""",
            (notification_id,),
        ).fetchone()[0]
        if pending == 0:
            self._conn.execute(
                "UPDATE mobile_notifications SET status = 'closed', updated_at = ? WHERE id = ?",
                (now, notification_id),
            )
        self._conn.commit()
        return "ok", self.get_notification(notification_id)

    def expire_due(self, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        rows = self._conn.execute(
            """SELECT id FROM mobile_notifications
            WHERE status IN ('unread', 'read') AND expires_at IS NOT NULL AND expires_at <= ?""",
            (now,),
        ).fetchall()
        if not rows:
            return 0
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"UPDATE mobile_notifications SET status = 'expired', updated_at = ? WHERE id IN ({placeholders})",
            [now] + ids,
        )
        self._conn.execute(
            f"UPDATE mobile_notification_actions SET status = 'rejected', updated_at = ? "
            f"WHERE notification_id IN ({placeholders}) AND status = 'pending'",
            [now] + ids,
        )
        self._conn.commit()
        return len(ids)

    def _row_to_notification(self, row: sqlite3.Row) -> Dict[str, Any]:
        actions = [
            {
                "id": action["id"],
                "group_key": action["group_key"],
                "label": action["label"],
                "value": action["value"],
                "status": action["status"],
                "created_at": action["created_at"],
                "updated_at": action["updated_at"],
            }
            for action in self._conn.execute(
                """SELECT * FROM mobile_notification_actions
                WHERE notification_id = ?
                ORDER BY created_at ASC, id ASC""",
                (row["id"],),
            ).fetchall()
        ]
        return {
            "id": row["id"],
            "kind": row["kind"],
            "title": row["title"],
            "body": row["body"],
            "detail_ref": row["detail_ref"],
            "status": row["status"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "actions": actions,
        }

    def _clean_text(
        self,
        value: Any,
        field_name: str,
        max_length: int,
        required: bool = True,
    ) -> str:
        if value is None:
            if required:
                raise ValueError(f"{field_name} is required")
            return ""
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if required and not value:
            raise ValueError(f"{field_name} is required")
        if len(value) > max_length:
            raise ValueError(f"{field_name} exceeds {max_length} chars")
        return value

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
