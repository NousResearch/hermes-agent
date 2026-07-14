"""Durable background-task and sidequest ledger for gateway sessions.

The gateway's /background command creates real async work, but without a
persistent handle users cannot later ask for status, artifacts, or follow-up.
This module keeps a small SQLite ledger keyed by platform/chat ownership so
messaging gateways can safely address background runs and promoted sidequests.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


class SidequestStore:
    """SQLite-backed ledger for background runs and durable sidequests."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_hermes_home() / "quests" / "quests.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS background_runs (
                    bg_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    latest_summary TEXT,
                    artifact_paths TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS quests (
                    quest_id TEXT PRIMARY KEY,
                    alias INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    user_id TEXT,
                    source_bg_id TEXT,
                    latest_summary TEXT,
                    artifact_paths TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived_at REAL,
                    UNIQUE(platform, chat_id, alias)
                );

                CREATE TABLE IF NOT EXISTS quest_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quest_followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    user_id TEXT,
                    created_at REAL NOT NULL,
                    consumed_at REAL
                );
                """
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _loads(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _row(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        data = dict(row)
        data["artifact_paths"] = SidequestStore._loads(data.get("artifact_paths"), [])
        return data

    def _record_event(self, conn: sqlite3.Connection, target_type: str, target_id: str,
                      event_type: str, payload: Optional[dict[str, Any]] = None) -> None:
        conn.execute(
            """
            INSERT INTO quest_events(target_type, target_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (target_type, target_id, event_type, self._json(payload or {}), time.time()),
        )

    def create_background_run(self, *, bg_id: str, prompt: str, platform: str,
                              chat_id: str, user_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO background_runs(
                    bg_id, prompt, status, platform, chat_id, user_id, session_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (bg_id, prompt, "queued", platform, chat_id, user_id, session_id, now, now),
            )
            self._record_event(conn, "background", bg_id, "background.created", {"prompt": prompt})
        return self.get_background_run(bg_id, platform=platform, chat_id=chat_id) or {}

    def get_background_run(self, bg_id: str, *, platform: str, chat_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM background_runs
                WHERE bg_id = ? AND platform = ? AND chat_id = ?
                """,
                (bg_id, platform, chat_id),
            ).fetchone()
        return self._row(row)

    def list_background_runs(self, *, platform: str, chat_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM background_runs
                WHERE platform = ? AND chat_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (platform, chat_id, limit),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def reconcile_incomplete_runs(
        self,
        reason: str = "gateway restarted before task completed",
    ) -> int:
        """Fail queued/running rows left behind by a previous gateway process."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT bg_id FROM background_runs WHERE status IN ('queued', 'running')"
            ).fetchall()
        for row in rows:
            self.mark_failed(row["bg_id"], reason)
        return len(rows)

    def mark_running(self, bg_id: str) -> None:
        self._set_background_status(bg_id, "running")

    def mark_failed(self, bg_id: str, error: str) -> None:
        self._set_background_status(bg_id, "failed", summary=error)
        self._update_linked_quests(bg_id, status="blocked", summary=error)

    def mark_completed(self, bg_id: str, *, summary: str,
                       artifact_paths: Optional[list[str]] = None) -> None:
        now = time.time()
        artifacts = artifact_paths or []
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_runs
                SET status = 'completed', latest_summary = ?, artifact_paths = ?,
                    updated_at = ?, completed_at = ?
                WHERE bg_id = ?
                """,
                (summary, self._json(artifacts), now, now, bg_id),
            )
            self._record_event(conn, "background", bg_id, "run.completed", {"summary": summary})
        self._update_linked_quests(bg_id, status="waiting", summary=summary, artifact_paths=artifacts)

    def attach_background_to_quest(self, *, quest_id: str, bg_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE quests
                SET source_bg_id = ?, status = 'running', updated_at = ?
                WHERE quest_id = ?
                """,
                (bg_id, now, quest_id),
            )
            self._record_event(conn, "quest", quest_id, "quest.run_attached", {"bg_id": bg_id})

    def _update_linked_quests(self, bg_id: str, *, status: str, summary: str,
                              artifact_paths: Optional[list[str]] = None) -> None:
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute("SELECT quest_id FROM quests WHERE source_bg_id = ?", (bg_id,)).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE quests
                    SET status = ?, latest_summary = ?, artifact_paths = ?, updated_at = ?
                    WHERE quest_id = ?
                    """,
                    (status, summary, self._json(artifact_paths or []), now, row["quest_id"]),
                )
                self._record_event(conn, "quest", row["quest_id"], f"quest.{status}", {"source_bg_id": bg_id})

    def _set_background_status(self, bg_id: str, status: str, summary: Optional[str] = None) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_runs
                SET status = ?, latest_summary = COALESCE(?, latest_summary), updated_at = ?
                WHERE bg_id = ?
                """,
                (status, summary, now, bg_id),
            )
            self._record_event(conn, "background", bg_id, f"run.{status}", {"summary": summary})

    def promote_background_run(self, bg_id: str, *, platform: str, chat_id: str) -> dict[str, Any]:
        run = self.get_background_run(bg_id, platform=platform, chat_id=chat_id)
        if not run:
            raise KeyError(bg_id)
        now = time.time()
        with self._connect() as conn:
            alias = self._next_alias(conn, platform, chat_id)
            quest_id = f"sq_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            title = run["prompt"][:80]
            conn.execute(
                """
                INSERT INTO quests(
                    quest_id, alias, title, status, platform, chat_id, user_id,
                    source_bg_id, latest_summary, artifact_paths, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quest_id, alias, title, "active", platform, chat_id, run.get("user_id"),
                    bg_id, run.get("latest_summary"), self._json(run.get("artifact_paths") or []),
                    now, now,
                ),
            )
            self._record_event(conn, "quest", quest_id, "quest.promoted", {"source_bg_id": bg_id})
        resolved = self.resolve_quest(str(alias), platform=platform, chat_id=chat_id)
        if not resolved:
            raise RuntimeError("failed to resolve promoted quest")
        return resolved

    def _next_alias(self, conn: sqlite3.Connection, platform: str, chat_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(alias), 0) + 1 AS next_alias FROM quests WHERE platform = ? AND chat_id = ?",
            (platform, chat_id),
        ).fetchone()
        return int(row["next_alias"])

    def resolve_quest(self, identifier: str, *, platform: str, chat_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            if identifier.isdigit() or identifier.startswith("#"):
                alias = int(identifier.lstrip("#"))
                row = conn.execute(
                    "SELECT * FROM quests WHERE platform = ? AND chat_id = ? AND alias = ?",
                    (platform, chat_id, alias),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM quests WHERE platform = ? AND chat_id = ? AND quest_id = ?",
                    (platform, chat_id, identifier),
                ).fetchone()
        return self._row(row)

    def list_quests(self, *, platform: str, chat_id: str, include_archived: bool = False,
                    limit: int = 10) -> list[dict[str, Any]]:
        where = "platform = ? AND chat_id = ?"
        params: list[Any] = [platform, chat_id]
        if not include_archived:
            where += " AND archived_at IS NULL"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM quests WHERE {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def create_quest(self, *, title: str, platform: str, chat_id: str,
                     user_id: Optional[str] = None) -> dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            alias = self._next_alias(conn, platform, chat_id)
            quest_id = f"sq_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            conn.execute(
                """
                INSERT INTO quests(quest_id, alias, title, status, platform, chat_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (quest_id, alias, title[:120], platform, chat_id, user_id, now, now),
            )
            self._record_event(conn, "quest", quest_id, "quest.created", {"title": title})
        return self.resolve_quest(str(alias), platform=platform, chat_id=chat_id) or {}

    def add_followup(self, *, target_id: str, message: str, platform: str, chat_id: str,
                     user_id: Optional[str] = None) -> dict[str, Any]:
        target_type = "background" if target_id.startswith("bg_") else "quest"
        if target_type == "quest":
            quest = self.resolve_quest(target_id, platform=platform, chat_id=chat_id)
            if quest:
                target_id = quest["quest_id"]
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO quest_followups(target_type, target_id, message, status, platform, chat_id, user_id, created_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (target_type, target_id, message, platform, chat_id, user_id, now),
            )
            self._record_event(conn, target_type, target_id, "followup.added", {"message": message})
            row_id = cur.lastrowid
            row = conn.execute("SELECT * FROM quest_followups WHERE id = ?", (row_id,)).fetchone()
        return dict(row) if row else {}

    def list_followups(self, target_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM quest_followups WHERE target_id = ? ORDER BY created_at ASC",
                (target_id,),
            ).fetchall()
        return [dict(row) for row in rows]
