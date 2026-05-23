"""Prompt Outbox persistence for the Hermes resource-aware dashboard."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_drafts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    priority INTEGER NOT NULL DEFAULT 50,
    project TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    provider TEXT NOT NULL DEFAULT 'openai-codex',
    send_condition_json TEXT NOT NULL DEFAULT '{"mode":"manual","require_confirmation":true}',
    attachments_json TEXT NOT NULL DEFAULT '[]',
    merged_from_json TEXT NOT NULL DEFAULT '[]',
    retention_policy TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    queued_at TEXT,
    sent_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_prompt_drafts_status_priority
    ON prompt_drafts(status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_prompt_drafts_project
    ON prompt_drafts(project);
"""

_VALID_STATUSES = {
    "draft",
    "queued",
    "scheduled",
    "waiting_quota",
    "blocked",
    "sent",
    "answered",
    "archived",
    "failed",
}
_VALID_SEND_MODES = {
    "manual",
    "quota_positive",
    "quota_above_threshold",
    "quota_full",
    "scheduled_time",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _normalize_send_condition(condition: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "mode": "manual",
        "require_confirmation": True,
    }
    if condition:
        normalized.update(condition)
    mode = normalized.get("mode")
    if mode not in _VALID_SEND_MODES:
        raise ValueError(f"Unsupported send condition mode: {mode}")
    threshold = normalized.get("threshold_percent")
    if normalized["mode"] == "quota_above_threshold" and threshold is None:
        raise ValueError("threshold_percent is required for quota_above_threshold")
    if threshold is not None:
        try:
            threshold_float = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold_percent must be a finite number") from exc
        if not math.isfinite(threshold_float) or threshold_float < 0 or threshold_float > 100:
            raise ValueError("threshold_percent must be between 0 and 100")
        normalized["threshold_percent"] = threshold_float
    return normalized


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    return [str(tag).strip() for tag in tags if str(tag).strip()]


@dataclass(frozen=True)
class PromptDraft:
    id: str
    title: str
    content: str
    status: str
    priority: int
    project: str | None
    tags: list[str]
    provider: str
    send_condition: dict[str, Any]
    attachments: list[dict[str, Any]]
    merged_from: list[str]
    retention_policy: str
    created_at: str
    updated_at: str
    queued_at: str | None = None
    sent_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptDraftStore:
    """SQLite-backed PromptDraft store used by the dashboard/API."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (get_hermes_home() / "outbox.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="outbox.db")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _init_schema(self) -> None:
        with self._lock:
            conn = cast(sqlite3.Connection, self._conn)
            conn.executescript(_SCHEMA_SQL)

    def _execute_write(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Cursor:
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(10):
            try:
                with self._lock:
                    conn = cast(sqlite3.Connection, self._conn)
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        cur = conn.execute(sql, params)
                        conn.commit()
                        return cur
                    except BaseException:
                        conn.rollback()
                        raise
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    last_exc = exc
                    time.sleep(min(0.05 * (attempt + 1), 0.5))
                    continue
                raise
        raise last_exc or sqlite3.OperationalError("outbox.db write failed")

    @staticmethod
    def _row_to_prompt(row: sqlite3.Row | None) -> PromptDraft | None:
        if row is None:
            return None
        return PromptDraft(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            status=row["status"],
            priority=int(row["priority"]),
            project=row["project"],
            tags=_json_loads(row["tags_json"], []),
            provider=row["provider"],
            send_condition=_json_loads(row["send_condition_json"], {"mode": "manual"}),
            attachments=_json_loads(row["attachments_json"], []),
            merged_from=_json_loads(row["merged_from_json"], []),
            retention_policy=row["retention_policy"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            queued_at=row["queued_at"],
            sent_at=row["sent_at"],
        )

    def list_prompts(
        self,
        *,
        include_archived: bool = False,
        status: str | None = None,
        project: str | None = None,
        limit: int = 100,
    ) -> list[PromptDraft]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            clauses.append("status != ?")
            params.append("archived")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if project:
            clauses.append("project = ?")
            params.append(project)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            conn = cast(sqlite3.Connection, self._conn)
            rows = conn.execute(
                f"""
                SELECT * FROM prompt_drafts
                {where}
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (*params, safe_limit),
            ).fetchall()
        return [prompt for row in rows if (prompt := self._row_to_prompt(row)) is not None]

    def get_prompt(self, prompt_id: str) -> PromptDraft | None:
        with self._lock:
            conn = cast(sqlite3.Connection, self._conn)
            row = conn.execute(
                "SELECT * FROM prompt_drafts WHERE id = ?",
                (prompt_id,),
            ).fetchone()
        return self._row_to_prompt(row)

    def create_prompt(
        self,
        *,
        title: str,
        content: str,
        status: str = "draft",
        priority: int = 50,
        project: str | None = None,
        tags: list[str] | None = None,
        provider: str = "openai-codex",
        send_condition: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        merged_from: list[str] | None = None,
        retention_policy: str = "default",
    ) -> PromptDraft:
        title = title.strip()
        content = content.strip()
        if not title:
            raise ValueError("title is required")
        if not content:
            raise ValueError("content is required")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported prompt status: {status}")
        now = _now_iso()
        prompt_id = str(uuid.uuid4())
        normalized_condition = _normalize_send_condition(send_condition)
        queued_at = now if status == "queued" else None
        self._execute_write(
            """
            INSERT INTO prompt_drafts (
                id, title, content, status, priority, project, tags_json,
                provider, send_condition_json, attachments_json, merged_from_json,
                retention_policy, created_at, updated_at, queued_at, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt_id,
                title,
                content,
                status,
                int(priority),
                project,
                _json_dumps(_normalize_tags(tags)),
                provider or "openai-codex",
                _json_dumps(normalized_condition),
                _json_dumps(attachments or []),
                _json_dumps(merged_from or []),
                retention_policy or "default",
                now,
                now,
                queued_at,
                None,
            ),
        )
        prompt = self.get_prompt(prompt_id)
        assert prompt is not None
        return prompt

    def update_prompt(self, prompt_id: str, **changes: Any) -> PromptDraft | None:
        existing = self.get_prompt(prompt_id)
        if existing is None:
            return None

        data = existing.to_dict()
        for key, value in changes.items():
            data[key] = value
        if data.get("title") is None:
            raise ValueError("title is required")
        if data.get("content") is None:
            raise ValueError("content is required")
        if data.get("status") is None:
            raise ValueError("status is required")
        data["title"] = str(data["title"]).strip()
        data["content"] = str(data["content"]).strip()
        if not data["title"]:
            raise ValueError("title is required")
        if not data["content"]:
            raise ValueError("content is required")
        if data["status"] not in _VALID_STATUSES:
            raise ValueError(f"Unsupported prompt status: {data['status']}")
        data["tags"] = _normalize_tags(data.get("tags"))
        data["send_condition"] = _normalize_send_condition(data.get("send_condition"))
        data["updated_at"] = _now_iso()
        if data["status"] == "queued" and not data.get("queued_at"):
            data["queued_at"] = data["updated_at"]

        self._execute_write(
            """
            UPDATE prompt_drafts
            SET title = ?, content = ?, status = ?, priority = ?, project = ?,
                tags_json = ?, provider = ?, send_condition_json = ?,
                attachments_json = ?, merged_from_json = ?, retention_policy = ?,
                updated_at = ?, queued_at = ?, sent_at = ?
            WHERE id = ?
            """,
            (
                data["title"],
                data["content"],
                data["status"],
                int(data["priority"]),
                data.get("project"),
                _json_dumps(data["tags"]),
                data.get("provider") or "openai-codex",
                _json_dumps(data["send_condition"]),
                _json_dumps(data.get("attachments") or []),
                _json_dumps(data.get("merged_from") or []),
                data.get("retention_policy") or "default",
                data["updated_at"],
                data.get("queued_at"),
                data.get("sent_at"),
                prompt_id,
            ),
        )
        return self.get_prompt(prompt_id)

    def delete_prompt(self, prompt_id: str) -> bool:
        cur = self._execute_write("DELETE FROM prompt_drafts WHERE id = ?", (prompt_id,))
        return cur.rowcount > 0
