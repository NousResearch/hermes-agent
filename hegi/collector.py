"""Read-only collection and deterministic cross-profile merge."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .config import AgentSourceConfig
from .models import SourceMessage
from .state import StateStore


_EXCLUDED_BLOCK_TYPES = {
    "tool_call",
    "tool_result",
    "function_call",
    "function_result",
    "reasoning",
    "system",
}
_TEXT_KEYS = ("text", "content", "message", "body", "output_text")


def extract_content(value: Any) -> str:
    """Extract human-visible text while excluding tool/reasoning blocks."""
    if value is None:
        return ""
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                parsed = stripped
        else:
            parsed = stripped
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        block_type = str(item.get("type", "")).lower()
        role = str(item.get("role", "")).lower()
        if block_type in _EXCLUDED_BLOCK_TYPES or role in {"system", "tool"}:
            return
        found = False
        for key in _TEXT_KEYS:
            if key in item:
                walk(item[key])
                found = True
        if not found:
            for child in item.values():
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(parsed)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def open_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


class HermesSQLiteCollector:
    def __init__(
        self,
        sources: list[AgentSourceConfig],
        state: StateStore,
        *,
        timestamp_bucket_seconds: int = 3,
    ):
        self.sources = sources
        self.state = state
        self.timestamp_bucket_seconds = max(1, timestamp_bucket_seconds)

    def collect(self, chat_id: str, *, since: float | None = None) -> list[SourceMessage]:
        collected: list[SourceMessage] = []
        high_water: dict[str, tuple[float, int]] = {}
        for source in self.sources:
            cursor_ts, cursor_id = self.state.get_cursor(str(source.db_path))
            effective_since = since if since is not None else cursor_ts
            rows = self._query_source(source, chat_id, effective_since, cursor_id)
            collected.extend(rows)
            if rows:
                newest = max(rows, key=lambda item: (item.timestamp, item.message_id))
                high_water[str(source.db_path)] = (newest.timestamp, newest.message_id)
        self.state.buffer_messages([asdict(message) for message in collected])
        for source_db, (timestamp, message_id) in high_water.items():
            self.state.set_cursor(source_db, timestamp, message_id)
        buffered = [
            SourceMessage(**payload) for payload in self.state.buffered_messages(chat_id)
        ]
        return deduplicate_messages(buffered, self.timestamp_bucket_seconds)

    @staticmethod
    def _query_source(
        source: AgentSourceConfig,
        chat_id: str,
        since: float,
        cursor_id: int,
    ) -> list[SourceMessage]:
        connection = open_readonly(source.db_path)
        try:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(messages)").fetchall()
            }
            required = {"id", "session_id", "role", "content", "timestamp"}
            missing = required - columns
            if missing:
                raise RuntimeError(
                    f"{source.db_path} messages schema missing: {sorted(missing)}"
                )
            optional = lambda name, fallback: f"m.{name}" if name in columns else fallback
            query = f"""
                SELECT m.id AS message_id, m.session_id, m.role, m.content, m.timestamp,
                    {optional("platform_message_id", "NULL")} AS platform_message_id,
                    {optional("active", "1")} AS active,
                    {optional("compacted", "0")} AS compacted,
                    s.chat_id, s.chat_type
                FROM messages AS m
                JOIN sessions AS s ON s.id = m.session_id
                WHERE s.chat_id = ?
                  AND m.role IN ('user', 'assistant')
                  AND COALESCE({optional("active", "1")}, 1) = 1
                  AND (m.timestamp > ? OR (m.timestamp = ? AND m.id > ?))
                ORDER BY m.timestamp, m.id
            """
            rows = connection.execute(query, (str(chat_id), since, since, cursor_id)).fetchall()
        finally:
            connection.close()
        result: list[SourceMessage] = []
        for row in rows:
            content = extract_content(row["content"])
            if not content:
                continue
            result.append(
                SourceMessage(
                    source_agent="교수" if str(row["role"]).lower() == "user" else source.name,
                    source_db=str(source.db_path),
                    message_id=int(row["message_id"]),
                    session_id=str(row["session_id"]),
                    platform_message_id=(
                        str(row["platform_message_id"])
                        if row["platform_message_id"] is not None
                        else None
                    ),
                    chat_id=str(row["chat_id"]),
                    chat_type=str(row["chat_type"] or ""),
                    role=str(row["role"]).lower(),  # type: ignore[arg-type]
                    content=content,
                    timestamp=float(row["timestamp"]),
                    active=bool(row["active"]),
                    compacted=bool(row["compacted"]),
                )
            )
        return result


def _dedup_key(message: SourceMessage, bucket_seconds: int) -> str:
    if message.platform_message_id:
        identity = f"platform:{message.role}:{message.platform_message_id}"
        if message.role == "assistant":
            identity += f":{message.source_agent}"
    else:
        bucket = int(message.timestamp // bucket_seconds)
        identity = f"{message.role}:{normalize_content(message.content)}:{bucket}"
        if message.role == "assistant":
            identity += f":{message.source_agent}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def deduplicate_messages(
    messages: Iterable[SourceMessage], bucket_seconds: int = 3
) -> list[SourceMessage]:
    unique: dict[str, SourceMessage] = {}
    for original in messages:
        message = (
            replace(original, source_agent="교수")
            if original.role == "user" and original.source_agent != "교수"
            else original
        )
        key = _dedup_key(message, bucket_seconds)
        existing = unique.get(key)
        if existing is None or (message.timestamp, message.message_id) < (
            existing.timestamp,
            existing.message_id,
        ):
            unique[key] = message
    return sorted(unique.values(), key=lambda item: (item.timestamp, item.message_id))
