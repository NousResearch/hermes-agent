"""PostgreSQL-native durable message, lineage, import, and cleanup operations."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from state_store.postgres.session_db_base import PostgresSessionDBBase
from state_store.session_api import normalize_row


class PostgresSessionDBMessageOperations(PostgresSessionDBBase):
    """Message operations intended for composition into ``PostgresSessionDB``.

    The SQL here is deliberately PostgreSQL-native.  Every cursor remains
    transaction-local, and bounded reads use ``fetchmany`` rather than
    materializing an unbounded result set.
    """

    _CONTENT_JSON_PREFIX = "\x00json:"
    _IMPORT_MAX_SESSIONS = 500
    _IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
    _IMPORT_MAX_TOTAL_MESSAGES = 50_000
    _IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
    _IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        if content is None or isinstance(content, (str, bytes, int, float)):
            return content
        try:
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            return str(content)

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX) :])
            except (json.JSONDecodeError, TypeError):
                return content
        return content

    @staticmethod
    def _timestamp(value: Any, now: float) -> float:
        if value is None:
            return now
        try:
            return float(value.timestamp()) if hasattr(value, "timestamp") else float(value)
        except (TypeError, ValueError):
            return now

    @staticmethod
    def _tool_call_count(tool_calls: Any) -> int:
        if tool_calls is None:
            return 0
        return len(tool_calls) if isinstance(tool_calls, list) else 1

    @staticmethod
    def _json_or_none(value: Any) -> Optional[str]:
        return json.dumps(value) if value else None

    def _rows(
        self,
        connection: Any,
        query: str,
        params: tuple[Any, ...] = (),
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or limit > self._MAX_READ_ROWS:
            raise ValueError("read limit exceeds PostgreSQL SessionDB maximum")
        cursor = connection.execute(query, params)
        columns = self._cursor_columns(cursor)
        result: list[dict[str, Any]] = []
        remaining = limit
        while remaining:
            batch = cursor.fetchmany(min(self._READ_BATCH_SIZE, remaining))
            if not batch:
                break
            for row in batch:
                normalized = normalize_row(row, columns=columns)
                if normalized is not None:
                    result.append(normalized)
            remaining -= len(batch)
        return result

    def _row(
        self,
        connection: Any,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> Optional[dict[str, Any]]:
        cursor = connection.execute(query, params)
        return normalize_row(cursor.fetchone(), columns=self._cursor_columns(cursor))

    def _hydrate_message(self, row: dict[str, Any]) -> dict[str, Any]:
        message = dict(row)
        message["content"] = self._decode_content(message.get("content"))
        for field, fallback in (
            ("tool_calls", []),
            ("reasoning_details", None),
            ("codex_reasoning_items", None),
            ("codex_message_items", None),
        ):
            if message.get(field):
                try:
                    message[field] = self._json_loads(message[field])
                except (json.JSONDecodeError, TypeError):
                    message[field] = fallback
        return message

    def _insert_message_rows(
        self,
        connection: Any,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, int]:
        now = time.time()
        inserted = 0
        tools = 0
        for message in messages:
            role = message.get("role", "unknown")
            tool_calls = message.get("tool_calls")
            timestamp = self._timestamp(message.get("timestamp"), now)
            reasoning_details = (
                message.get("reasoning_details") if role == "assistant" else None
            )
            codex_reasoning_items = (
                message.get("codex_reasoning_items") if role == "assistant" else None
            )
            codex_message_items = (
                message.get("codex_message_items") if role == "assistant" else None
            )
            connection.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, tool_call_id, tool_calls, tool_name,
                    effect_disposition, timestamp, token_count, finish_reason,
                    reasoning, reasoning_content, reasoning_details,
                    codex_reasoning_items, codex_message_items, platform_message_id,
                    observed, active, compacted
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, 1, 0
                )
                """,
                (
                    session_id,
                    role,
                    self._encode_content(message.get("content")),
                    message.get("tool_call_id"),
                    self._json_or_none(tool_calls),
                    message.get("tool_name"),
                    message.get("effect_disposition"),
                    timestamp,
                    message.get("token_count"),
                    message.get("finish_reason"),
                    message.get("reasoning") if role == "assistant" else None,
                    message.get("reasoning_content") if role == "assistant" else None,
                    self._json_or_none(reasoning_details),
                    self._json_or_none(codex_reasoning_items),
                    self._json_or_none(codex_message_items),
                    message.get("platform_message_id") or message.get("message_id"),
                    1 if message.get("observed") else 0,
                ),
            )
            inserted += 1
            tools += self._tool_call_count(tool_calls)
            now = max(now + 0.000001, timestamp + 0.000001)
        return inserted, tools

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
        effect_disposition: Optional[str] = None,
        timestamp: Any = None,
    ) -> int:
        message_timestamp = self._timestamp(timestamp, time.time())
        tool_count = self._tool_call_count(tool_calls)

        def operation(connection: Any) -> int:
            row = self._row(
                connection,
                """
                INSERT INTO messages (
                    session_id, role, content, tool_call_id, tool_calls, tool_name,
                    effect_disposition, timestamp, token_count, finish_reason,
                    reasoning, reasoning_content, reasoning_details,
                    codex_reasoning_items, codex_message_items, platform_message_id,
                    observed, active, compacted
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, 1, 0
                ) RETURNING id
                """,
                (
                    session_id,
                    role,
                    self._encode_content(content),
                    tool_call_id,
                    self._json_or_none(tool_calls),
                    tool_name,
                    effect_disposition,
                    message_timestamp,
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_content,
                    self._json_or_none(reasoning_details),
                    self._json_or_none(codex_reasoning_items),
                    self._json_or_none(codex_message_items),
                    platform_message_id,
                    1 if observed else 0,
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET message_count = COALESCE(message_count, 0) + 1,
                    tool_call_count = COALESCE(tool_call_count, 0) + %s
                WHERE id = %s
                """,
                (tool_count, session_id),
            )
            if row is None or row.get("id") is None:
                raise RuntimeError("PostgreSQL message insert did not return an identity")
            return int(row["id"])

        return self._run(operation, read_only=False)

    def replace_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        active_only: bool = False,
    ) -> None:
        def operation(connection: Any) -> None:
            active_clause = " AND active = 1" if active_only else ""
            connection.execute(
                f"DELETE FROM messages WHERE session_id = %s{active_clause}",
                (session_id,),
            )
            inserted, tools = self._insert_message_rows(connection, session_id, messages)
            connection.execute(
                """
                UPDATE sessions
                SET message_count = %s, tool_call_count = %s
                WHERE id = %s
                """,
                (inserted, tools, session_id),
            )

        self._run(operation, read_only=False)

    def has_archived_messages(self, session_id: str) -> bool:
        return self._read_one(
            "SELECT 1 AS found FROM messages "
            "WHERE session_id = %s AND active = 0 LIMIT 1",
            (session_id,),
        ) is not None

    def archive_and_compact(
        self, session_id: str, compacted_messages: List[Dict[str, Any]]
    ) -> int:
        def operation(connection: Any) -> int:
            connection.execute(
                """
                UPDATE messages SET active = 0, compacted = 1
                WHERE session_id = %s AND active = 1
                """,
                (session_id,),
            )
            inserted, tools = self._insert_message_rows(
                connection, session_id, compacted_messages
            )
            connection.execute(
                """
                UPDATE sessions SET message_count = %s, tool_call_count = %s
                WHERE id = %s
                """,
                (inserted, tools, session_id),
            )
            return inserted

        return self._run(operation, read_only=False)

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        requested = self._MAX_READ_ROWS if limit is None else limit
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise ValueError("limit must be a positive integer")
        if requested > self._MAX_READ_ROWS:
            raise ValueError("limit exceeds PostgreSQL SessionDB maximum")
        active = "" if include_inactive else " AND active = 1"
        rows = self._read_many(
            f"""
            SELECT * FROM messages
            WHERE session_id = %s{active}
            ORDER BY id ASC LIMIT %s OFFSET %s
            """,
            (session_id, requested, offset),
            limit=requested,
        )
        return [self._hydrate_message(row) for row in rows]

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        window = max(0, min(int(window), self._MAX_READ_ROWS // 2))

        def operation(connection: Any) -> Dict[str, Any]:
            if self._row(
                connection,
                "SELECT 1 AS found FROM messages WHERE id = %s AND session_id = %s",
                (around_message_id, session_id),
            ) is None:
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before = self._rows(
                connection,
                """
                SELECT * FROM messages WHERE session_id = %s AND id <= %s
                ORDER BY id DESC LIMIT %s
                """,
                (session_id, around_message_id, window + 1),
                limit=window + 1,
            )
            after = self._rows(
                connection,
                """
                SELECT * FROM messages WHERE session_id = %s AND id > %s
                ORDER BY id ASC LIMIT %s
                """,
                (session_id, around_message_id, window),
                limit=max(window, 1),
            ) if window else []
            rows = list(reversed(before)) + after
            return {
                "window": [self._hydrate_message(row) for row in rows],
                "messages_before": max(0, len(before) - 1),
                "messages_after": len(after),
            }

        return self._run(operation, read_only=True)

    def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
        primitive = self.get_messages_around(session_id, around_message_id, window)
        rows = primitive["window"]
        if not rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }
        bookend = max(0, min(int(bookend), self._MAX_READ_ROWS))
        if keep_roles is not None:
            keep = set(keep_roles)
            visible = [
                row
                for row in rows
                if row.get("id") == around_message_id or row.get("role") in keep
            ]
        else:
            visible = rows
        if not bookend:
            return {
                **primitive,
                "window": visible,
                "bookend_start": [],
                "bookend_end": [],
            }
        first_id, last_id = rows[0]["id"], rows[-1]["id"]
        role_sql = ""
        role_params: tuple[Any, ...] = ()
        if keep_roles is not None:
            role_sql = " AND role = ANY(%s)"
            role_params = (list(keep_roles),)

        def operation(connection: Any) -> Dict[str, Any]:
            start = self._rows(
                connection,
                f"""
                SELECT * FROM messages WHERE session_id = %s AND id < %s
                {role_sql} AND COALESCE(content, '') <> ''
                ORDER BY id ASC LIMIT %s
                """,
                (session_id, first_id, *role_params, bookend),
                limit=bookend,
            )
            end = self._rows(
                connection,
                f"""
                SELECT * FROM messages WHERE session_id = %s AND id > %s
                {role_sql} AND COALESCE(content, '') <> ''
                ORDER BY id DESC LIMIT %s
                """,
                (session_id, last_id, *role_params, bookend),
                limit=bookend,
            )
            return {
                **primitive,
                "window": visible,
                "bookend_start": [self._hydrate_message(row) for row in start],
                "bookend_end": [
                    self._hydrate_message(row) for row in reversed(end)
                ],
            }

        return self._run(operation, read_only=True)

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]

        def operation(connection: Any) -> List[str]:
            chain: list[str] = []
            current = session_id
            seen: set[str] = set()
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = self._row(
                    connection,
                    "SELECT parent_session_id FROM sessions WHERE id = %s",
                    (current,),
                )
                if row is None:
                    break
                current = row.get("parent_session_id")
            return list(reversed(chain)) or [session_id]

        return self._run(operation, read_only=True)

    def get_conversation_root(self, session_id: str) -> str:
        chain = self._session_lineage_root_to_tip(session_id)
        return chain[0] if chain and chain[0] else session_id

    def _compression_tip_in(self, connection: Any, session_id: str) -> str:
        current = session_id
        seen = {current}
        for _ in range(32):
            row = self._row(
                connection,
                """
                SELECT s.id FROM sessions s
                WHERE s.parent_session_id = %s
                  AND COALESCE(s.model_config, '') NOT LIKE '%%"_branched_from"%%'
                  AND COALESCE(s.model_config, '') NOT LIKE '%%"_delegate_from"%%'
                  AND COALESCE(s.source, '') <> 'tool'
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = s.parent_session_id
                        AND p.end_reason = 'compression'
                  )
                ORDER BY s.started_at DESC, s.id DESC LIMIT 1
                """,
                (current,),
            )
            child = row.get("id") if row else None
            if not child or child in seen:
                break
            seen.add(child)
            current = str(child)
        return current

    def resolve_resume_session_id(self, session_id: str) -> str:
        if not session_id:
            return session_id

        def operation(connection: Any) -> str:
            current = self._compression_tip_in(connection, session_id)
            best: Optional[str] = None
            seen = {current}
            for _ in range(32):
                if self._row(
                    connection,
                    "SELECT 1 AS found FROM messages WHERE session_id = %s LIMIT 1",
                    (current,),
                ) is not None:
                    best = current
                child_row = self._row(
                    connection,
                    """
                    SELECT id FROM sessions
                    WHERE parent_session_id = %s
                      AND COALESCE(model_config, '') NOT LIKE '%%"_branched_from"%%'
                      AND COALESCE(model_config, '') NOT LIKE '%%"_delegate_from"%%'
                      AND COALESCE(source, '') <> 'tool'
                    ORDER BY started_at DESC, id DESC LIMIT 1
                    """,
                    (current,),
                )
                child = child_row.get("id") if child_row else None
                if not child or child in seen:
                    break
                seen.add(child)
                current = str(child)
            return best or current

        return self._run(operation, read_only=True)

    @staticmethod
    def _is_duplicate_replayed_user_message(
        messages: List[Dict[str, Any]], message: Dict[str, Any]
    ) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return False
        for previous in reversed(messages):
            if previous.get("role") == "user" and previous.get("content") == content:
                return True
            if previous.get("role") == "assistant" and (
                previous.get("content") or previous.get("tool_calls")
            ):
                return False
        return False

    def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        repair_alternation: bool = False,
    ) -> List[Dict[str, Any]]:
        session_ids = (
            self._session_lineage_root_to_tip(session_id)
            if include_ancestors
            else [session_id]
        )
        active = "" if include_inactive else " AND active = 1"
        rows = self._read_many(
            f"""
            SELECT role, content, tool_call_id, tool_calls, tool_name,
                   effect_disposition, finish_reason, reasoning, reasoning_content,
                   reasoning_details, codex_reasoning_items, codex_message_items,
                   platform_message_id, observed, timestamp
            FROM messages WHERE session_id = ANY(%s){active} ORDER BY id ASC
            """,
            (session_ids,),
            limit=self._MAX_READ_ROWS,
        )
        messages: list[dict[str, Any]] = []
        for row in rows:
            hydrated = self._hydrate_message(row)
            message: dict[str, Any] = {
                "role": hydrated["role"],
                "content": hydrated.get("content"),
            }
            for field in (
                "timestamp",
                "tool_call_id",
                "tool_name",
                "effect_disposition",
                "finish_reason",
                "reasoning",
                "reasoning_content",
                "reasoning_details",
                "codex_reasoning_items",
                "codex_message_items",
            ):
                if hydrated.get(field) is not None:
                    message[field] = hydrated[field]
            if hydrated.get("tool_calls") is not None:
                message["tool_calls"] = hydrated["tool_calls"]
            if hydrated.get("platform_message_id"):
                message["message_id"] = hydrated["platform_message_id"]
            if hydrated.get("observed"):
                message["observed"] = True
            if include_ancestors and self._is_duplicate_replayed_user_message(
                messages, message
            ):
                continue
            messages.append(message)
        if repair_alternation and messages:
            from agent.agent_runtime_helpers import repair_message_sequence

            repair_message_sequence(None, messages)
        return messages

    def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        def operation(connection: Any) -> Dict[str, Any]:
            target = self._row(
                connection,
                "SELECT * FROM messages WHERE id = %s AND session_id = %s FOR UPDATE",
                (target_message_id, session_id),
            )
            if target is None:
                raise ValueError(
                    f"message {target_message_id} not found in session {session_id}"
                )
            if target.get("role") != "user":
                raise ValueError(
                    "rewind target must be a 'user' message "
                    f"(got role={target.get('role')!r}, id={target_message_id})"
                )
            rewound = self._rows(
                connection,
                """
                UPDATE messages SET active = 0
                WHERE session_id = %s AND id >= %s AND active = 1
                RETURNING id
                """,
                (session_id, target_message_id),
                limit=self._MAX_READ_ROWS,
            )
            connection.execute(
                """
                UPDATE sessions
                SET rewind_count = COALESCE(rewind_count, 0) + 1
                WHERE id = %s
                """,
                (session_id,),
            )
            head = self._row(
                connection,
                """
                SELECT id FROM messages WHERE session_id = %s AND active = 1
                ORDER BY id DESC LIMIT 1
                """,
                (session_id,),
            )
            target["content"] = self._decode_content(target.get("content"))
            return {
                "rewound_count": len(rewound),
                "target_message": target,
                "new_head_id": head.get("id") if head else None,
            }

        return self._run(operation, read_only=False)

    def restore_rewound(self, session_id: str, since_message_id: int) -> int:
        def operation(connection: Any) -> int:
            cursor = connection.execute(
                """
                UPDATE messages SET active = 1
                WHERE session_id = %s AND id >= %s AND active = 0
                """,
                (session_id, since_message_id),
            )
            return int(cursor.rowcount or 0)

        return self._run(operation, read_only=False)

    def list_recent_user_messages(
        self,
        session_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, self._MAX_READ_ROWS)
        active = "" if include_inactive else " AND active = 1"
        rows = self._read_many(
            f"""
            SELECT id, timestamp, content FROM messages
            WHERE session_id = %s AND role = 'user'{active}
            ORDER BY id DESC LIMIT %s
            """,
            (session_id, limit),
            limit=limit,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            content = self._decode_content(row.get("content"))
            if isinstance(content, list):
                preview = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip() or "[multimodal content]"
            else:
                preview = content if isinstance(content, str) else ""
            preview = " ".join(preview.split())
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "preview": preview[:77] + "..." if len(preview) > 80 else preview,
                }
            )
        return result

    def message_count(self, session_id: str = None) -> int:
        query = "SELECT COUNT(*) AS count FROM messages"
        params: tuple[Any, ...] = ()
        if session_id:
            query += " WHERE session_id = %s"
            params = (session_id,)
        row = self._read_one(query, params)
        return int(row["count"]) if row else 0

    def has_platform_message_id(
        self, session_id: str, platform_message_id: str
    ) -> bool:
        return self._read_one(
            """
            SELECT 1 AS found FROM messages
            WHERE session_id = %s AND platform_message_id = %s LIMIT 1
            """,
            (session_id, platform_message_id),
        ) is not None

    @staticmethod
    def _is_branch_child_row(session: Dict[str, Any]) -> bool:
        raw = session.get("model_config")
        if not raw:
            return False
        try:
            config = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(config, dict) and config.get("_branched_from") is not None

    def _get_session_in(
        self, connection: Any, session_id: str
    ) -> Optional[dict[str, Any]]:
        return self._row(
            connection, "SELECT * FROM sessions WHERE id = %s", (session_id,)
        )

    def _compression_lineage_in(self, connection: Any, session_id: str) -> list[str]:
        session = self._get_session_in(connection, session_id)
        if session is None:
            return []
        if self._is_branch_child_row(session):
            return [session_id]
        root = session
        seen = {root["id"]}
        while root.get("parent_session_id") and not self._is_branch_child_row(root):
            parent = self._get_session_in(connection, root["parent_session_id"])
            if not parent or parent.get("end_reason") != "compression" or parent["id"] in seen:
                break
            root = parent
            seen.add(root["id"])
        lineage = [root["id"]]
        current = root
        while current.get("end_reason") == "compression":
            children = self._rows(
                connection,
                """
                SELECT * FROM sessions WHERE parent_session_id = %s
                ORDER BY started_at ASC, id ASC LIMIT %s
                """,
                (current["id"], self._MAX_READ_ROWS),
                limit=self._MAX_READ_ROWS,
            )
            child = next(
                (candidate for candidate in children if not self._is_branch_child_row(candidate)),
                None,
            )
            if not child or child["id"] in seen:
                break
            lineage.append(child["id"])
            seen.add(child["id"])
            current = child
        return lineage if session_id in lineage else [session_id]

    def get_compression_lineage(self, session_id: str) -> List[str]:
        return self._run(
            lambda connection: self._compression_lineage_in(connection, session_id),
            read_only=True,
        )

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        def operation(connection: Any) -> Optional[dict[str, Any]]:
            session = self._get_session_in(connection, session_id)
            if session is None:
                return None
            rows = self._rows(
                connection,
                """
                SELECT * FROM messages
                WHERE session_id = %s AND active = 1 ORDER BY id ASC
                """,
                (session_id,),
                limit=self._MAX_READ_ROWS,
            )
            return {**session, "messages": [self._hydrate_message(row) for row in rows]}

        return self._run(operation, read_only=True)

    def export_session_lineage(self, session_id: str) -> Optional[Dict[str, Any]]:
        def operation(connection: Any) -> Optional[dict[str, Any]]:
            lineage = self._compression_lineage_in(connection, session_id)
            if not lineage:
                return None
            segments: list[dict[str, Any]] = []
            for segment_id in lineage:
                session = self._get_session_in(connection, segment_id)
                if session is None:
                    continue
                rows = self._rows(
                    connection,
                    """
                    SELECT * FROM messages WHERE session_id = %s AND active = 1
                    ORDER BY id ASC
                    """,
                    (segment_id,),
                    limit=self._MAX_READ_ROWS,
                )
                segments.append(
                    {**session, "messages": [self._hydrate_message(row) for row in rows]}
                )
            if not segments:
                return None
            base = dict(segments[-1])
            base["segments"] = segments
            base["lineage_session_ids"] = [segment["id"] for segment in segments]
            base["message_count"] = sum(len(segment["messages"]) for segment in segments)
            base["messages"] = [
                message for segment in segments for message in segment["messages"]
            ]
            return base

        return self._run(operation, read_only=True)

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        def operation(connection: Any) -> list[dict[str, Any]]:
            where, params = "", ()
            if source is not None:
                where, params = " WHERE source = %s", (source,)
            sessions = self._rows(
                connection,
                f"SELECT * FROM sessions{where} ORDER BY started_at ASC, id ASC LIMIT %s",
                (*params, self._MAX_READ_ROWS),
                limit=self._MAX_READ_ROWS,
            )
            exported: list[dict[str, Any]] = []
            for session in sessions:
                messages = self._rows(
                    connection,
                    """
                    SELECT * FROM messages
                    WHERE session_id = %s AND active = 1 ORDER BY id ASC
                    """,
                    (session["id"],),
                    limit=self._MAX_READ_ROWS,
                )
                exported.append(
                    {**session, "messages": [self._hydrate_message(row) for row in messages]}
                )
            return exported

        return self._run(operation, read_only=True)

    @staticmethod
    def _import_text_or_none(value: Any, field: str) -> Optional[str]:
        if value is None or isinstance(value, str):
            return value
        raise ValueError(f"{field} must be a string")

    @staticmethod
    def _import_json_object_or_none(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"{field} must be valid JSON") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"{field} must be a JSON object")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object")
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be JSON serializable") from error

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _import_int_or_none(value: Any, field: str) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be an integer") from error

    @staticmethod
    def _int_or_default(value: Any, default: int = 0) -> int:
        try:
            return default if value is None else int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _import_error(index: int, session_id: str, error: str) -> Dict[str, Any]:
        result: dict[str, Any] = {"index": index, "error": error}
        if session_id:
            result["session_id"] = session_id
        return result

    def _normalize_import(
        self, sessions: List[Dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:
            raise ValueError(
                f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries"
            )
        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen: set[str] = set()
        total_messages = 0
        total_bytes = 0
        text_fields = (
            "source", "user_id", "model", "system_prompt", "end_reason", "cwd",
            "git_branch", "git_repo_root", "billing_provider", "billing_base_url",
            "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
        )
        message_text_fields = (
            "tool_call_id", "tool_name", "effect_disposition", "finish_reason",
            "reasoning", "reasoning_content", "platform_message_id", "message_id",
        )
        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                errors.append(self._import_error(index, "", "session must be an object"))
                continue
            session_id = str(raw.get("id") or "").strip()
            if not session_id:
                errors.append(self._import_error(index, "", "session id is required"))
                continue
            if session_id in seen:
                errors.append(self._import_error(index, session_id, "duplicate session id"))
                continue
            messages = raw.get("messages") or []
            if not isinstance(messages, list):
                errors.append(self._import_error(index, session_id, "messages must be a list"))
                continue
            if len(messages) > self._IMPORT_MAX_MESSAGES_PER_SESSION:
                errors.append(
                    self._import_error(
                        index, session_id, "messages exceeds the per-session import limit"
                    )
                )
                continue
            if any(not isinstance(message, dict) for message in messages):
                errors.append(
                    self._import_error(
                        index, session_id, "messages must contain only objects"
                    )
                )
                continue
            try:
                size = len(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
            except (TypeError, ValueError):
                errors.append(
                    self._import_error(index, session_id, "session must be JSON serializable")
                )
                continue
            if size > self._IMPORT_MAX_SESSION_BYTES:
                errors.append(
                    self._import_error(index, session_id, "session exceeds the import size limit")
                )
                continue
            total_bytes += size
            if total_bytes > self._IMPORT_MAX_TOTAL_BYTES:
                errors.append(
                    self._import_error(index, session_id, "import exceeds the total size limit")
                )
                continue
            try:
                session = dict(raw)
                session["id"] = session_id
                session["model_config"] = self._import_json_object_or_none(
                    session.get("model_config"), "model_config"
                )
                session["parent_session_id"] = self._import_text_or_none(
                    session.get("parent_session_id"), "parent_session_id"
                )
                for field in text_fields:
                    session[field] = self._import_text_or_none(session.get(field), field)
                clean_messages: list[dict[str, Any]] = []
                for message_index, message in enumerate(messages):
                    clean = dict(message)
                    if not isinstance(clean.get("role"), str) or not clean["role"]:
                        raise ValueError(
                            f"messages[{message_index}].role must be a non-empty string"
                        )
                    for field in message_text_fields:
                        clean[field] = self._import_text_or_none(clean.get(field), field)
                    clean["token_count"] = self._import_int_or_none(
                        clean.get("token_count"), "token_count"
                    )
                    clean_messages.append(clean)
            except ValueError as error:
                errors.append(self._import_error(index, session_id, str(error)))
                continue
            total_messages += len(clean_messages)
            if total_messages > self._IMPORT_MAX_TOTAL_MESSAGES:
                errors.append(
                    self._import_error(
                        index, session_id, "messages exceeds the total import limit"
                    )
                )
                continue
            seen.add(session_id)
            normalized.append({"session": session, "messages": clean_messages})
        return normalized, errors

    def import_sessions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized, errors = self._normalize_import(sessions)
        if errors:
            return {
                "ok": False, "imported": 0, "skipped": 0, "detached": 0,
                "errors": errors,
            }

        def operation(connection: Any) -> dict[str, Any]:
            imported: list[str] = []
            skipped: list[str] = []
            requested_parents: dict[str, str] = {}
            for item in normalized:
                raw, messages = item["session"], item["messages"]
                session_id = raw["id"]
                if self._row(
                    connection, "SELECT 1 AS found FROM sessions WHERE id = %s", (session_id,)
                ):
                    skipped.append(session_id)
                    continue
                connection.execute(
                    """
                    INSERT INTO sessions (
                        id, source, user_id, model, model_config, system_prompt,
                        parent_session_id, started_at, ended_at, end_reason,
                        message_count, tool_call_count, input_tokens, output_tokens,
                        cache_read_tokens, cache_write_tokens, reasoning_tokens, cwd,
                        git_branch, git_repo_root, billing_provider, billing_base_url,
                        billing_mode, estimated_cost_usd, actual_cost_usd, cost_status,
                        cost_source, pricing_version, title, api_call_count, archived
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, 0, 0,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        session_id, str(raw.get("source") or "import"), raw.get("user_id"),
                        raw.get("model"), raw.get("model_config"), raw.get("system_prompt"),
                        self._float_or_none(raw.get("started_at")) or time.time(),
                        self._float_or_none(raw.get("ended_at")), raw.get("end_reason"),
                        self._int_or_default(raw.get("input_tokens")),
                        self._int_or_default(raw.get("output_tokens")),
                        self._int_or_default(raw.get("cache_read_tokens")),
                        self._int_or_default(raw.get("cache_write_tokens")),
                        self._int_or_default(raw.get("reasoning_tokens")), raw.get("cwd"),
                        raw.get("git_branch"), raw.get("git_repo_root"),
                        raw.get("billing_provider"), raw.get("billing_base_url"),
                        raw.get("billing_mode"), self._float_or_none(raw.get("estimated_cost_usd")),
                        self._float_or_none(raw.get("actual_cost_usd")), raw.get("cost_status"),
                        raw.get("cost_source"), raw.get("pricing_version"), raw.get("title"),
                        self._int_or_default(raw.get("api_call_count")),
                        1 if raw.get("archived") else 0,
                    ),
                )
                inserted, tools = self._insert_message_rows(connection, session_id, messages)
                connection.execute(
                    "UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
                    (inserted, tools, session_id),
                )
                parent = str(raw.get("parent_session_id") or "").strip()
                if parent:
                    requested_parents[session_id] = parent
                imported.append(session_id)
            detached = 0
            for child, parent in requested_parents.items():
                parent_exists = self._row(
                    connection, "SELECT 1 AS found FROM sessions WHERE id = %s", (parent,)
                )
                current, seen = parent, {child}
                while current and current not in seen:
                    seen.add(current)
                    current = requested_parents.get(current, "")
                if not parent_exists or current in seen and current == child:
                    detached += 1
                    continue
                connection.execute(
                    "UPDATE sessions SET parent_session_id = %s WHERE id = %s",
                    (parent, child),
                )
            return {
                "ok": True, "imported": len(imported), "skipped": len(skipped),
                "detached": detached, "imported_ids": imported, "skipped_ids": skipped,
                "errors": [],
            }

        return self._run(operation, read_only=False)

    def clear_messages(self, session_id: str) -> None:
        def operation(connection: Any) -> None:
            connection.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            connection.execute(
                """
                UPDATE sessions SET message_count = 0, tool_call_count = 0
                WHERE id = %s
                """,
                (session_id,),
            )

        self._run(operation, read_only=False)

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            try:
                (sessions_dir / f"{session_id}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass
        try:
            for path in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def _delegate_children_in(
        self, connection: Any, parent_ids: list[str]
    ) -> list[str]:
        found: set[str] = set(parent_ids)
        frontier = list(parent_ids)
        while frontier:
            children = self._rows(
                connection,
                """
                SELECT id FROM sessions
                WHERE parent_session_id = ANY(%s)
                  AND COALESCE(model_config, '') LIKE '%%"_delegate_from"%%'
                ORDER BY id ASC LIMIT %s
                """,
                (frontier, self._MAX_READ_ROWS),
                limit=self._MAX_READ_ROWS,
            )
            frontier = [str(row["id"]) for row in children if row["id"] not in found]
            found.update(frontier)
        return [session_id for session_id in found if session_id not in parent_ids]

    def _delete_ids_in(
        self, connection: Any, ids: list[str], *, cascade_delegates: bool
    ) -> tuple[list[str], list[str]]:
        existing = self._rows(
            connection,
            "SELECT id FROM sessions WHERE id = ANY(%s) ORDER BY id ASC LIMIT %s",
            (ids, self._MAX_READ_ROWS),
            limit=self._MAX_READ_ROWS,
        )
        roots = [str(row["id"]) for row in existing]
        if not roots:
            return [], []
        delegates = self._delegate_children_in(connection, roots) if cascade_delegates else []
        all_ids = roots + delegates
        connection.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ANY(%s)",
            (all_ids,),
        )
        connection.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (all_ids,))
        connection.execute("DELETE FROM sessions WHERE id = ANY(%s)", (all_ids,))
        return roots, delegates

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        roots, delegates = self._run(
            lambda connection: self._delete_ids_in(
                connection, [session_id], cascade_delegates=True
            ),
            read_only=False,
        )
        for removed in delegates + roots:
            self._remove_session_files(sessions_dir, removed)
        return bool(roots)

    def delete_session_if_empty(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        def operation(connection: Any) -> bool:
            cursor = connection.execute(
                """
                DELETE FROM sessions
                WHERE id = %s AND title IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM sessions child
                    WHERE child.parent_session_id = sessions.id
                  )
                """,
                (session_id,),
            )
            return bool(cursor.rowcount)

        deleted = self._run(operation, read_only=False)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return deleted

    def delete_sessions(
        self,
        session_ids: List[str],
        sessions_dir: Optional[Path] = None,
    ) -> int:
        ids = sorted({session_id for session_id in session_ids if isinstance(session_id, str) and session_id})
        if not ids:
            return 0
        roots, delegates = self._run(
            lambda connection: self._delete_ids_in(
                connection, ids, cascade_delegates=True
            ),
            read_only=False,
        )
        for removed in delegates + roots:
            self._remove_session_files(sessions_dir, removed)
        return len(roots)

    def count_empty_sessions(self) -> int:
        row = self._read_one(
            """
            SELECT COUNT(*) AS count FROM sessions
            WHERE message_count = 0 AND ended_at IS NOT NULL AND archived = 0
            """
        )
        return int(row["count"]) if row else 0

    def delete_empty_sessions(self, sessions_dir: Optional[Path] = None) -> int:
        def operation(connection: Any) -> list[str]:
            rows = self._rows(
                connection,
                """
                SELECT id FROM sessions
                WHERE message_count = 0 AND ended_at IS NOT NULL AND archived = 0
                ORDER BY id ASC LIMIT %s
                """,
                (self._MAX_READ_ROWS,),
                limit=self._MAX_READ_ROWS,
            )
            ids = [str(row["id"]) for row in rows]
            if not ids:
                return []
            connection.execute(
                "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ANY(%s)",
                (ids,),
            )
            connection.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (ids,))
            connection.execute("DELETE FROM sessions WHERE id = ANY(%s)", (ids,))
            return ids

        removed = self._run(operation, read_only=False)
        for session_id in removed:
            self._remove_session_files(sessions_dir, session_id)
        return len(removed)

    def prune_empty_ghost_sessions(self, sessions_dir: "Optional[Path]" = None) -> int:
        cutoff = time.time() - 86400

        def operation(connection: Any) -> list[str]:
            rows = self._rows(
                connection,
                """
                DELETE FROM sessions
                WHERE source = 'tui' AND title IS NULL AND ended_at IS NOT NULL
                  AND started_at < %s
                  AND NOT EXISTS (
                    SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
                RETURNING id
                """,
                (cutoff,),
                limit=self._MAX_READ_ROWS,
            )
            return [str(row["id"]) for row in rows]

        removed = self._run(operation, read_only=False)
        for session_id in removed:
            self._remove_session_files(sessions_dir, session_id)
        return len(removed)

    def finalize_orphaned_compression_sessions(self) -> int:
        def operation(connection: Any) -> int:
            cursor = connection.execute(
                """
                UPDATE sessions s
                SET ended_at = %s, end_reason = 'orphaned_compression'
                WHERE s.api_call_count = 0 AND s.end_reason IS NULL
                  AND s.ended_at IS NULL AND s.started_at < %s
                  AND s.parent_session_id IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM sessions p
                    WHERE p.id = s.parent_session_id
                      AND p.end_reason = 'compression' AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
                """,
                (time.time(), time.time() - 604800),
            )
            return int(cursor.rowcount or 0)

        return self._run(operation, read_only=False)

    @staticmethod
    def _prune_filter_where(
        *,
        started_before: Optional[float] = None,
        started_after: Optional[float] = None,
        source: Optional[str] = None,
        title_like: Optional[str] = None,
        end_reason: Optional[str] = None,
        cwd_prefix: Optional[str] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        archived: Optional[bool] = None,
        model_like: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        branch_like: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_tool_calls: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
    ) -> Tuple[str, list]:
        clauses = ["s.ended_at IS NOT NULL"]
        params: list[Any] = []
        scalar_filters = (
            (started_before, "s.started_at < %s"),
            (started_after, "s.started_at >= %s"),
            (source, "s.source = %s"),
            (end_reason, "s.end_reason = %s"),
            (min_messages, "s.message_count >= %s"),
            (max_messages, "s.message_count <= %s"),
            (user_id, "s.user_id = %s"),
            (chat_id, "s.chat_id = %s"),
            (chat_type, "s.chat_type = %s"),
            (min_tool_calls, "COALESCE(s.tool_call_count, 0) >= %s"),
            (max_tool_calls, "COALESCE(s.tool_call_count, 0) <= %s"),
        )
        for value, clause in scalar_filters:
            if value is not None and value != "":
                clauses.append(clause)
                params.append(value)
        for value, column in ((title_like, "title"), (model_like, "model"), (branch_like, "git_branch")):
            if value:
                clauses.append(f"LOWER(COALESCE(s.{column}, '')) LIKE %s")
                params.append(f"%{value.lower()}%")
        if cwd_prefix:
            prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
            clauses.append("(s.cwd = %s OR s.cwd LIKE %s OR s.cwd LIKE %s)")
            params.extend((prefix, f"{prefix}/%", f"{prefix}\\%"))
        if provider:
            clauses.append("LOWER(COALESCE(s.billing_provider, '')) = %s")
            params.append(provider.lower())
        if min_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) >= %s"
            )
            params.append(min_tokens)
        if max_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) <= %s"
            )
            params.append(max_tokens)
        if min_cost is not None:
            clauses.append("COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) >= %s")
            params.append(min_cost)
        if max_cost is not None:
            clauses.append("COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) <= %s")
            params.append(max_cost)
        if archived is not None:
            clauses.append("s.archived = %s")
            params.append(1 if archived else 0)
        return " AND ".join(clauses), params

    def list_prune_candidates(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> List[Dict[str, Any]]:
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - older_than_days * 86400
        where, params = self._prune_filter_where(source=source, **filters)
        return self._read_many(
            f"""
            SELECT s.id, s.source, s.title, s.model, s.started_at, s.ended_at,
                   s.message_count, s.archived
            FROM sessions s WHERE {where}
            ORDER BY s.started_at ASC, s.id ASC LIMIT %s
            """,
            (*params, self._MAX_READ_ROWS),
            limit=self._MAX_READ_ROWS,
        )

    def archive_sessions(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(
            older_than_days=older_than_days, source=source, **filters
        )

        def operation(connection: Any) -> int:
            count = 0
            for row in rows:
                lineage = self._compression_lineage_in(connection, row["id"])
                cursor = connection.execute(
                    "UPDATE sessions SET archived = 1 WHERE id = ANY(%s) AND archived = 0",
                    (lineage,),
                )
                count += int(cursor.rowcount or 0)
            return count

        self._run(operation, read_only=False)
        return len(rows)

    def prune_sessions(
        self,
        older_than_days: Optional[float] = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
        **filters,
    ) -> int:
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - older_than_days * 86400
        where, params = self._prune_filter_where(source=source, **filters)

        def operation(connection: Any) -> list[str]:
            rows = self._rows(
                connection,
                f"SELECT s.id FROM sessions s WHERE {where} ORDER BY s.id ASC LIMIT %s",
                (*params, self._MAX_READ_ROWS),
                limit=self._MAX_READ_ROWS,
            )
            ids = [str(row["id"]) for row in rows]
            if not ids:
                return []
            connection.execute(
                "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ANY(%s)",
                (ids,),
            )
            connection.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (ids,))
            connection.execute("DELETE FROM sessions WHERE id = ANY(%s)", (ids,))
            return ids

        removed = self._run(operation, read_only=False)
        for session_id in removed:
            self._remove_session_files(sessions_dir, session_id)
        return len(removed)

    def vacuum(self) -> int:
        """PostgreSQL autovacuum owns physical reclamation; report no SQLite work."""
        return 0

    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
        sessions_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            now = time.time()

            def operation(connection: Any) -> bool:
                row = self._row(
                    connection, "SELECT value FROM state_meta WHERE key = %s FOR UPDATE",
                    ("last_auto_prune",),
                )
                if row is not None:
                    try:
                        if now - float(row["value"]) < min_interval_hours * 3600:
                            return True
                    except (TypeError, ValueError):
                        pass
                connection.execute(
                    """
                    INSERT INTO state_meta (key, value) VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    ("last_auto_prune", str(now)),
                )
                return False

            if self._run(operation, read_only=False):
                result["skipped"] = True
                return result
            result["pruned"] = self.prune_sessions(
                older_than_days=retention_days, sessions_dir=sessions_dir
            )
            if vacuum and result["pruned"]:
                result["vacuumed"] = self.vacuum() >= 0
        except Exception as error:
            result["error"] = str(error)
        return result
