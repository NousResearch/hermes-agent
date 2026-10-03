"""Durable Todo authority projection for SessionDB.

The sparse materialization and bounded legacy migration live in their own mixin
so transcript persistence remains a focused module.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set, Tuple

_TODO_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
_TODO_STATUS_BY_MARKER = {" ": "pending", ">": "in_progress", "~": "cancelled", "x": "completed"}
_TODO_ITEM_RE = re.compile(
    r"^( *)- \[([x> ~])\] (.+?)\. (.+) \((pending|in_progress|completed|cancelled)\)$"
)
_TODO_ITEM_V2_RE = re.compile(r"^( *)- \[([x> ~])\] (\{.+\})$")
_TODO_SKILL_CALLS_RE = re.compile(
    r"skill_view\(name='[^'\r\n]+'\)(?:; skill_view\(name='[^'\r\n]+'\))*"
)
_TODO_SKILL_NOTICE_HEADER = (
    "[Skills pruned during compression — reload before acting on these tasks]"
)
_TODO_SKILL_NOTICE_PREFIX = (
    "The task list above crossed the compression boundary verbatim, but the skill "
    "instructions that governed it were pruned. Before executing any preserved task "
    "that depends on these skills, reload them first: "
)
_TODO_SKILL_NOTICE_SUFFIX = (
    ". After reloading, re-check that each pending task is still justified — findings "
    "recorded before the boundary may have invalidated it."
)


def _parse_todo_value(value: Any, depth: int = 0) -> Optional[List[Dict[str, Any]]]:
    """Match Desktop's bounded Todo JSON grammar; ``None`` means invalid."""
    from tools.todo_tool import MAX_TODO_RESULT_CHARS

    if depth > 2:
        return None
    try:
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if len(encoded) > MAX_TODO_RESULT_CHARS:
        return None
    if isinstance(value, list):
        parsed: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict) or item.get("status") not in _TODO_STATUSES:
                return None
            raw_id = item.get("id")
            raw_content = item.get("content")
            raw_parent = item.get("parent")
            if not isinstance(raw_id, str) or not isinstance(raw_content, str):
                return None
            if raw_parent is not None and not isinstance(raw_parent, str):
                return None
            item_id = raw_id.strip()
            content = raw_content.strip()
            parent = raw_parent.strip() if isinstance(raw_parent, str) else ""
            if not item_id or not content:
                return None
            parsed.append(
                {
                    "id": item_id,
                    "content": content,
                    "status": item["status"],
                    **({"parent": parent} if parent and parent != item_id else {}),
                }
            )
        return parsed
    if isinstance(value, str) and value.strip():
        try:
            return _parse_todo_value(json.loads(value), depth + 1)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(value, dict) and "todos" in value:
        return _parse_todo_value(value["todos"], depth + 1)
    return None


def _is_producer_todo_skill_notice(lines: List[str]) -> bool:
    if len(lines) != 3 or lines[0] != "" or lines[1] != _TODO_SKILL_NOTICE_HEADER:
        return False
    notice = lines[2]
    if not notice.startswith(_TODO_SKILL_NOTICE_PREFIX) or not notice.endswith(
        _TODO_SKILL_NOTICE_SUFFIX
    ):
        return False
    calls = notice[len(_TODO_SKILL_NOTICE_PREFIX) : -len(_TODO_SKILL_NOTICE_SUFFIX)]
    return _TODO_SKILL_CALLS_RE.fullmatch(calls) is not None


def _is_valid_legacy_todo_carrier(value: Any) -> bool:
    """Validate the exact V1/V2 standalone carrier grammar used by Desktop."""
    from tools.todo_tool import (
        MAX_TODO_RESULT_CHARS,
        TODO_INJECTION_FORMAT_V2,
        TODO_INJECTION_HEADER,
    )

    if not isinstance(value, str) or len(value) > MAX_TODO_RESULT_CHARS:
        return False
    if "\r" in value and "\r\n" not in value:
        return False
    lines = value.replace("\r\n", "\n").split("\n")
    if not lines or lines[0] != TODO_INJECTION_HEADER:
        return False
    versioned = len(lines) > 1 and lines[1] == TODO_INJECTION_FORMAT_V2
    index = 2 if versioned else 1
    todos: List[Dict[str, Any]] = []
    ancestors: List[str] = []
    ids: Set[str] = set()
    while index < len(lines):
        match = (_TODO_ITEM_V2_RE if versioned else _TODO_ITEM_RE).fullmatch(lines[index])
        if match is None:
            break
        indent, marker = match.group(1), match.group(2)
        if versioned:
            try:
                payload = json.loads(match.group(3))
            except (json.JSONDecodeError, TypeError):
                return False
            if (
                not isinstance(payload, dict)
                or set(payload) != {"id", "content", "status"}
                or not isinstance(payload["id"], str)
                or not isinstance(payload["content"], str)
                or payload["status"] not in _TODO_STATUSES
            ):
                return False
            item_id, content, status = payload["id"], payload["content"], payload["status"]
        else:
            item_id, content, status = match.group(3), match.group(4), match.group(5)
            if ". " in content:
                return False
        if (
            len(indent) % 2
            or item_id != item_id.strip()
            or content != content.strip()
            or not item_id
            or not content
            or item_id in ids
            or _TODO_STATUS_BY_MARKER.get(marker) != status
        ):
            return False
        depth = len(indent) // 2
        if depth > len(ancestors) or (depth > 0 and not ancestors[depth - 1]):
            return False
        parent = ancestors[depth - 1] if depth else None
        todos.append({"id": item_id, "content": content, "status": status, "parent": parent})
        ids.add(item_id)
        ancestors = ancestors[:depth]
        ancestors.append(item_id)
        index += 1
    if not todos:
        return False
    remainder = lines[index:]
    if remainder and not _is_producer_todo_skill_notice(remainder):
        return False
    retained_parents: Set[str] = set()
    for todo in reversed(todos):
        active = todo["status"] in {"pending", "in_progress"}
        if not active and todo["id"] not in retained_parents:
            return False
        if todo["parent"]:
            retained_parents.add(todo["parent"])
    return True



class SessionTodoMixin:
    """Bounded durable Todo authority lookup and legacy migration."""

    _TODO_STATE_MAX_MATERIALIZED_BYTES = 1_000_000
    _TODO_MIGRATION_PAYLOAD_BYTES = 65_536
    _TODO_MIGRATION_SMALL_FIELD_BYTES = 4_096
    _TODO_MIGRATION_ROWS_PER_SLICE = 64
    _TODO_MIGRATION_MAX_SLICES_PER_READ = 9
    _TODO_MIGRATION_MAX_PENDING_RESULTS = 16
    _TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ = 2_000_000
    _TODO_MIGRATION_MAX_PROGRESS_BYTES = 1_100_000
    _TODO_MIGRATION_MAX_SECONDS = 0.05
    # These are per-field read ceilings, not a global 64 KiB BLOB ceiling.
    # Legacy source payloads stay at 64 KiB; already-materialized authority and
    # durable migration-progress representations are larger but independently
    # capped. The decoded-byte budget below bounds their aggregate per attempt.
    _TODO_BLOB_READ_CAPS = {
        ("messages", "role"): 16,
        ("messages", "content"): _TODO_MIGRATION_PAYLOAD_BYTES,
        ("messages", "tool_call_id"): _TODO_MIGRATION_SMALL_FIELD_BYTES,
        ("messages", "tool_calls"): _TODO_MIGRATION_PAYLOAD_BYTES,
        ("messages", "tool_name"): _TODO_MIGRATION_SMALL_FIELD_BYTES,
        ("messages", "display_kind"): _TODO_MIGRATION_SMALL_FIELD_BYTES,
        ("messages", "display_metadata"): _TODO_MIGRATION_PAYLOAD_BYTES,
        ("messages", "todo_authority_json"): _TODO_STATE_MAX_MATERIALIZED_BYTES,
        ("todo_authorities", "authority_json"): _TODO_STATE_MAX_MATERIALIZED_BYTES,
        ("todo_pair_boundaries", "boundary_json"): _TODO_STATE_MAX_MATERIALIZED_BYTES,
        (
            "todo_authority_migrations",
            "pending_results_json",
        ): _TODO_MIGRATION_MAX_PROGRESS_BYTES,
        (
            "todo_authority_migrations",
            "deferred_pending_results_json",
        ): _TODO_MIGRATION_MAX_PROGRESS_BYTES,
        ("state_meta", "value"): 32,
    }
    # Same-host CPython tracemalloc regression ceiling for the adversarial
    # one-million-byte authority read+UTF-8 decode+JSON parse path. This is a
    # tested allocation ceiling, not a portable hard RSS limit.
    _TODO_NEAR_LIMIT_TRACEMALLOC_CEILING_BYTES = 6_000_000
    _TODO_STATE_LOOKUP_SQL = """
        SELECT message_id
        FROM todo_authorities INDEXED BY idx_todo_authorities_latest
        WHERE session_id = ? AND (active = 1 OR compacted = 1)
        ORDER BY message_id DESC
        LIMIT 1
    """
    _TODO_BLOB_COLUMNS = frozenset(_TODO_BLOB_READ_CAPS)

    @classmethod
    def _todo_blob_resource_contract_for_tests(cls) -> Dict[str, Any]:
        """Expose the executable Todo read/allocation contract to regressions."""
        if set(cls._TODO_BLOB_READ_CAPS) != set(cls._TODO_BLOB_COLUMNS):
            raise AssertionError("Todo BLOB source inventory is incomplete")
        return {
            "field_read_caps_bytes": {
                f"{table}.{column}": cap
                for (table, column), cap in sorted(cls._TODO_BLOB_READ_CAPS.items())
            },
            "single_result_source_bytes": cls._TODO_MIGRATION_PAYLOAD_BYTES,
            "materialized_authority_bytes": cls._TODO_STATE_MAX_MATERIALIZED_BYTES,
            "pending_result_count": cls._TODO_MIGRATION_MAX_PENDING_RESULTS,
            "pending_results_json_bytes": cls._TODO_MIGRATION_MAX_PROGRESS_BYTES,
            "attempt_decoded_bytes": cls._TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ,
            "migration_rows_per_slice": cls._TODO_MIGRATION_ROWS_PER_SLICE,
            "migration_slices_per_attempt": cls._TODO_MIGRATION_MAX_SLICES_PER_READ,
            "cooperative_deadline_seconds": cls._TODO_MIGRATION_MAX_SECONDS,
            "deadline_kind": "cooperative",
            "hard_deadline_seconds": None,
            "deadline_checkpoints": [
                "before_each_slice_after_first",
                "before_each_row_after_first_progress",
            ],
            "non_preemptible_operations": [
                "sqlite_statement",
                "bounded_blob_read",
                "utf8_decode",
                "json_decode",
                "json_encode",
                "sqlite_write",
            ],
            "near_limit_tracemalloc_ceiling_bytes": cls._TODO_NEAR_LIMIT_TRACEMALLOC_CEILING_BYTES,
        }

    @classmethod
    def _read_bounded_blob_text(
        cls,
        conn,
        *,
        table: str,
        column: str,
        row_id: int,
        max_bytes: int,
        budget_bytes: Optional[int] = None,
    ) -> Tuple[str, Optional[str], int]:
        """Read TEXT incrementally without materializing a value before its cap."""
        declared_cap = cls._TODO_BLOB_READ_CAPS.get((table, column))
        if declared_cap is None:
            raise ValueError(f"Unsupported bounded BLOB source: {table}.{column}")
        if max_bytes > declared_cap:
            raise ValueError(
                f"Todo BLOB read cap {max_bytes} exceeds declared "
                f"{table}.{column} cap {declared_cap}"
            )
        try:
            blob = conn.blobopen(table, column, int(row_id), readonly=True)
        except sqlite3.Error:
            return "missing", None, 0
        with blob:
            size = len(blob)
            if size > max_bytes:
                return "oversized", None, size
            if budget_bytes is not None and size > max(0, budget_bytes):
                return "budget", None, size
            raw = blob.read(size)
        try:
            return "ok", raw.decode("utf-8"), size
        except UnicodeDecodeError:
            return "invalid", None, size

    def _todo_migration_stats_for_tests(self) -> Dict[str, Any]:
        """Return the most recent bounded migration-attempt counters."""
        return dict(
            getattr(
                self,
                "_todo_last_migration_stats",
                {
                    "rows": 0,
                    "processed_rows": 0,
                    "selects": 0,
                    "decoded_bytes": 0,
                    "blob_reads": 0,
                    "max_blob_read_bytes": 0,
                    "max_source_blob_bytes": 0,
                    "pending_results": 0,
                    "elapsed": 0.0,
                },
            )
        )

    def _validate_todo_authority_text(
        self,
        raw: str,
        session_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Validate a complete materialized authority before publication."""
        try:
            messages = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(messages, list) or not 1 <= len(messages) <= 2:
            return None
        if any(not isinstance(message, dict) for message in messages):
            return None
        if any(message.get("session_id") != session_id for message in messages):
            return None

        if len(messages) == 1:
            message = messages[0]
            if message.get("role") != "user":
                return None
            metadata = message.get("display_metadata")
            snapshot = metadata.get("todo_snapshot") if isinstance(metadata, dict) else None
            structured = (
                isinstance(snapshot, dict)
                and set(snapshot) == {"todos"}
                and _parse_todo_value(snapshot.get("todos")) is not None
            )
            legacy = (
                message.get("display_kind") is None
                and metadata is None
                and _is_valid_legacy_todo_carrier(message.get("content"))
            )
            return messages if structured or legacy else None

        assistant, result = messages
        if assistant.get("role") != "assistant" or result.get("role") != "tool":
            return None
        call_id = str(result.get("tool_call_id") or "").strip()
        calls = assistant.get("tool_calls")
        if not call_id or not isinstance(calls, list) or len(calls) != 1:
            return None
        call = calls[0]
        if not isinstance(call, dict):
            return None
        function = call.get("function")
        function = function if isinstance(function, dict) else call
        name = str(function.get("name") or call.get("name") or "").strip()
        candidate_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
        if name != "todo" or candidate_id != call_id:
            return None
        if _parse_todo_value(result.get("content")) is None:
            return None
        return messages

    def _load_todo_authority(
        self,
        conn,
        session_id: str,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        row = conn.execute(self._TODO_STATE_LOOKUP_SQL, (session_id,)).fetchone()
        if row is None:
            return "absent", None
        status, raw, _size = self._read_bounded_blob_text(
            conn,
            table="todo_authorities",
            column="authority_json",
            row_id=int(row["message_id"]),
            max_bytes=self._TODO_STATE_MAX_MATERIALIZED_BYTES,
        )
        if status != "ok" or raw is None:
            return "invalid", None
        messages = self._validate_todo_authority_text(raw, session_id)
        return ("found", messages) if messages is not None else ("invalid", None)

    def _store_todo_authority(
        self,
        conn,
        *,
        message_id: int,
        session_id: str,
        authority_json: str,
    ) -> None:
        """Publish one bounded authority into the sparse lookup table."""
        conn.execute(
            "UPDATE messages SET todo_authority_json = ? WHERE id = ?",
            (authority_json, message_id),
        )
        conn.execute(
            """INSERT INTO todo_authorities
                   (message_id, session_id, authority_json, active, compacted)
               SELECT id, session_id, ?, active, compacted
                 FROM messages WHERE id = ? AND session_id = ?
               ON CONFLICT(message_id) DO UPDATE SET
                   session_id = excluded.session_id,
                   authority_json = excluded.authority_json,
                   active = excluded.active,
                   compacted = excluded.compacted""",
            (authority_json, message_id, session_id),
        )

    def _index_todo_message(
        self,
        conn,
        *,
        message_id: int,
        session_id: str,
        role: str,
        content: Any,
        tool_calls: Any,
        tool_call_id: Optional[str],
        tool_name: Optional[str],
        display_kind: Optional[str],
        display_metadata: Any,
        timestamp: float,
    ) -> None:
        """Publish the bounded Todo authority implied by one freshly inserted row."""
        self._record_todo_pair_boundary(
            conn,
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_name=tool_name,
            display_kind=display_kind,
            display_metadata=display_metadata,
            timestamp=timestamp,
        )
        authority_json = self._todo_carrier_authority_json(
            session_id=session_id,
            row_id=message_id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            display_kind=display_kind,
            display_metadata=display_metadata,
            timestamp=timestamp,
        )
        if authority_json is None:
            authority_json = self._todo_tool_result_authority_json(
                conn,
                session_id=session_id,
                row_id=message_id,
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                display_kind=display_kind,
                display_metadata=display_metadata,
                timestamp=timestamp,
            )
        if authority_json is not None:
            self._store_todo_authority(
                conn,
                message_id=message_id,
                session_id=session_id,
                authority_json=authority_json,
            )

    def _record_todo_pair_boundary(
        self,
        conn,
        *,
        message_id: int,
        session_id: str,
        role: str,
        content: Any,
        tool_calls: Any,
        tool_name: Optional[str],
        display_kind: Optional[str],
        display_metadata: Any,
        timestamp: float,
    ) -> None:
        """Record the latest non-tool boundary for constant-work result pairing."""
        if role not in {"assistant", "user", "system"}:
            return

        boundary_json = None
        if role == "assistant" and isinstance(tool_calls, list):
            todo_calls: List[Dict[str, Any]] = []
            overflow = False
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                function = function if isinstance(function, dict) else call
                name = str(function.get("name") or call.get("name") or "").strip()
                call_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
                if name != "todo" or not call_id:
                    continue
                todo_calls.append(call)
                if len(todo_calls) > self._TODO_MIGRATION_MAX_PENDING_RESULTS:
                    overflow = True
                    break
            if todo_calls and not overflow:
                projected = {
                    "id": message_id,
                    "session_id": session_id,
                    "role": role,
                    "content": "",
                    "tool_calls": todo_calls,
                    "tool_name": tool_name,
                    "display_kind": display_kind,
                    "display_metadata": display_metadata if isinstance(display_metadata, dict) else None,
                    "timestamp": timestamp,
                }
                encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
                if len(encoded.encode("utf-8")) <= self._TODO_STATE_MAX_MATERIALIZED_BYTES:
                    boundary_json = encoded

        conn.execute(
            "UPDATE messages SET todo_pair_boundary_json = ? WHERE id = ?",
            (boundary_json, message_id),
        )
        conn.execute(
            """INSERT INTO todo_pair_boundaries
                   (session_id, message_id, role, boundary_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   message_id = excluded.message_id,
                   role = excluded.role,
                   boundary_json = excluded.boundary_json
               WHERE excluded.message_id >= todo_pair_boundaries.message_id""",
            (session_id, message_id, role, boundary_json),
        )

    def _todo_carrier_authority_json(
        self,
        *,
        session_id: str,
        row_id: int,
        role: str,
        content: Any,
        tool_call_id: Optional[str],
        tool_name: Optional[str],
        display_kind: Optional[str],
        display_metadata: Any,
        timestamp: float,
    ) -> Optional[str]:
        """Materialize one trusted Todo carrier at the persistence boundary."""
        metadata = display_metadata if isinstance(display_metadata, dict) else None
        snapshot = metadata.get("todo_snapshot") if metadata is not None else None
        structured = (
            role == "user"
            and isinstance(snapshot, dict)
            and set(snapshot) == {"todos"}
            and _parse_todo_value(snapshot.get("todos")) is not None
        )
        legacy = (
            role == "user"
            and display_kind is None
            and display_metadata is None
            and _is_valid_legacy_todo_carrier(content)
        )
        if not structured and not legacy:
            return None

        projected = {
            "id": row_id,
            "session_id": session_id,
            "role": role,
            "content": "" if structured else content,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "display_kind": display_kind,
            "display_metadata": metadata,
            "timestamp": timestamp,
        }
        encoded = json.dumps([projected], ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self._TODO_STATE_MAX_MATERIALIZED_BYTES:
            return None
        return encoded

    def _todo_tool_result_authority_json(
        self,
        conn,
        *,
        session_id: str,
        row_id: int,
        role: str,
        content: Any,
        tool_call_id: Optional[str],
        tool_name: Optional[str],
        display_kind: Optional[str],
        display_metadata: Any,
        timestamp: float,
    ) -> Optional[str]:
        """Materialize a Todo result only when its nearest call boundary matches."""
        call_id = str(tool_call_id or "").strip()
        if role != "tool" or not call_id:
            return None

        boundary = conn.execute(
            """SELECT rowid, message_id, role
                 FROM todo_pair_boundaries
                WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if (
            boundary is None
            or boundary["role"] != "assistant"
            or boundary["message_id"] >= row_id
        ):
            return None
        boundary_status, boundary_raw, _boundary_size = self._read_bounded_blob_text(
            conn,
            table="todo_pair_boundaries",
            column="boundary_json",
            row_id=int(boundary["rowid"]),
            max_bytes=self._TODO_STATE_MAX_MATERIALIZED_BYTES,
        )
        if boundary_status != "ok" or boundary_raw is None:
            return None
        try:
            boundary_message = json.loads(boundary_raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(boundary_message, dict):
            return None
        calls = boundary_message.get("tool_calls")
        if not isinstance(calls, list):
            return None

        matching_call = None
        remaining_calls = []
        for call in calls:
            if isinstance(call, dict):
                function = call.get("function")
                function = function if isinstance(function, dict) else call
                name = str(function.get("name") or call.get("name") or "").strip()
                candidate_id = str(
                    call.get("id") or call.get("tool_call_id") or ""
                ).strip()
                if name == "todo" and candidate_id == call_id:
                    # A call ID is one-shot. Keep the last duplicate declaration,
                    # matching the renderer's Map semantics, but consume every
                    # declaration before validating the result payload.
                    matching_call = call
                    continue
            remaining_calls.append(call)
        if matching_call is None:
            return None

        boundary_message["tool_calls"] = remaining_calls
        consumed_boundary_json = json.dumps(
            boundary_message,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE messages SET todo_pair_boundary_json = ? "
            "WHERE id = ? AND session_id = ?",
            (consumed_boundary_json, int(boundary["message_id"]), session_id),
        )
        conn.execute(
            "UPDATE todo_pair_boundaries SET boundary_json = ? "
            "WHERE rowid = ? AND session_id = ? AND message_id = ?",
            (
                consumed_boundary_json,
                int(boundary["rowid"]),
                session_id,
                int(boundary["message_id"]),
            ),
        )

        if isinstance(content, str):
            if len(content) > self._TODO_STATE_MAX_MATERIALIZED_BYTES:
                return None
            if len(content.encode("utf-8")) > self._TODO_STATE_MAX_MATERIALIZED_BYTES:
                return None
        if _parse_todo_value(content) is None:
            return None

        result_metadata = display_metadata if isinstance(display_metadata, dict) else None
        evidence = [
            {
                "id": boundary_message.get("id"),
                "session_id": session_id,
                "role": "assistant",
                "content": "",
                "tool_call_id": None,
                "tool_name": boundary_message.get("tool_name"),
                "display_kind": boundary_message.get("display_kind"),
                "display_metadata": boundary_message.get("display_metadata"),
                "timestamp": boundary_message.get("timestamp"),
                "tool_calls": [matching_call],
            },
            {
                "id": row_id,
                "session_id": session_id,
                "role": "tool",
                "content": content,
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "display_kind": display_kind,
                "display_metadata": result_metadata,
                "timestamp": timestamp,
            },
        ]
        encoded = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self._TODO_STATE_MAX_MATERIALIZED_BYTES:
            return None
        return encoded

    def _advance_todo_authority_migration(self, conn, session_id: str) -> str:
        """Advance one bounded legacy scan; return found, complete, or pending."""
        started = time.perf_counter()
        stats: Dict[str, Any] = {
            "rows": 0,
            "processed_rows": 0,
            "selects": 0,
            "decoded_bytes": 0,
            "blob_reads": 0,
            "max_blob_read_bytes": 0,
            "max_source_blob_bytes": 0,
            "pending_results": 0,
            "elapsed": 0.0,
        }
        self._todo_last_migration_stats = stats
        pending_results: List[Dict[str, Any]] = []

        def finish(status: str) -> str:
            stats["pending_results"] = len(pending_results)
            stats["elapsed"] = time.perf_counter() - started
            return status

        stats["selects"] += 1
        if conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
        ).fetchone() is None:
            stats["progress_action"] = "session_absent"
            return finish("absent")

        def read_accounted(
            *, table: str, column: str, row_id: int, max_bytes: int
        ) -> Tuple[str, Optional[str]]:
            remaining = (
                self._TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ
                - int(stats["decoded_bytes"])
            )
            status, value, size = self._read_bounded_blob_text(
                conn,
                table=table,
                column=column,
                row_id=row_id,
                max_bytes=max_bytes,
                budget_bytes=remaining,
            )
            stats["blob_reads"] += 1
            stats["max_source_blob_bytes"] = max(
                int(stats["max_source_blob_bytes"]),
                size,
            )
            if status in {"ok", "invalid"}:
                stats["decoded_bytes"] += size
                stats["max_blob_read_bytes"] = max(
                    int(stats["max_blob_read_bytes"]),
                    size,
                )
            return status, value

        stats["selects"] += 1
        high_water_row = conn.execute(
            "SELECT rowid FROM state_meta "
            "WHERE key = 'todo_authority_legacy_high_water'"
        ).fetchone()
        high_water = 0
        if high_water_row is not None:
            status, high_water_text = read_accounted(
                table="state_meta",
                column="value",
                row_id=int(high_water_row["rowid"]),
                max_bytes=32,
            )
            if status == "ok" and high_water_text is not None:
                try:
                    high_water = max(0, int(high_water_text))
                except ValueError:
                    high_water = 0

        stats["selects"] += 1
        progress = conn.execute(
            """SELECT rowid, before_message_id, complete, phase,
                      authority_checked_message_id
                 FROM todo_authority_migrations WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        migration_phase = (
            str(progress["phase"] or "scan") if progress is not None else "scan"
        )
        if migration_phase not in {"scan", "authority_probe", "authority_checked"}:
            migration_phase = "scan"
        authority_checked_message_id = (
            int(progress["authority_checked_message_id"])
            if progress is not None
            and progress["authority_checked_message_id"] is not None
            else None
        )
        stats["phase"] = migration_phase
        if progress is not None and progress["complete"]:
            return finish("complete")

        before_message_id = (
            int(progress["before_message_id"])
            if progress is not None and progress["before_message_id"] is not None
            else high_water + 1
        )
        if progress is not None:
            status, pending_text = read_accounted(
                table="todo_authority_migrations",
                column="pending_results_json",
                row_id=int(progress["rowid"]),
                max_bytes=self._TODO_MIGRATION_MAX_PROGRESS_BYTES,
            )
            if status == "ok" and pending_text:
                try:
                    decoded_pending = json.loads(pending_text)
                except (json.JSONDecodeError, TypeError):
                    decoded_pending = None
                if isinstance(decoded_pending, list):
                    pending_results = [
                        value
                        for value in decoded_pending
                        if isinstance(value, dict)
                    ][: self._TODO_MIGRATION_MAX_PENDING_RESULTS]

        deadline = started + self._TODO_MIGRATION_MAX_SECONDS
        complete = high_water <= 0
        slices = 0
        stop_for_budget = False

        def read_field(
            row: sqlite3.Row,
            column: str,
            max_bytes: int,
        ) -> Tuple[str, Optional[str]]:
            value_type = row[f"{column}_type"]
            if value_type == "null":
                return "null", None
            if value_type not in {"text", "blob"}:
                return "invalid", None
            return read_accounted(
                table="messages",
                column=column,
                row_id=int(row["id"]),
                max_bytes=max_bytes,
            )

        def read_optional_fields(
            row: sqlite3.Row,
            columns: Tuple[str, ...],
        ) -> Tuple[str, Dict[str, Optional[str]]]:
            values: Dict[str, Optional[str]] = {}
            for column in columns:
                status, value = read_field(
                    row,
                    column,
                    self._TODO_MIGRATION_SMALL_FIELD_BYTES,
                )
                if status == "budget":
                    return status, {}
                if status not in {"ok", "null"}:
                    return "invalid", {}
                values[column] = value
            return "ok", values

        def consume(row_id: int) -> None:
            nonlocal before_message_id, migration_phase, authority_checked_message_id
            before_message_id = row_id
            stats["processed_rows"] += 1
            if authority_checked_message_id == row_id:
                migration_phase = "scan"
                authority_checked_message_id = None

        def remember_pending_result(candidate: Dict[str, Any]) -> None:
            """Keep the earliest bounded row for one call while scanning backward."""
            nonlocal pending_results
            call_id = str(candidate.get("tool_call_id") or "").strip()
            if not call_id:
                return
            replacement_index = next(
                (
                    index
                    for index, pending in enumerate(pending_results)
                    if str(pending.get("tool_call_id") or "").strip() == call_id
                ),
                None,
            )
            if (
                replacement_index is None
                and len(pending_results) >= self._TODO_MIGRATION_MAX_PENDING_RESULTS
            ):
                return

            tombstone = {"id": candidate.get("id"), "tool_call_id": call_id}
            variants = [candidate]
            if candidate != tombstone:
                variants.append(tombstone)
            for variant in variants:
                updated = list(pending_results)
                if replacement_index is None:
                    updated.append(variant)
                else:
                    updated[replacement_index] = variant
                encoded = json.dumps(
                    updated,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(encoded.encode("utf-8")) <= self._TODO_MIGRATION_MAX_PROGRESS_BYTES:
                    pending_results = updated
                    return

        while not complete and slices < self._TODO_MIGRATION_MAX_SLICES_PER_READ:
            if slices and time.perf_counter() >= deadline:
                break
            stats["selects"] += 1
            rows = conn.execute(
                """SELECT id,
                          CASE WHEN typeof(timestamp) IN ('integer', 'real')
                               THEN timestamp END AS timestamp,
                          typeof(role) AS role_type,
                          typeof(content) AS content_type,
                          typeof(tool_call_id) AS tool_call_id_type,
                          typeof(tool_calls) AS tool_calls_type,
                          typeof(tool_name) AS tool_name_type,
                          typeof(display_kind) AS display_kind_type,
                          typeof(display_metadata) AS display_metadata_type,
                          typeof(todo_authority_json) AS todo_authority_json_type
                     FROM messages INDEXED BY idx_messages_session_id
                    WHERE session_id = ? AND id < ? AND id <= ?
                      AND (active = 1 OR compacted = 1)
                    ORDER BY id DESC LIMIT ?""",
                (
                    session_id,
                    before_message_id,
                    high_water,
                    self._TODO_MIGRATION_ROWS_PER_SLICE,
                ),
            ).fetchall()
            slices += 1
            stats["rows"] += len(rows)
            if not rows:
                if migration_phase == "authority_probe" and progress is not None:
                    conn.execute(
                        """UPDATE todo_authority_migrations
                              SET pending_results_json = deferred_pending_results_json,
                                  deferred_pending_results_json = '[]',
                                  phase = 'scan', authority_checked_message_id = NULL
                            WHERE session_id = ?""",
                        (session_id,),
                    )
                    stats["progress_action"] = "restored_after_missing_probe_row"
                    return finish("pending")
                complete = True
                break

            for row in rows:
                if stats["processed_rows"] and time.perf_counter() >= deadline:
                    stop_for_budget = True
                    break
                if stats["decoded_bytes"] >= self._TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ:
                    stop_for_budget = True
                    break

                row_id = int(row["id"])
                if (
                    migration_phase == "authority_probe"
                    and authority_checked_message_id != row_id
                ):
                    conn.execute(
                        """UPDATE todo_authority_migrations
                              SET pending_results_json = deferred_pending_results_json,
                                  deferred_pending_results_json = '[]',
                                  phase = 'scan', authority_checked_message_id = NULL
                            WHERE session_id = ?""",
                        (session_id,),
                    )
                    stats["progress_action"] = "restored_after_probe_row_changed"
                    return finish("pending")
                if (
                    migration_phase == "authority_checked"
                    and authority_checked_message_id != row_id
                ):
                    migration_phase = "scan"
                    authority_checked_message_id = None

                skip_authority = (
                    migration_phase == "authority_checked"
                    and authority_checked_message_id == row_id
                )
                authority_status = "skipped"
                authority_raw = None
                if not skip_authority:
                    authority_status, authority_raw = read_field(
                        row,
                        "todo_authority_json",
                        self._TODO_STATE_MAX_MATERIALIZED_BYTES,
                    )
                    if authority_status == "budget":
                        if pending_results and progress is not None:
                            conn.execute(
                                """UPDATE todo_authority_migrations
                                      SET deferred_pending_results_json = pending_results_json,
                                          pending_results_json = '[]',
                                          phase = 'authority_probe',
                                          authority_checked_message_id = ?
                                    WHERE session_id = ?""",
                                (row_id, session_id),
                            )
                            stats["progress_action"] = "deferred_pending_for_authority_probe"
                            return finish("pending")
                        stop_for_budget = True
                        break
                    if authority_status == "ok" and authority_raw is not None:
                        authority = self._validate_todo_authority_text(
                            authority_raw,
                            session_id,
                        )
                        if authority is not None:
                            self._store_todo_authority(
                                conn,
                                message_id=row_id,
                                session_id=session_id,
                                authority_json=authority_raw,
                            )
                            consume(row_id)
                            complete = True
                            migration_phase = "scan"
                            authority_checked_message_id = None
                            break
                    if migration_phase == "authority_probe" and progress is not None:
                        conn.execute(
                            """UPDATE todo_authority_migrations
                                  SET pending_results_json = deferred_pending_results_json,
                                      deferred_pending_results_json = '[]',
                                      phase = 'authority_checked',
                                      authority_checked_message_id = ?
                                WHERE session_id = ?""",
                            (row_id, session_id),
                        )
                        stats["progress_action"] = "authority_probe_completed"
                        return finish("pending")
                    if authority_status in {"ok", "invalid"}:
                        migration_phase = "authority_checked"
                        authority_checked_message_id = row_id
                        stats["progress_action"] = "authority_read_checkpointed"
                else:
                    migration_phase = "scan"
                    authority_checked_message_id = None

                role_status, role = read_field(row, "role", 16)
                if role_status == "budget":
                    stop_for_budget = True
                    break
                if role_status != "ok" or role not in {"assistant", "user", "system", "tool"}:
                    consume(row_id)
                    continue

                if role in {"assistant", "user", "system"}:
                    if role == "assistant" and pending_results:
                        calls_status, tool_calls_raw = read_field(
                            row,
                            "tool_calls",
                            self._TODO_MIGRATION_PAYLOAD_BYTES,
                        )
                        if calls_status == "budget":
                            stop_for_budget = True
                            break
                        calls = None
                        if calls_status == "ok" and tool_calls_raw:
                            try:
                                calls = json.loads(tool_calls_raw)
                            except (json.JSONDecodeError, TypeError):
                                calls = None
                        if isinstance(calls, list):
                            todo_calls_by_id: Dict[str, Dict[str, Any]] = {}
                            for call in calls:
                                if not isinstance(call, dict):
                                    continue
                                function = call.get("function")
                                function = function if isinstance(function, dict) else call
                                name = str(
                                    function.get("name") or call.get("name") or ""
                                ).strip()
                                candidate_id = str(
                                    call.get("id") or call.get("tool_call_id") or ""
                                ).strip()
                                if name == "todo" and candidate_id:
                                    # The renderer's Map keeps the last declaration
                                    # for a duplicate ID and consumes that ID once.
                                    todo_calls_by_id[candidate_id] = call

                            earliest_by_call_id: Dict[
                                str, Tuple[int, Dict[str, Any]]
                            ] = {}
                            for result in pending_results:
                                call_id = str(
                                    result.get("tool_call_id") or ""
                                ).strip()
                                if call_id not in todo_calls_by_id:
                                    continue
                                try:
                                    result_id = int(result.get("id"))
                                except (TypeError, ValueError):
                                    continue
                                current = earliest_by_call_id.get(call_id)
                                if current is None or result_id < current[0]:
                                    earliest_by_call_id[call_id] = (result_id, result)

                            matching_result = None
                            matching_call = None
                            matching_result_id = -1
                            for call_id, (result_id, result) in earliest_by_call_id.items():
                                # The first row consumes this call even when its
                                # payload is malformed. Only a valid first result
                                # may become authority; later duplicates are ignored.
                                if _parse_todo_value(result.get("content")) is None:
                                    continue
                                if result_id > matching_result_id:
                                    matching_result = result
                                    matching_call = todo_calls_by_id[call_id]
                                    matching_result_id = result_id
                            if matching_result is not None and matching_call is not None:
                                fields_status, fields = read_optional_fields(
                                    row,
                                    ("tool_name", "display_kind"),
                                )
                                metadata_status, metadata_raw = read_field(
                                    row,
                                    "display_metadata",
                                    self._TODO_MIGRATION_PAYLOAD_BYTES,
                                )
                                if fields_status == "budget" or metadata_status == "budget":
                                    stop_for_budget = True
                                    break
                                assistant_metadata = None
                                metadata_valid = metadata_status == "null"
                                if metadata_status == "ok" and metadata_raw is not None:
                                    try:
                                        decoded_metadata = json.loads(metadata_raw)
                                    except (json.JSONDecodeError, TypeError):
                                        decoded_metadata = None
                                    if isinstance(decoded_metadata, dict):
                                        assistant_metadata = decoded_metadata
                                        metadata_valid = True
                                if fields_status == "ok" and metadata_valid:
                                    evidence = [
                                        {
                                            "id": row_id,
                                            "session_id": session_id,
                                            "role": "assistant",
                                            "content": "",
                                            "tool_call_id": None,
                                            "tool_name": fields["tool_name"],
                                            "display_kind": fields["display_kind"],
                                            "display_metadata": assistant_metadata,
                                            "timestamp": row["timestamp"],
                                            "tool_calls": [matching_call],
                                        },
                                        matching_result,
                                    ]
                                    encoded = json.dumps(
                                        evidence,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    if (
                                        len(encoded.encode("utf-8"))
                                        <= self._TODO_STATE_MAX_MATERIALIZED_BYTES
                                        and self._validate_todo_authority_text(
                                            encoded,
                                            session_id,
                                        )
                                        is not None
                                    ):
                                        self._store_todo_authority(
                                            conn,
                                            message_id=int(matching_result["id"]),
                                            session_id=session_id,
                                            authority_json=encoded,
                                        )
                                        consume(row_id)
                                        complete = True
                                        break
                    if stop_for_budget or complete:
                        break
                    pending_results = []
                    if role == "user":
                        metadata_status, metadata_raw = read_field(
                            row,
                            "display_metadata",
                            self._TODO_MIGRATION_PAYLOAD_BYTES,
                        )
                        if metadata_status == "budget":
                            stop_for_budget = True
                            break
                        metadata = None
                        metadata_valid = metadata_status == "null"
                        if metadata_status == "ok" and metadata_raw is not None:
                            metadata = self._decode_display_metadata(metadata_raw)
                            metadata_valid = isinstance(metadata, dict)

                        kind_status, display_kind = read_field(
                            row,
                            "display_kind",
                            self._TODO_MIGRATION_SMALL_FIELD_BYTES,
                        )
                        if kind_status == "budget":
                            stop_for_budget = True
                            break
                        kind_valid = kind_status in {"ok", "null"}
                        snapshot = (
                            metadata.get("todo_snapshot")
                            if isinstance(metadata, dict)
                            else None
                        )
                        structured = (
                            isinstance(snapshot, dict)
                            and set(snapshot) == {"todos"}
                            and _parse_todo_value(snapshot.get("todos")) is not None
                        )

                        content = "" if structured else None
                        content_status = "ok"
                        if not structured and metadata_status == "null" and kind_status == "null":
                            content_status, content_raw = read_field(
                                row,
                                "content",
                                self._TODO_MIGRATION_PAYLOAD_BYTES,
                            )
                            if content_status == "budget":
                                stop_for_budget = True
                                break
                            if content_status == "ok" and content_raw is not None:
                                content = self._decode_content(content_raw)

                        candidate_shape = structured or (
                            metadata_status == "null"
                            and kind_status == "null"
                            and content_status == "ok"
                            and _is_valid_legacy_todo_carrier(content)
                        )
                        if metadata_valid and kind_valid and candidate_shape:
                            fields_status, fields = read_optional_fields(
                                row,
                                ("tool_call_id", "tool_name"),
                            )
                            if fields_status == "budget":
                                stop_for_budget = True
                                break
                            if fields_status == "ok":
                                authority_json = self._todo_carrier_authority_json(
                                    session_id=session_id,
                                    row_id=row_id,
                                    role=role,
                                    content=content,
                                    tool_call_id=fields["tool_call_id"],
                                    tool_name=fields["tool_name"],
                                    display_kind=display_kind,
                                    display_metadata=metadata,
                                    timestamp=row["timestamp"],
                                )
                                if authority_json is not None:
                                    self._store_todo_authority(
                                        conn,
                                        message_id=row_id,
                                        session_id=session_id,
                                        authority_json=authority_json,
                                    )
                                    consume(row_id)
                                    complete = True
                                    break
                    if stop_for_budget or complete:
                        break
                    consume(row_id)
                elif role == "tool":
                    call_id_status, call_id_raw = read_field(
                        row,
                        "tool_call_id",
                        self._TODO_MIGRATION_SMALL_FIELD_BYTES,
                    )
                    fields_status, fields = read_optional_fields(
                        row,
                        ("tool_name", "display_kind"),
                    )
                    content_status, content_raw = read_field(
                        row,
                        "content",
                        self._TODO_MIGRATION_PAYLOAD_BYTES,
                    )
                    metadata_status, metadata_raw = read_field(
                        row,
                        "display_metadata",
                        self._TODO_MIGRATION_PAYLOAD_BYTES,
                    )
                    if (
                        call_id_status == "budget"
                        or fields_status == "budget"
                        or content_status == "budget"
                        or metadata_status == "budget"
                    ):
                        stop_for_budget = True
                        break
                    content = (
                        self._decode_content(content_raw)
                        if content_status == "ok" and content_raw is not None
                        else None
                    )
                    metadata = None
                    metadata_valid = metadata_status == "null"
                    if metadata_status == "ok" and metadata_raw is not None:
                        metadata = self._decode_display_metadata(metadata_raw)
                        metadata_valid = isinstance(metadata, dict)
                    call_id = (
                        str(call_id_raw or "").strip()
                        if call_id_status == "ok"
                        else ""
                    )
                    if call_id:
                        candidate: Dict[str, Any] = {
                            "id": row_id,
                            "tool_call_id": call_id,
                        }
                        if (
                            fields_status == "ok"
                            and content_status == "ok"
                            and metadata_valid
                            and _parse_todo_value(content) is not None
                        ):
                            candidate = {
                                "id": row_id,
                                "session_id": session_id,
                                "role": "tool",
                                "content": content,
                                "tool_call_id": call_id,
                                "tool_name": fields["tool_name"],
                                "display_kind": fields["display_kind"],
                                "display_metadata": metadata,
                                "timestamp": row["timestamp"],
                            }
                        # Backward scanning sees later duplicates first. Replacing
                        # by call ID retains the earliest row, including a compact
                        # tombstone when that consuming row is not a valid Todo.
                        remember_pending_result(candidate)
                    consume(row_id)

            if stop_for_budget:
                break
            if len(rows) < self._TODO_MIGRATION_ROWS_PER_SLICE:
                complete = True

        pending_json = json.dumps(
            pending_results,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(pending_json.encode("utf-8")) > self._TODO_MIGRATION_MAX_PROGRESS_BYTES:
            pending_results = []
            pending_json = "[]"
        persisted_phase = "scan" if complete else migration_phase
        persisted_checked_message_id = (
            None if complete else authority_checked_message_id
        )
        conn.execute(
            """INSERT INTO todo_authority_migrations
                   (session_id, before_message_id, pending_results_json,
                    deferred_pending_results_json, phase,
                    authority_checked_message_id, complete)
               VALUES (?, ?, ?, '[]', ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   before_message_id = excluded.before_message_id,
                   pending_results_json = excluded.pending_results_json,
                   deferred_pending_results_json = '[]',
                   phase = excluded.phase,
                   authority_checked_message_id = excluded.authority_checked_message_id,
                   complete = excluded.complete""",
            (
                session_id,
                before_message_id,
                pending_json,
                persisted_phase,
                persisted_checked_message_id,
                1 if complete else 0,
            ),
        )
        stats["selects"] += 1
        if conn.execute(self._TODO_STATE_LOOKUP_SQL, (session_id,)).fetchone() is not None:
            return finish("found")
        return finish("complete" if complete else "pending")

    def get_todo_state_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return at most one validated Todo evidence pair for a whole session.

        Materialized authorities are validated at the persistence boundary and
        read through one partial-index lookup. Rejected transcript rows are not
        visited, decoded, or retained by this method.
        """
        with self._read_ctx() as conn:
            state, messages = self._load_todo_authority(conn, session_id)
        if state == "found":
            return messages
        if state == "invalid":
            return []

        status = self._execute_write(
            lambda conn: self._advance_todo_authority_migration(conn, session_id),
            patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S,
        )
        if status == "pending":
            return None
        with self._read_ctx() as conn:
            state, messages = self._load_todo_authority(conn, session_id)
        return messages if state == "found" else []
