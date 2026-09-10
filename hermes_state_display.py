"""Durable command output for UI display, separate from the model transcript."""

from __future__ import annotations

import re
import time
import uuid

from hermes_state_common import _placeholders


def _display_event_message(row):
    return {
        "id": "display:" + row["event_id"],
        "role": "system",
        "content": "slash:" + row["command"] + "\n" + row["output"],
        "timestamp": row["created_at"],
        "display_kind": "command_result",
    }


class SessionDisplayMixin:
    def _display_events_available(self) -> bool:
        # Read-only attachments to older profile DBs must not migrate them.
        return self._read_one(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_display_events'"
        ) is not None

    def _display_event_lineage(self, session_id: str) -> list[str]:
        lineage = self._resume_lineage_ids(session_id)
        # A branch's later compression tip is not itself marked as a branch.
        # Keep its branch-local events without reaching the pre-branch parent.
        for index in range(len(lineage) - 1, -1, -1):
            if self._is_explicit_branch_session(lineage[index]):
                return lineage[index:]
        return lineage

    def append_display_event(self, session_id: str, event_id: str, command: str, output: str) -> dict:
        """Persist an immutable UUID event; identical retries return its original UI row.

        A reused UUID with a different owner or payload raises ValueError. The caller owns
        session creation and command execution; this method never creates or retries either.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        if not isinstance(event_id, str) or str(uuid.UUID(event_id)) != event_id:
            raise ValueError("event_id must be a canonical UUID")
        if not isinstance(command, str) or not re.fullmatch(r"/?[A-Za-z0-9][A-Za-z0-9_-]*", command):
            raise ValueError("command must be a token without arguments")
        if not isinstance(output, str):
            raise ValueError("output must be text")
        command = "/" + command.removeprefix("/")

        def _do(conn):
            row = conn.execute(
                "SELECT * FROM session_display_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is not None:
                if (row["session_id"], row["command"], row["output"]) != (session_id, command, output):
                    raise ValueError("display event UUID already has a different owner or payload")
                return _display_event_message(row)
            created_at = time.time()
            conn.execute(
                "INSERT INTO session_display_events (event_id, session_id, command, output, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (event_id, session_id, command, output, created_at),
            )
            conn.execute(
                "UPDATE sessions SET last_activity_at = ? "
                "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
                (created_at, session_id, created_at),
            )
            return _display_event_message(dict(
                event_id=event_id, command=command, output=output, created_at=created_at))

        return self._execute_write(_do)

    def get_display_events(self, session_id: str, include_ancestors: bool = False) -> list[dict]:
        """UI events in timestamp/insertion order; lineage opt-in never crosses an explicit branch."""
        if not self._display_events_available():
            return []
        session_ids = self._display_event_lineage(session_id) if include_ancestors else [session_id]
        return [_display_event_message(row) for row in self._read_all(
            f"SELECT * FROM session_display_events WHERE session_id IN ({_placeholders(session_ids)}) "
            "ORDER BY created_at, ordinal", session_ids,
        )]

    def get_display_messages(
        self, session_id: str, limit: int | None = None, offset: int = 0,
        latest: bool = False, include_compacted: bool = False,
    ) -> list[dict]:
        """Page the UNION of real messages and UI events; model reads remain unchanged.

        Order is effective timestamp, real-message-before-event, then logical id/event ordinal.
        Real timestamps are prefix-maxed for sorting only: clock regression and compaction's
        carried-forward timestamps must not reorder the existing transcript. Output times stay original.
        ``latest`` offsets from the newest end but returns the page oldest-first. There is
        deliberately no message-id cursor: an integer message id cannot address this union.
        Events follow the resume lineage; explicit branches retain only their own events.
        """
        if (limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0)
                or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
            raise ValueError("limit and offset must be nonnegative integers")
        if not self._display_events_available():
            return self.get_messages(session_id, limit=limit, offset=offset, latest=latest,
                                     include_compacted=include_compacted)
        if include_compacted and not self._ensure_display_order(session_id):
            # The existing read-only legacy projection owns compaction deduplication.
            # It cannot backfill display_order; preserve that fallback before paging.
            messages = self.get_messages(session_id, include_compacted=True)
            events = self.get_display_events(session_id, include_ancestors=True)
            ordered = []
            effective_time = float("-inf")
            for index, row in enumerate(messages):
                effective_time = max(effective_time, row["timestamp"])
                ordered.append(((effective_time, 0, index), row))
            ordered.extend(((row["timestamp"], 1, index), row) for index, row in enumerate(events))
            merged = [row for _, row in sorted(ordered, key=lambda item: item[0])]
            return merged[::-1][offset:][:limit][::-1] if latest else merged[offset:][:limit]
        real_rows = (
            "SELECT id, timestamp, display_order AS ordering, ROW_NUMBER() OVER ("
            "PARTITION BY display_order ORDER BY active DESC, id DESC) AS rank "
            "FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)"
            if include_compacted else
            "SELECT id, timestamp, id AS ordering, 1 AS rank "
            "FROM messages WHERE session_id = ? AND active = 1"
        )
        direction = "DESC" if latest else "ASC"
        event_sessions = self._display_event_lineage(session_id)
        query = f"""WITH real_rows AS ({real_rows}), timeline AS (
                SELECT id AS row_id, MAX(timestamp) OVER (
                    ORDER BY ordering ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS timestamp, ordering, 0 AS kind FROM real_rows WHERE rank = 1
                UNION ALL
                SELECT ordinal, created_at, ordinal, 1 FROM session_display_events
                WHERE session_id IN ({_placeholders(event_sessions)})
            ), page AS (
                SELECT * FROM timeline ORDER BY timestamp {direction}, kind {direction}, ordering {direction}
                LIMIT ? OFFSET ?
            )
            SELECT m.*, e.event_id AS _event_id, e.command AS _command,
                   e.output AS _output, e.created_at AS _created_at
            FROM page
            LEFT JOIN messages m ON page.kind = 0 AND m.id = page.row_id
            LEFT JOIN session_display_events e ON page.kind = 1 AND e.ordinal = page.row_id
            ORDER BY page.timestamp, page.kind, page.ordering"""
        rows = self._read_all(query, (session_id, *event_sessions, -1 if limit is None else limit, offset))
        result = []
        for row in rows:
            message = dict(row)
            event_id = message.pop("_event_id")
            command = message.pop("_command")
            output = message.pop("_output")
            created_at = message.pop("_created_at")
            result.append(_display_event_message(dict(
                event_id=event_id, command=command, output=output, created_at=created_at))
                if event_id is not None else self._row_to_message_dict(
                    message, warn_context="get_display_messages", summary_flag=True))
        return result
