"""``ctx.kanban_events`` — the plugin-facing Kanban event log.

Plugins that keep workflow state on a board (verdicts, gate decisions,
failure records) previously imported ``kanban_db._append_event`` and read
``task_events`` with raw SQL. Both are internals. This facade is the stable
surface: append goes through the same transaction helper as core writes,
reads use an id cursor, and plugin kinds are namespaced ``<plugin_id>:<kind>``
so a plugin can never emit (or be confused with) a core lifecycle event.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

_KIND = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
MAX_PAYLOAD_BYTES = 64 * 1024
MAX_READ_LIMIT = 1000


class KanbanEvents:
    """Append/read Kanban task events on behalf of one plugin."""

    def __init__(self, plugin_id: str):
        self.plugin_id = plugin_id

    def kind(self, kind: str) -> str:
        """The stored kind for this plugin's ``kind`` (``<plugin_id>:<kind>``)."""
        if not isinstance(kind, str) or not _KIND.fullmatch(kind):
            raise ValueError("event kind must match [a-z0-9][a-z0-9_.-]{0,63}")
        return f"{self.plugin_id}:{kind}"

    def append(
        self, task_id: str, kind: str, payload: Optional[dict] = None, *,
        board: Optional[str] = None, run_id: Optional[int] = None,
    ) -> int:
        """Append one event to ``task_id`` and return its id.

        Raises ``KeyError`` for an unknown task, ``ValueError`` for a bad kind
        or a payload that is not a JSON object under 64 KiB."""
        from hermes_cli import kanban_db as kb
        from hermes_cli.kanban_db_connect import connect_closing, write_txn

        stored = self.kind(kind)
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError("event payload must be a dict or None")
            if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_PAYLOAD_BYTES:
                raise ValueError("event payload exceeds 64 KiB")
        with connect_closing(board=board) as conn:
            with write_txn(conn):
                if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
                    raise KeyError(task_id)
                kb._append_event(conn, task_id, stored, payload, run_id=run_id)
                return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def read(
        self, *, task_id: Optional[str] = None, since_id: int = 0,
        kinds: Optional[Iterable[str]] = None, board: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Events with ``id > since_id`` in id order (a resumable cursor).

        ``kinds`` filters on stored kind names: core kinds (``completed``,
        ``blocked`` …) as-is, plugin kinds in their ``<plugin_id>:<kind>`` form
        (see :meth:`kind`). Payloads are decoded JSON (``None`` if absent)."""
        from hermes_cli import kanban_db as kb
        from hermes_cli.kanban_db_connect import connect_closing

        limit = max(1, min(int(limit), MAX_READ_LIMIT))
        sql = "SELECT * FROM task_events WHERE id > ?"
        params: list[Any] = [int(since_id)]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        if kinds is not None:
            kinds = list(kinds)
            if not kinds:
                return []
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        with connect_closing(board=board) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"id": e.id, "task_id": e.task_id, "run_id": e.run_id, "kind": e.kind,
             "payload": e.payload, "created_at": e.created_at}
            for e in (kb.Event.from_row(r) for r in rows)
        ]
