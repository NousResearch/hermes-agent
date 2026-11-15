"""Stored ``delegate_task`` children of a conversation, for the classic CLI's subagent viewer.

Every child runs as its own stored session (``source='subagent'``, ``parent_session_id`` = the
spawning session, ``model_config._delegate_from`` set). A conversation can span several stored
sessions when compression rotates it, so the children of every segment of the compression chain
are collected, then their own children (nested orchestrators), oldest first.

Read-only: the gateway-backed CLI store serves ``list_subagent_sessions`` from its read-only
reader (``hermes_cli.gateway_session_store._READS``).
"""
from __future__ import annotations

import json
from typing import Any

# Bounds keep a pathological store (cycles, runaway fan-out) from stalling the CLI's tick.
_MAX_DEPTH = 5
_MAX_ROWS = 500

_CHILDREN_SQL = """
    SELECT s.id, s.parent_session_id, s.title, s.model, s.started_at, s.ended_at, s.end_reason,
           s.message_count, s.tool_call_count,
           (SELECT m.content FROM messages m
             WHERE m.session_id = s.id AND m.role = 'user' ORDER BY m.id LIMIT 1) AS goal
      FROM sessions s
     WHERE s.parent_session_id IN ({placeholders})
       AND (s.source = 'subagent'
            OR (json_valid(s.model_config)
                AND json_type(s.model_config, '$._delegate_from') = 'text'))
     ORDER BY s.started_at, s.id
"""


def _goal_text(content: Any) -> str:
    """The child's brief: its first user message, flattened when stored as content parts."""
    if not isinstance(content, str):
        return ""
    stripped = content.lstrip()
    if stripped.startswith("["):
        try:
            parts = json.loads(stripped)
        except ValueError:
            return content
        if isinstance(parts, list):
            return "\n".join(str(p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
    return content


class SessionSubagentsMixin:
    def conversation_segments(self, session_id: str) -> list[str]:
        """Stored ids of the conversation ``session_id`` belongs to, root first: the compression
        chain only, never the conversation a branch was forked from."""
        if not session_id or self.get_session(session_id) is None:
            return []
        root = session_id
        seen = {root}
        while True:
            parent = (self.get_session(root) or {}).get("parent_session_id")
            if not parent or parent in seen:
                break
            parent_row = self.get_session(parent) or {}
            if parent_row.get("end_reason") != "compression" or root not in self.get_compression_chain(parent):
                break
            seen.add(parent)
            root = parent
        chain = self.get_compression_chain(root) or [root]
        return chain if session_id in chain else [*chain, session_id]

    def list_subagent_sessions(self, session_id: str) -> list[dict[str, Any]]:
        """Every stored subagent of the conversation, direct children first by start time, then
        nested children; each row carries ``depth`` (1 = spawned by the conversation itself)."""
        frontier = self.conversation_segments(session_id)
        seen = set(frontier)
        rows: list[dict[str, Any]] = []
        depth = 0
        while frontier and depth < _MAX_DEPTH and len(rows) < _MAX_ROWS:
            depth += 1
            with self._read_ctx() as conn:
                found = conn.execute(_CHILDREN_SQL.format(placeholders=",".join("?" * len(frontier))),
                                     frontier).fetchall()
            frontier = []
            for record in found:
                child_id = record["id"]
                if child_id in seen:
                    continue
                seen.add(child_id)
                frontier.append(child_id)
                rows.append({
                    "session_id": child_id, "parent_session_id": record["parent_session_id"], "depth": depth,
                    "goal": _goal_text(record["goal"]), "title": record["title"], "model": record["model"],
                    "started_at": record["started_at"], "ended_at": record["ended_at"],
                    "end_reason": record["end_reason"], "message_count": record["message_count"],
                    "tool_call_count": record["tool_call_count"],
                })
        return rows[:_MAX_ROWS]
