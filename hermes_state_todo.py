"""The newest Todo result a resume restores: an indexed, bounded search of the resume display scope.
Split out of ``hermes_state_messages`` and bound onto SessionDB through ``SessionMessagesMixin``; built
on its _resume_lineage_ids/_read_ctx primitives."""

from __future__ import annotations

import json
from typing import Optional


class SessionTodoResultMixin:
    """Latest visible, paired and valid Todo tool result of a session's lineage."""

    def get_latest_todo_result(self, session_id: str, *, max_chars: Optional[int] = None) -> Optional[str]:
        """Newest visible, paired and valid Todo result in the resume display scope.

        Search indexed, bounded pages rather than loading a potentially long transcript. A branch owns its
        copied history; compression continuations also show their ancestors' archived rows. Invalid results
        are skipped just as the model's Todo hydration does, and explicit empty results are retained.
        """
        if not session_id:
            return None
        if max_chars is None:
            from tools.todo_tool import MAX_TODO_RESULT_CHARS
            max_chars = MAX_TODO_RESULT_CHARS
        session_ids = self._resume_lineage_ids(session_id)
        with self._read_ctx() as conn:
            for sid in reversed(session_ids):
                before_id = 2**63 - 1
                while True:
                    rows = conn.execute(
                        "SELECT id, content, tool_call_id FROM messages INDEXED BY idx_messages_session_id "
                        "WHERE session_id = ? AND id < ? AND role = 'tool' "
                        "AND tool_name IN ('todo_list', 'todo') AND (active = 1 OR compacted = 1) "
                        "AND length(content) <= ? ORDER BY id DESC LIMIT 100",
                        (sid, before_id, max_chars),
                    ).fetchall()
                    if not rows:
                        break
                    for row in rows:
                        call_id = row["tool_call_id"]
                        if not call_id:
                            continue
                        prior = conn.execute(
                            "SELECT role, tool_calls FROM messages INDEXED BY idx_messages_session_id "
                            "WHERE session_id = ? AND id < ? AND role IN ('assistant', 'user', 'system') "
                            "AND (active = 1 OR compacted = 1) ORDER BY id DESC LIMIT 1",
                            (sid, row["id"]),
                        ).fetchone()
                        if prior is None or prior["role"] != "assistant":
                            continue
                        try:
                            calls = json.loads(prior["tool_calls"]) if prior["tool_calls"] else []
                            def is_todo_call(call):
                                if not isinstance(call, dict) or call.get("id") != call_id:
                                    return False
                                function = call.get("function")
                                if not isinstance(function, dict):
                                    return False
                                if function.get("name") in ("todo_list", "todo"):
                                    return True
                                if function.get("name") != "tool_call":
                                    return False
                                arguments = function.get("arguments")
                                if isinstance(arguments, str):
                                    arguments = json.loads(arguments)
                                if not isinstance(arguments, dict):
                                    return False
                                inner_calls = arguments.get("calls")
                                return isinstance(inner_calls, list) and any(
                                    isinstance(inner, dict) and inner.get("name") in ("todo_list", "todo")
                                    for inner in inner_calls)

                            paired = isinstance(calls, list) and any(is_todo_call(call) for call in calls)
                            if not paired:
                                continue
                            data = json.loads(row["content"])
                            if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
                                continue
                            int(data.get("revision") or 0)
                        except (ValueError, TypeError):
                            continue
                        return row["content"]
                    before_id = rows[-1]["id"]
        return None
