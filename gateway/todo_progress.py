"""Turn-scoped editable Telegram todo status; ordinary progress has its own lifetime."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from tools.todo_tool import is_todo_tool_name

logger = logging.getLogger("gateway.run")
_STATUS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "cancelled": "[-]"}


@dataclass(frozen=True)
class TodoChecklist:
    text: str


def _render_todo_checklist(result) -> str:
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return ""
    lines = []
    for item in data["todos"]:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            return ""
        content = " ".join(item["content"].split())[:80]
        status = _STATUS.get(str(item.get("status", "pending")).lower(), "[ ]")
        item_id = " ".join(str(item.get("id") or "").split())[:32]
        lines.append(f"{status} {content}" + (f" ({item_id})" if item_id else ""))
    return "\n".join(["📋 Task List", *(lines or ["No tasks."])])


def _dispatch_todo_progress(progress_queue, event_type, tool_name, result, task_logger) -> bool:
    if event_type != "tool.completed" or not is_todo_tool_name(tool_name):
        return False
    text = _render_todo_checklist(result)
    if not text or progress_queue is None:
        task_logger.debug("Skipping todo checklist without a valid snapshot or progress queue")
        return False
    progress_queue.put(TodoChecklist(text))
    return True


class TodoProgress:
    """One message binding per sender, unaffected by generic bubble resets or grouping."""

    def __init__(self, progress, adapter):
        self.progress = progress
        self.state = progress._progress_edit_state(adapter)
        self.latest = self.delivered = ""
        self.next_edit = self.retry_at = 0.0
        self.disabled = False

    def absorb(self, snapshot: TodoChecklist) -> None:
        # Keep whole rows and reserve space for an omission notice in platform length units.
        st = self.state
        lines = snapshot.text.splitlines()
        kept = []
        for index, line in enumerate(lines):
            footer = f"… ({len(lines) - index} more tasks)"
            candidate = "\n".join([*kept, line, *([footer] if index < len(lines) - 1 else [])])
            if st._progress_len_fn(candidate) > st._PROGRESS_TEXT_LIMIT:
                kept.append(footer)
                break
            kept.append(line)
        self.latest = "\n".join(kept)

    async def flush(self, *, force=False) -> None:
        if self.disabled or not self.latest or self.latest == self.delivered:
            return
        now = time.monotonic()
        if now < self.retry_at or (not force and now < self.next_edit):
            return
        ctx = self.progress._ctx
        if not ctx._run_still_current() or self.progress._agent_interrupted():
            return
        # Bound attempts even when Telegram rejects a send or edit. Retryable failures retain
        # the exact message id; a permanent refusal disables status updates, never the turn.
        self.next_edit = time.monotonic() + 1.5
        try:
            if self.state.progress_msg_id is None:
                result = await self.progress._send_progress_text(self.state, self.latest)
            else:
                result = await self.progress._edit_progress_message(
                    self.state, self.state.progress_msg_id, self.latest,
                )
        except Exception:
            self.retry_at = self.next_edit
            logger.debug("Todo status delivery failed", exc_info=True)
            return
        if result.success:
            self.state.progress_msg_id = result.message_id or self.state.progress_msg_id
            self.disabled = not bool(self.state.progress_msg_id)
            self.delivered = self.latest
            self.retry_at = 0.0
        elif not result.success:
            retry_after = result.retry_after or 0
            self.retry_at = time.monotonic() + max(1.5, retry_after)
            self.disabled = not (result.retryable or retry_after)
            logger.debug("Todo status delivery refused; retryable=%s", not self.disabled)
