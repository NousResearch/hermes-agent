"""Per-turn native task state shared by Slack cards and iLink lifecycle messages."""

import dataclasses
import re
from typing import Any, Optional

from agent.i18n import t


@dataclasses.dataclass
class TaskCardState:
    """Task-card rail state for ``_send_native_task_card_progress``."""
    adapter: Any
    tasks: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    task_order: list[str] = dataclasses.field(default_factory=list)
    fallback_msg_id: Optional[str] = None
    native_failed: bool = False
    # TERMINAL for the turn, distinct from native_failed: no later publication
    # in this turn may deliver task text through the native lane OR the text
    # fallback. Two causes, both properties of the destination rather than of
    # one attempt: the connector's egress guard refused the chat, or the chat
    # cannot host a card (no thread anchor). Declared rather than set
    # dynamically so the state is visible where it lives.
    publication_suppressed: bool = False
    anonymous_seq: int = 0

    @staticmethod
    def _compact(value: Any, limit: int = 120) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    def visible_tasks(self) -> list[dict[str, str]]:
        limit = getattr(type(self.adapter), "native_task_card_task_limit", 8)
        order = self.task_order[-limit:] if limit else self.task_order
        return [self.tasks[task_id] for task_id in order]

    def fallback_text(self) -> str:
        labels = {"in_progress": t("gateway.progress.task_status_running"),
                  "complete": t("gateway.progress.task_status_complete"),
                  "error": t("gateway.progress.task_status_error")}
        lines = [t("gateway.progress.task_line", title=task["title"], status=labels.get(task["status"], task["status"]))
                 for task in self.visible_tasks()]
        return t("gateway.progress.task_card_title") + "\n" + "\n".join(lines)

    def _upsert(self, call_id: str, title: str) -> dict[str, str]:
        if call_id not in self.tasks:
            self.task_order.append(call_id)
        self.tasks[call_id] = {"id": call_id, "title": self._compact(title), "status": "in_progress"}
        return self.tasks[call_id]

    def apply_event(self, raw: Any) -> bool:
        event_type = raw.get("type") if isinstance(raw, dict) else None
        if event_type not in {"tool.started", "tool.completed"}:
            return False
        call_id = str(raw.get("tool_call_id") or "")
        if not call_id:
            self.anonymous_seq += 1
            call_id = f"anonymous_{self.anonymous_seq}"
        tool_name = str(raw.get("tool_name") or "tool")
        if event_type == "tool.started":
            preview = self._compact(raw.get("preview"), 64)
            self._upsert(call_id, f"{tool_name} - {preview}" if preview else tool_name)
            return True
        # Completion-only events are rare but valid on some runtimes; keep their real ID instead
        # of guessing a same-name pending call.
        task = self.tasks.get(call_id) or self._upsert(call_id, tool_name)
        task["status"] = "error" if raw.get("is_error") else "complete"
        return True
