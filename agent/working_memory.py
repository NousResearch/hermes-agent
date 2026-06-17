"""Session-local working memory for active task state.

Working memory is deliberately *not* part of agent/system_prompt.py.
That module builds the cached/stored system prompt once per session. This
module renders a small request-time-only overlay that can change as the
active task changes without rewriting the cached prompt snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional


MAX_GOAL_CHARS = 240
MAX_ITEM_CHARS = 180
MAX_ACTIVE_TODOS = 8
_TRUNCATION_MARKER = "…"


def _compact_line(value: Any, limit: int) -> str:
    """Collapse whitespace and cap a single working-memory line."""
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER


@dataclass
class WorkingMemory:
    """Tiny, explicit, session-local state for the task currently in focus.

    This is not durable semantic memory and not episodic history. It should
    hold only the active goal and immediate open loops that help the next
    model call stay oriented.
    """

    current_goal: str = ""
    constraints: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

    def observe_user_turn(self, user_message: Any) -> None:
        """Refresh the active goal from the latest user turn.

        PR1 keeps this intentionally conservative: it records the latest user
        ask as the active goal and leaves richer summarisation/consolidation to
        a later PR. The raw transcript remains the episodic source of truth.
        """
        goal = _compact_line(user_message, MAX_GOAL_CHARS)
        if goal:
            self.current_goal = goal

    def clear(self) -> None:
        self.current_goal = ""
        self.constraints.clear()
        self.blockers.clear()
        self.hypotheses.clear()
        self.next_actions.clear()

    def format_for_ephemeral_system_prompt(self, agent: Any = None) -> str:
        """Render a compact system-prompt overlay for this request only."""
        lines: List[str] = []
        if self.current_goal:
            lines.append(f"Current goal: {self.current_goal}")

        _append_bucket(lines, "Constraints", self.constraints)
        _append_bucket(lines, "Blockers", self.blockers)
        _append_bucket(lines, "Hypotheses", self.hypotheses)
        _append_bucket(lines, "Next actions", self.next_actions)

        todo_lines = _format_active_todos(getattr(agent, "_todo_store", None)) if agent is not None else []
        if todo_lines:
            lines.append("Active todos:")
            lines.extend(todo_lines)

        if not lines:
            return ""

        return "\n".join([
            "WORKING MEMORY (ephemeral, request-time only)",
            "This is the active task state for the current session. Use it to stay oriented, but do not treat it as durable memory or save it verbatim.",
            *lines,
        ])


def _append_bucket(lines: List[str], label: str, values: Iterable[Any]) -> None:
    items = [_compact_line(v, MAX_ITEM_CHARS) for v in values if _compact_line(v, MAX_ITEM_CHARS)]
    if not items:
        return
    lines.append(f"{label}:")
    lines.extend(f"- {item}" for item in items[:MAX_ACTIVE_TODOS])


def _format_active_todos(todo_store: Any) -> List[str]:
    if todo_store is None or not hasattr(todo_store, "read"):
        return []
    try:
        todos = todo_store.read()
    except Exception:
        return []

    active = []
    for item in todos:
        status = str(item.get("status", "")).strip().lower()
        if status not in {"pending", "in_progress"}:
            continue
        content = _compact_line(item.get("content", ""), MAX_ITEM_CHARS)
        if not content:
            continue
        item_id = _compact_line(item.get("id", "?"), 40) or "?"
        active.append(f"- [{status}] {item_id}: {content}")
        if len(active) >= MAX_ACTIVE_TODOS:
            break
    return active


def build_working_memory_ephemeral_prompt(agent: Any) -> str:
    """Return the agent's request-time working-memory overlay, if any."""
    working_memory = getattr(agent, "_working_memory", None)
    if working_memory is None or not hasattr(working_memory, "format_for_ephemeral_system_prompt"):
        return ""
    try:
        return working_memory.format_for_ephemeral_system_prompt(agent)
    except Exception:
        return ""


__all__ = ["WorkingMemory", "build_working_memory_ephemeral_prompt"]
