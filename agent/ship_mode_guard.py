"""Deterministic routing guard for serious app/product ship requests.

This is intentionally cheap and conservative. It does not try to solve the
request; it only decides whether a turn needs the app-ship/Kanban operating
loop made explicit at API-call time so it is not dependent on the model
remembering an optional skill from prior context.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping, Sequence


_SHIP_MODE_GUARD_MARKER = "Ship-mode routing guard"

_TINY_FIX_RE = re.compile(
    r"\b(fix|change|update|rename|tweak|adjust)\b.{0,80}\b(typo|copy|text|label|wording|readme|comment|color|colour|padding|margin)\b",
    re.IGNORECASE | re.DOTALL,
)

_PLANNING_QUESTION_RE = re.compile(r"^\s*(what|why|how|should|could|would|is|are)\b", re.IGNORECASE)

_ACTION_TERMS = (
    "build", "ship", "implement", "finish", "create", "make", "launch",
    "wire", "add", "restore", "repair", "redesign", "migrate", "port",
)

_SERIOUS_SCOPE_TERMS = (
    "app", "feature", "product", "prd", "full-stack", "full stack",
    "frontend", "backend", "api", "database", "db", "ui", "ux",
    "runtime", "preview", "deploy", "deployment", "release", "workflow",
    "onboarding", "dashboard", "agent", "kanban", "obsidian", "source-of-truth",
)

_HEAVY_TERMS = (
    "end-to-end", "e2e", "ship", "shipped", "all the way", "don't stop",
    "dont stop", "no stopping", "finish it", "full", "complete", "production",
    "verify", "verified", "smoke-test", "smoke test", "review", "external review",
    "multi-agent", "parallel", "background", "kanban", "preview",
)

_EXPLICIT_TRIGGERS = (
    "app ship mode", "ship mode", "ship this end-to-end", "ship this end to end",
    "build this app", "build me an app", "build an app", "new app mode",
    "finish this end-to-end", "finish this end to end", "full app-dev",
    "full app dev", "do not stop", "don't stop", "dont stop",
)


def _lower_text(value: Any) -> str:
    return str(value or "").lower()


def _history_text(conversation_history: Sequence[Mapping[str, Any]] | None, *, limit: int = 8) -> str:
    if not conversation_history:
        return ""
    parts: list[str] = []
    for msg in list(conversation_history)[-limit:]:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content[:500])
    return "\n".join(parts)


def is_serious_app_ship_request(
    user_message: str,
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Return True for app/product build requests that need ship-mode routing.

    The detector is conservative by design:
    - explicit ship/app-mode phrases always match;
    - otherwise it requires an action, an app/product/dev scope marker, and a
      heavy/durable marker, with recent history allowed to provide scope;
    - tiny one-file/copy fixes and question-only planning prompts are excluded.
    """
    text = _lower_text(user_message)
    if not text or _SHIP_MODE_GUARD_MARKER.lower() in text:
        return False
    if _TINY_FIX_RE.search(text):
        return False
    if _PLANNING_QUESTION_RE.search(text) and not any(trigger in text for trigger in _EXPLICIT_TRIGGERS):
        return False

    recent = _lower_text(_history_text(conversation_history))
    combined = f"{text}\n{recent}"

    if any(trigger in text for trigger in _EXPLICIT_TRIGGERS):
        return True

    has_action = any(re.search(rf"\b{re.escape(term)}\b", text) for term in _ACTION_TERMS)
    if not has_action:
        return False

    has_scope = any(term in combined for term in _SERIOUS_SCOPE_TERMS)
    has_heavy = any(term in combined for term in _HEAVY_TERMS)

    return bool(has_scope and has_heavy)


def build_ship_mode_routing_context(
    user_message: str,
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
    *,
    platform: str | None = None,
    valid_tool_names: Iterable[str] | None = None,
) -> str:
    """Build the API-only routing nudge for serious ship-mode turns.

    Returns an empty string when the turn should proceed normally. The context
    is intended to be appended to the current user message at API-call time, not
    persisted to transcripts and not added to the cached system prompt.
    """
    platform_key = (platform or "").lower().strip()
    if platform_key == "cron" or os.getenv("HERMES_KANBAN_TASK"):
        return ""
    if not is_serious_app_ship_request(user_message, conversation_history):
        return ""

    tools = set(valid_tool_names or [])
    todo_phrase = "Use the todo tool" if "todo" in tools else "Write a concrete todo list"
    kanban_phrase = (
        "route durable/parallel lanes into Kanban tasks when appropriate"
        if "kanban_create" in tools or "kanban_show" in tools
        else "route durable/parallel lanes into Kanban or explicit worker tasks when appropriate"
    )

    return (
        f"[{_SHIP_MODE_GUARD_MARKER}: this request matches serious app/build/product execution. "
        "Treat app-ship-mode/Kanban operating rules as mandatory for this turn, even if the skill was not previously loaded. "
        f"{todo_phrase} before implementation; resolve source-of-truth/PRD/Obsidian/repo context before coding; "
        f"{kanban_phrase}; preserve the requested scope instead of shrinking it; verify real runtime behavior and run review before claiming shipped. "
        "Tiny/single-file fixes are intentionally excluded from this guard.]"
    )


__all__ = [
    "build_ship_mode_routing_context",
    "is_serious_app_ship_request",
]
