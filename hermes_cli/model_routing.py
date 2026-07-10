"""Config-driven model route selection for agent turns.

This module is deliberately pure: it reads config shapes and returns the
requested route, while callers keep using their normal provider resolver for
credentials and client setup.
"""

from __future__ import annotations

import re
from typing import Any


_TOOL_USE_HINT_RE = re.compile(
    r"(?i)\b("
    r"run|execute|test|pytest|npm|ruff|mypy|lint|build|commit|diff|patch|"
    r"edit|write|modify|create|delete|rename|move|read|open|inspect|search|"
    r"grep|rg|find|file|directory|repo|code|bug|debug|fix|implement|shell|"
    r"terminal|command|browse|web|fetch|download"
    r")\b|(?:^|\s)(?:\./|/|~\/|[A-Za-z]:\\)"
)


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _configured_context(config: dict[str, Any]) -> str:
    routing = _as_mapping(config.get("routing"))
    value = str(routing.get("context") or routing.get("default_context") or "").strip()
    return value


def classify_task_context(config: dict[str, Any] | None, user_message: Any) -> str:
    """Return the routing context for a turn.

    ``routing.context`` can pin a context explicitly. Otherwise, only configs
    that define both ``routing.chat`` and ``routing.tool_use`` get automatic
    chat/tool-use classification; this keeps existing installs inert.
    """

    config = config or {}
    explicit = _configured_context(config)
    if explicit:
        return explicit

    routing = _as_mapping(config.get("routing"))
    if "chat" not in routing or "tool_use" not in routing:
        return ""

    text = ""
    if isinstance(user_message, str):
        text = user_message
    elif isinstance(user_message, list):
        parts: list[str] = []
        for part in user_message:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        text = "\n".join(parts)

    return "tool_use" if _TOOL_USE_HINT_RE.search(text or "") else "chat"


def get_route_override(
    config: dict[str, Any] | None,
    task_context: str,
) -> dict[str, Any]:
    """Return ``routing.<task_context>`` when it names a provider/model route."""

    routing = _as_mapping((config or {}).get("routing"))
    entry = _as_mapping(routing.get(task_context))
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or entry.get("default") or "").strip()
    if not provider and not model:
        return {}

    route = dict(entry)
    if provider:
        route["provider"] = provider
    if model:
        route["model"] = model
    return route


def fallback_chain_signature(chain: Any) -> tuple[tuple[str, str, str], ...]:
    """Stable identity for deciding whether a cached agent's fallback changed."""

    if not isinstance(chain, list):
        return ()
    out: list[tuple[str, str, str]] = []
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        out.append((
            str(entry.get("provider") or "").strip().lower(),
            str(entry.get("model") or "").strip(),
            str(entry.get("base_url") or "").strip().rstrip("/").lower(),
        ))
    return tuple(out)
