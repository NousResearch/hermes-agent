"""Channel-aware gateway toolset routing.

Routes must be selected before agent construction so an AIAgent's system prompt
and tool schemas stay stable for the life of its cached session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gateway.config import Platform

SASHA_HOME_CHANNEL_IDS = {"1513198617198858434"}

# Keep these as concrete built-in toolset names. ``platform_toolsets`` currently
# does not expand custom_toolsets aliases in the prompt-size/tool resolution
# path, so returning aliases here would silently produce an agent with no tools.
SASHA_DISCORD_LEAN_TOOLSETS = [
    "clarify",
    "homeassistant",
    "memory",
    "session_search",
    "skills",
    "todo",
    "vision",
    "web",
]
SASHA_DISCORD_CODING_TOOLSETS = [
    "clarify",
    "code_execution",
    "delegation",
    "file",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
    "web",
]
SASHA_DISCORD_HEAVY_TOOLSETS = [
    "browser",
    "clarify",
    "code_execution",
    "computer_use",
    "cronjob",
    "delegation",
    "file",
    "homeassistant",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
    "vision",
    "web",
]

_PROJECT_CHANNEL_RE = re.compile(r"(?:^|[/#\s])#?proj-[\w-]+", re.IGNORECASE)
_HEAVY_INTENT_PATTERNS = (
    re.compile(r"\b(use|open|drive)\s+(?:the\s+)?browser\b", re.IGNORECASE),
    re.compile(
        r"\b(logged[- ]in browser|computer[- ]use|drive my desktop|click|screenshot)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(schedule|cron|remind me)\b", re.IGNORECASE),
    re.compile(r"\b(generate|make|create)\s+(?:an?\s+)?(?:image|audio|video)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ToolsetRoute:
    toolsets: list[str]
    reason: str


def _platform_value(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def _source_chat_ids(source: Any) -> set[str]:
    ids: set[str] = set()
    for attr in ("chat_id", "parent_chat_id"):
        value = getattr(source, attr, None)
        if value is not None:
            text = str(value).strip()
            if text:
                ids.add(text)
    return ids


def _source_channel_name(source: Any) -> str:
    values = []
    for attr in ("chat_name", "channel_name", "display_name", "chat_topic"):
        value = getattr(source, attr, None)
        if value:
            values.append(str(value))
    return " / ".join(values)


def _has_heavy_intent(message: str | None) -> bool:
    text = str(message or "")
    return any(pattern.search(text) for pattern in _HEAVY_INTENT_PATTERNS)


def route_toolsets_for_source(source: Any, *, message: str | None = None) -> list[str] | None:
    """Return a Sasha Discord toolset override for ``source``, or None.

    Home is forced lean even if the message asks for code/heavy capability.
    That keeps Ben's Home channel non-coding and avoids changing an existing
    Home session's prompt/tool schema. Heavy tools route only on explicit
    requests outside Home; callers should apply this before creating an agent.
    """
    route = route_toolsets_for_source_with_reason(source, message=message)
    return None if route is None else route.toolsets


def route_toolsets_for_source_with_reason(
    source: Any, *, message: str | None = None
) -> ToolsetRoute | None:
    """Return routed toolsets plus a compact log-safe reason."""
    if _platform_value(getattr(source, "platform", None)) != Platform.DISCORD.value:
        return None
    if _source_chat_ids(source) & SASHA_HOME_CHANNEL_IDS:
        return ToolsetRoute(list(SASHA_DISCORD_LEAN_TOOLSETS), "home_channel")
    if _has_heavy_intent(message):
        return ToolsetRoute(list(SASHA_DISCORD_HEAVY_TOOLSETS), "heavy_intent")
    if _PROJECT_CHANNEL_RE.search(_source_channel_name(source)):
        return ToolsetRoute(list(SASHA_DISCORD_CODING_TOOLSETS), "proj_channel")
    return ToolsetRoute(list(SASHA_DISCORD_LEAN_TOOLSETS), "default")
