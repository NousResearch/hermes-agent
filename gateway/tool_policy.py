"""Gateway toolset selection policy."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable


LIGHTWEIGHT_DISCORD_TOOLSETS = frozenset(
    {
        "skills",
        "memory",
        "session_search",
        "clarify",
        "chat",
        "channel-notes",
        "budget",
    }
)

WORK_INTENT_RE = re.compile(
    r"\b("
    r"code|coding|debug|bug|fix|implement|repo|repository|branch|commit|push|pr|"
    r"test|pytest|ruff|mypy|terminal|shell|bash|python|file|patch|diff|docker|"
    r"build|compile|deploy|server|logs?|browser|screenshot|delegate|subagent"
    r")\b",
    re.IGNORECASE,
)


def select_gateway_toolsets(
    *,
    platform: str,
    configured_toolsets: Iterable[str],
    user_text: str | None = None,
    auto_skill: object = None,
) -> list[str]:
    """Return toolsets for a gateway turn.

    Discord gets a lightweight default for casual chat. Explicit work intent or
    auto-loaded skills keep the configured toolsets so the agent can actually
    perform coding/debugging tasks when asked.
    """

    configured = sorted({str(toolset) for toolset in configured_toolsets})
    if platform != "discord":
        return configured
    if not should_use_lightweight_discord_context(
        platform=platform,
        user_text=user_text,
        auto_skill=auto_skill,
    ):
        return configured
    preserved_mcp = [name for name in configured if name.startswith("mcp-")]
    selected = sorted(
        (set(configured) & set(LIGHTWEIGHT_DISCORD_TOOLSETS)) | set(preserved_mcp)
    )
    return selected or configured


def should_use_lightweight_discord_context(
    *,
    platform: str,
    user_text: str | None = None,
    auto_skill: object = None,
) -> bool:
    """Return whether this Discord turn should use the lightweight base context."""

    if platform != "discord":
        return False
    if not _lightweight_enabled():
        return False
    if auto_skill:
        return False
    return WORK_INTENT_RE.search(user_text or "") is None


def _lightweight_enabled() -> bool:
    raw = os.getenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}
