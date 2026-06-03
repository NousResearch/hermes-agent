"""Telegram group no-action suppression.

This plugin is intentionally narrow:
- It only applies to Telegram non-DM contexts.
- It skips obvious casual/no-action group chatter before agent dispatch.
- It preserves explicit Jimmy/action/dev/approval/blocker/correction signals.
- It does not inspect secrets, logs, env, auth, or runtime configuration.
"""

from __future__ import annotations

import re
from typing import Any, Optional


_REASON = "telegram-group-casual-no-action"

_EXPLICIT_HEADERS = (
    "JIMMY:",
    "REQUEST:",
    "ACTION:",
    "REVIEW:",
    "APPROVAL:",
    "BLOCKER:",
    "ROUTE:",
    "CODEX READ-ONLY:",
    "TIMMY:",
    "BEBE:",
    "DECISION:",
    "GATE:",
    "ESCALATION:",
    "CLOSEOUT:",
    "QUESTION FOR JIMMY:",
)

_PRESERVE_KEYWORDS = (
    "approval",
    "approve",
    "auto-approve",
    "read-only",
    "readonly",
    "codex",
    "timmy",
    "bebe",
    "blocker",
    "blocked",
    "review",
    "correction",
    "correct",
    "gate",
    "escalat",
    "closeout",
    "decision",
    "route",
    "implement",
    "implementation",
    "evidence",
    "qa",
    "test",
)


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").strip().lower()


def _is_telegram_group_context(source: Any) -> bool:
    if source is None:
        return False

    if _platform_name(getattr(source, "platform", None)) != "telegram":
        return False

    chat_type = str(getattr(source, "chat_type", "") or "").strip().lower()

    if chat_type in {"dm", "private"}:
        return False

    return chat_type in {"group", "supergroup", "channel", "thread"}


def _has_preserve_signal(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True

    upper = stripped.upper()
    lower = stripped.lower()

    if any(header in upper for header in _EXPLICIT_HEADERS):
        return True

    # Preserve any direct natural-language address to Jimmy.
    if re.search(r"\bjimmy\b", lower):
        return True

    # Preserve questions rather than risking false-positive suppression.
    if "?" in stripped:
        return True

    if any(keyword in lower for keyword in _PRESERVE_KEYWORDS):
        return True

    return False


def _on_pre_gateway_dispatch(event: Any, **_: Any) -> Optional[dict[str, str]]:
    source = getattr(event, "source", None)

    if not _is_telegram_group_context(source):
        return None

    text = str(getattr(event, "text", "") or "").strip()

    if _has_preserve_signal(text):
        return None

    return {"action": "skip", "reason": _REASON}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
