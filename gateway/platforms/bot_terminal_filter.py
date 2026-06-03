"""Deterministic terminal bot-message suppression for platform adapters.

When a platform's allow-bots policy permits bot-authored messages to reach
agent dispatch, bot-to-bot loops can form: agent A sends an ACK/final, agent B's
gateway dispatches it, B's agent replies with its own ACK/final, and so on.
Prompt doctrine alone is not a sufficient loop brake.

This module classifies bot-authored message text as terminal/no-work chatter
(suppress) versus actionable handoff/request (preserve), using fixed regexes
only — no LLM classification. Callers remain responsible for gating on
``is_bot`` and the platform's enable toggle.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "should_suppress_terminal_bot_message",
    "message_requests_no_visible_ack",
    "terminal_bot_suppression_enabled",
    "SUPPRESS_TERMINAL_BOT_MESSAGES_ENV",
]

SUPPRESS_TERMINAL_BOT_MESSAGES_ENV = "DISCORD_SUPPRESS_TERMINAL_BOT_MESSAGES"

_FALSY = {"false", "0", "no", "off"}

_ACTIONABLE_PATTERNS = [
    re.compile(r"\bkind\s*[:=]\s*request\b", re.IGNORECASE),
    re.compile(r"\brequires[\s_-]?ack\b", re.IGNORECASE),
    re.compile(r"\bhandoff\b", re.IGNORECASE),
    re.compile(r"\boperator[\s_-]?gated\b", re.IGNORECASE),
]

_NO_VISIBLE_ACK_PATTERN = re.compile(
    r"\back[\s_-]?policy\s*[:=]\s*none\b", re.IGNORECASE
)
_KIND_PATTERN = re.compile(r"\bkind\s*[:=]\s*([a-z0-9_-]+)\b", re.IGNORECASE)
_TRANSPORT_RECEIPT_KINDS = {"transport_receipt", "transport-receipt", "receipt"}

_TERMINAL_PATTERNS = [
    re.compile(r"\bkind\s*[:=]\s*ack\b", re.IGNORECASE),
    re.compile(r"\bkind\s*[:=]\s*final\b", re.IGNORECASE),
    re.compile(r"\bkind\s*[:=]\s*status\b", re.IGNORECASE),
    re.compile(r"\b(?:done|complete|completed)\b\.?", re.IGNORECASE),
    re.compile(r"\back(?:s|ed|nowledged)?\b", re.IGNORECASE),
    re.compile(r"\backs?\b\s+do\s+not\s+require\s+\backs?\b", re.IGNORECASE),
    re.compile(r"\bno[\s_-]?ack\b", re.IGNORECASE),
    re.compile(r"\bno[\s_-]?op\b", re.IGNORECASE),
    re.compile(r"\bstanding[\s_-]?down\b", re.IGNORECASE),
    re.compile(r"\bclosed\b", re.IGNORECASE),
    re.compile(r"\bno\s+further\s+action\b", re.IGNORECASE),
]


def terminal_bot_suppression_enabled(env: "os._Environ[str] | None" = None) -> bool:
    """Return whether terminal bot-message suppression is enabled.

    Auditable and default-enabled: only an explicit falsy value disables it.
    """

    environ = os.environ if env is None else env
    raw = environ.get(SUPPRESS_TERMINAL_BOT_MESSAGES_ENV, "").strip().lower()
    if raw in _FALSY:
        return False
    return True


def message_requests_no_visible_ack(text: str) -> bool:
    """Return True when *text* declares no visible ACK/final is wanted."""

    if not text or not text.strip():
        return False
    if re.search(r"\brequires[\s_-]?ack\b", text, re.IGNORECASE):
        return False
    return bool(_NO_VISIBLE_ACK_PATTERN.search(text))


def should_suppress_terminal_bot_message(text: str) -> bool:
    """Classify bot-authored *text* as terminal/no-work chatter.

    Returns ``True`` when the message should be dropped before agent dispatch.
    Pure function: callers gate on ``is_bot`` and on
    :func:`terminal_bot_suppression_enabled` themselves.
    """

    if not text or not text.strip():
        return False

    # Transport receipts with ack_policy:none are deliberately non-actionable
    # acknowledgements of delivery. Suppress only when the top-level structured
    # kind is a receipt. Do not scan every embedded/code-quoted kind token: a
    # transport_test instruction may quote the receipt text the recipient should
    # send, and suppressing the instruction would swallow real work.
    first_kind = _KIND_PATTERN.search(text)
    if (
        message_requests_no_visible_ack(text)
        and first_kind is not None
        and first_kind.group(1).lower() in _TRANSPORT_RECEIPT_KINDS
    ):
        return True

    # Actionable markers win: preserve genuine handoffs/requests even if the
    # message also contains terminal-looking words.
    for pattern in _ACTIONABLE_PATTERNS:
        if pattern.search(text):
            return False

    for pattern in _TERMINAL_PATTERNS:
        if pattern.search(text):
            return True

    return False
