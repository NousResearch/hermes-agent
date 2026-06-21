"""Outgoing-text sanitiser with status-emoji exemption + optional foreign-mention strip.

Two passes:
  1. Em/en-dash strip from LLM-generated content. (Skipped if text starts
     with a Hermes status emoji.)
  2. Foreign @-mention strip: when a per-turn allowed-author context var is
     set AND env HERMES_STRIP_FOREIGN_MENTIONS=true, replace `<@id>` tokens
     where id is NOT in the allowed set with `@<someone>`. Lets the bot
     ping the message author and any user the author themself @-mentioned,
     but blocks accidental cross-pings (rule-4 SOUL violations).

Status-emoji-prefixed messages are exempt from BOTH passes — they're
Hermes' own UI strings and we want them verbatim.
"""
from __future__ import annotations

import contextvars
import os
import re

_DASHES = {
    "—": ", ",
    "–": ", ",
    "―": ", ",
}

_STATUS_EMOJIS = (
    "⚠️", "❌", "📦", "🔧", "⟳", "⏳",
    "📅", "⏰", "💻", "📚", "🔍", "🧠",
    "💾", "⚙️", "⚡", "✓", "✅",
)

_MENTION_RE = re.compile(r"<@[!&]?(\d+)>")

# Per-turn allowed mention IDs. Set by the platform handler before invoking
# the agent (e.g. {message.author.id, *[m.id for m in message.mentions]}).
# When empty, the foreign-mention pass is skipped.
_ALLOWED_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_allowed_mention_ids", default=frozenset()
)


def set_allowed_mention_ids(ids):
    """Set the per-turn allowlist. Returns a token; pass it to reset()."""
    return _ALLOWED_CTX.set(frozenset(str(i) for i in ids if i))


def reset_allowed_mention_ids(token) -> None:
    try:
        _ALLOWED_CTX.reset(token)
    except Exception:
        pass


def _strip_dashes(text: str) -> str:
    out = text
    for bad, good in _DASHES.items():
        if bad in out:
            out = out.replace(bad, good)
    return out


def _strip_foreign_mentions(text: str) -> str:
    if os.environ.get("HERMES_STRIP_FOREIGN_MENTIONS", "").lower() not in ("1", "true", "yes"):
        return text
    allowed = _ALLOWED_CTX.get()
    if not allowed:
        return text  # no per-turn context set; nothing to strip against

    def _sub(m: "re.Match[str]") -> str:
        uid = m.group(1)
        return m.group(0) if uid in allowed else "@<someone>"

    return _MENTION_RE.sub(_sub, text)


def clean(text):
    """Idempotent. Returns input unchanged if it has no targeted chars OR
    if it starts with a Hermes status-message emoji."""
    if not text or not isinstance(text, str):
        return text
    stripped = text.lstrip()
    for prefix in _STATUS_EMOJIS:
        if stripped.startswith(prefix):
            return text
    out = _strip_dashes(text)
    out = _strip_foreign_mentions(out)
    return out
