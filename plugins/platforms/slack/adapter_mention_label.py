"""How the bot's own ``<@BOT>`` mention reads in the text the agent receives.

Legacy (default): every self-mention is removed. With ``inline_mention_label: true`` only the
leading run that addresses the bot is dropped; a later one becomes ``@<bot name> (you)`` so
"ask @bot later about X" keeps its meaning. Routing (``is_mentioned``, ``reply_expected``) reads
the raw event text, so neither mode changes who the message is for.
"""
import re
from typing import Any

_TRUE = {"1", "true", "yes", "on"}


def inline_mention_label_enabled(extra: dict) -> bool:
    raw = (extra or {}).get("inline_mention_label", False)
    return raw.strip().lower() in _TRUE if isinstance(raw, str) else bool(raw)


def bot_display_name(adapter: Any, team_id: str) -> str:
    """The bot's handle in ``team_id`` (multi-workspace installs), else the primary one."""
    names = getattr(adapter, "_team_bot_names", None) or {}
    return ((team_id and names.get(team_id)) or getattr(adapter, "_bot_display_name", None) or "").strip()


def self_mention_rewritten(adapter: Any, text: str, bot_uid: str, team_id: str) -> str:
    """``text`` as the agent should see it, per the adapter's ``inline_mention_label`` setting."""
    return rewrite_self_mentions(
        text, bot_uid, label=inline_mention_label_enabled(adapter.config.extra),
        name=bot_display_name(adapter, team_id))


def rewrite_self_mentions(text: str, bot_uid: str, *, label: bool, name: str) -> str:
    """``text`` with the bot's own mention handled per the mode described in the module docstring."""
    legacy = text.replace(f"<@{bot_uid}>", "").strip()
    if not label or not name:
        return legacy
    mention = rf"<@{re.escape(bot_uid)}(?:\|[^>]*)?>"
    labelled = re.sub(mention, lambda _m: f"@{name} (you)", re.sub(
        rf"^\s*(?:{mention}[\s,:]*)+", "", text)).strip()
    # Display must never create a command: "@bot, /stop" stays text, as the canonical probe saw it.
    return legacy if labelled.startswith("/") and not legacy.startswith("/") else labelled
