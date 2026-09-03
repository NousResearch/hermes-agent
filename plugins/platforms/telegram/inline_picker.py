#!/usr/bin/env python3
"""Telegram inline command picker — searchable access to EVERY command/skill.

Telegram's BotCommand menu is capped (100 per scope, ~4KB payload; Hermes
defaults to 60 slots), so most skill commands can never appear in the ``/``
menu. Inline mode has no such cap: typing ``@yourbot <query>`` in any chat
asks the bot for results live, per keystroke, paginated 50 at a time — the
same trick Discord's ``/skill`` autocomplete uses (options fetched
dynamically, nothing pre-registered).

Tapping a result sends the command text (e.g. ``/plan migrate the auth``)
into the chat as the user. Because the sent message starts with ``/``, the
bot receives it even under Telegram's default privacy mode ("messages with
commands meant for the bot" are always delivered), and it dispatches through
the existing command path — zero new dispatch code.

This module is PTB-object-free on purpose: it returns plain dicts so the
catalog/filter/pagination logic is unit-testable without python-telegram-bot
installed. The adapter converts dicts to ``InlineQueryResultArticle``.

Setup note (docs): inline mode must be enabled once per bot via BotFather's
``/setinline``. Until then Telegram never delivers ``inline_query`` updates,
so the registered handler is inert — safe to ship enabled by default.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Telegram hard limit: max 50 results per answerInlineQuery call.
PAGE_SIZE = 50

# Results depend on the caller's auth and the install's skill set — never
# share cached results across users, and keep the cache short so freshly
# installed skills appear quickly.
CACHE_TIME_SECONDS = 10


def collect_inline_catalog() -> List[Dict[str, str]]:
    """Return every dispatchable command as ``{name, description}`` dicts.

    Sources, deduped in priority order (first occurrence wins):
      1. Core gateway-visible ``CommandDef`` commands (Telegram-sanitized
         names, same gating as the BotCommand menu).
      2. Plugin slash commands + built-in skill commands via the shared
         collector — with ``max_slots=None`` so NOTHING is trimmed. This is
         the whole point: the inline picker has no cap.

    Skill entries honor the same filtering as the menu (hub excluded,
    per-platform disabled excluded, external-dir allowlist).
    """
    catalog: List[Dict[str, str]] = []
    seen: set[str] = set()

    try:
        from hermes_cli.commands import (
            _collect_gateway_skill_entries,
            _sanitize_telegram_name,
            telegram_bot_commands,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("inline picker: commands registry unavailable", exc_info=True)
        return catalog

    try:
        for name, desc in telegram_bot_commands():
            if name and name not in seen:
                seen.add(name)
                catalog.append({"name": name, "description": desc or ""})
    except Exception:
        logger.debug("inline picker: core command collection failed", exc_info=True)

    try:
        entries, _hidden = _collect_gateway_skill_entries(
            platform="telegram",
            max_slots=None,  # inline mode has no cap — collect everything
            reserved_names=set(seen),
            desc_limit=100,
            sanitize_name=_sanitize_telegram_name,
        )
        for entry in entries:
            # Entry shape is (name, desc, cmd_key[, raw_name]) — tolerate both.
            name, desc = entry[0], entry[1]
            if name and name not in seen:
                seen.add(name)
                catalog.append({"name": name, "description": desc or ""})
    except Exception:
        logger.debug("inline picker: skill/plugin collection failed", exc_info=True)

    return catalog


def filter_catalog(catalog: List[Dict[str, str]], term: str) -> List[Dict[str, str]]:
    """Rank *catalog* against *term*: prefix > name-substring > description.

    Empty term returns the full catalog in its collection order (core first,
    then plugins, then skills alphabetically) — the "browse" view.
    """
    term = (term or "").strip().lower().lstrip("/")
    if not term:
        return list(catalog)

    prefix: List[Dict[str, str]] = []
    name_sub: List[Dict[str, str]] = []
    desc_sub: List[Dict[str, str]] = []
    # Treat hyphens/underscores as equivalent, mirroring command dispatch.
    norm_term = term.replace("-", "_")
    for item in catalog:
        norm_name = item["name"].lower().replace("-", "_")
        if norm_name.startswith(norm_term):
            prefix.append(item)
        elif norm_term in norm_name:
            name_sub.append(item)
        elif term in (item.get("description") or "").lower():
            desc_sub.append(item)
    return prefix + name_sub + desc_sub


def build_inline_results(
    query: str,
    offset: str = "",
    page_size: int = PAGE_SIZE,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build one page of inline results for *query*.

    The first whitespace-separated token of *query* filters the catalog; any
    remainder is carried into the sent command as its argument. Example:
    ``@bot plan migrate auth to OIDC`` → filter ``plan``, and tapping the
    ``/plan`` result sends ``/plan migrate auth to OIDC``.

    Returns ``(results, next_offset)`` where each result is
    ``{"id", "title", "description", "message_text"}`` and *next_offset* is
    ``""`` when this is the last page (Telegram's stop signal).
    """
    query = (query or "").strip()
    parts = query.split(None, 1)
    term = parts[0] if parts else ""
    args = parts[1].strip() if len(parts) > 1 else ""

    matches = filter_catalog(collect_inline_catalog(), term)

    try:
        start = int(offset) if offset else 0
    except (TypeError, ValueError):
        start = 0
    page = matches[start:start + page_size]
    next_offset = str(start + page_size) if len(matches) > start + page_size else ""

    results: List[Dict[str, Any]] = []
    for item in page:
        message_text = f"/{item['name']}"
        if args:
            message_text += f" {args}"
        results.append(
            {
                # Offset-scoped ids stay unique across pages of one query.
                "id": f"{start}:{item['name']}"[:64],
                "title": f"/{item['name']}",
                "description": (item.get("description") or "")[:100],
                "message_text": message_text[:4096],
            }
        )
    return results, next_offset
