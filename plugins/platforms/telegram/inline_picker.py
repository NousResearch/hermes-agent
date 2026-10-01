#!/usr/bin/env python3
"""Telegram inline command picker — searchable access to EVERY command/skill.

The BotCommand menu is capped (100/scope, Hermes uses 60), so inline mode (``@yourbot <query>``, 50 per
page) exposes the rest; tapping a result sends ``/cmd args`` as the user through the normal command path.
PTB-object-free on purpose: plain dicts keep catalog/filter/pagination unit-testable; the adapter converts
to ``InlineQueryResultArticle``. Inert until inline mode is enabled via BotFather ``/setinline``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Telegram hard limit: max 50 results per answerInlineQuery call.
PAGE_SIZE = 50

# Results depend on caller auth + installed skills: never share across users, keep short.
CACHE_TIME_SECONDS = 10


def collect_inline_catalog() -> List[Dict[str, str]]:
    """Every dispatchable command as ``{name, description}``, first occurrence wins: core gateway commands
    first (menu gating), then plugin + skill commands with ``max_slots=None`` (no cap)."""
    catalog: List[Dict[str, str]] = []
    seen: set[str] = set()

    def _add(name, desc):
        if name and name not in seen:
            seen.add(name)
            catalog.append({"name": name, "description": desc or ""})

    try:
        from hermes_cli.commands_platforms import _collect_gateway_skill_entries, _sanitize_telegram_name, telegram_bot_commands
    except Exception:  # pragma: no cover - defensive
        logger.debug("inline picker: commands registry unavailable", exc_info=True)
        return catalog
    try:
        for name, desc in telegram_bot_commands():
            _add(name, desc)
    except Exception:
        logger.debug("inline picker: core command collection failed", exc_info=True)
    try:
        entries, _hidden = _collect_gateway_skill_entries(
            platform="telegram", max_slots=None, reserved_names=set(seen), desc_limit=100, sanitize_name=_sanitize_telegram_name,
        )
        for entry in entries:  # shape is (name, desc, cmd_key[, raw_name])
            _add(entry[0], entry[1])
    except Exception:
        logger.debug("inline picker: skill/plugin collection failed", exc_info=True)
    return catalog


def filter_catalog(catalog: List[Dict[str, str]], term: str) -> List[Dict[str, str]]:
    """Rank *catalog* against *term*: prefix > name-substring > description.
    Empty term returns the full catalog in collection order (the "browse" view)."""
    term = (term or "").strip().lower().lstrip("/")
    if not term:
        return list(catalog)
    prefix: List[Dict[str, str]] = []
    name_sub: List[Dict[str, str]] = []
    desc_sub: List[Dict[str, str]] = []
    norm_term = term.replace("-", "_")  # hyphens/underscores equivalent, mirroring command dispatch
    for item in catalog:
        norm_name = item["name"].lower().replace("-", "_")
        if norm_name.startswith(norm_term):
            prefix.append(item)
        elif norm_term in norm_name:
            name_sub.append(item)
        elif term in (item.get("description") or "").lower():
            desc_sub.append(item)
    return prefix + name_sub + desc_sub


def build_inline_results(query: str, offset: str = "", page_size: int = PAGE_SIZE) -> Tuple[List[Dict[str, Any]], str]:
    """One page of inline results for *query*: first token filters the catalog; the remainder becomes the
    command argument (``@bot plan migrate auth`` → ``/plan migrate auth``). ``next_offset == ""`` = last page."""
    parts = (query or "").strip().split(None, 1)
    term = parts[0] if parts else ""
    args = parts[1].strip() if len(parts) > 1 else ""
    matches = filter_catalog(collect_inline_catalog(), term)
    try:
        start = int(offset) if offset else 0
    except (TypeError, ValueError):
        start = 0
    next_offset = str(start + page_size) if len(matches) > start + page_size else ""
    results: List[Dict[str, Any]] = []
    for item in matches[start:start + page_size]:
        message_text = f"/{item['name']}" + (f" {args}" if args else "")
        results.append({
            "id": f"{start}:{item['name']}"[:64],  # offset-scoped: unique across pages
            "title": f"/{item['name']}",
            "description": (item.get("description") or "")[:100],
            "message_text": message_text[:4096],
        })
    return results, next_offset
