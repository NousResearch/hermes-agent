#!/usr/bin/env python3
"""Telegram inline command picker — searchable access to EVERY command/skill.

The BotCommand menu is capped (100/scope, Hermes uses 60), so inline mode (``@yourbot <query>``, 50 per
page) exposes the rest; tapping a result sends ``/cmd args`` as the user through the normal command path.
PTB-object-free on purpose: plain dicts keep catalog/filter/pagination unit-testable; the adapter converts
to ``InlineQueryResultArticle``. Inert until inline mode is enabled via BotFather ``/setinline``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

# Telegram hard limit: max 50 results per answerInlineQuery call.
PAGE_SIZE = 50

# Results depend on caller auth + installed skills: never share across users, keep short.
CACHE_TIME_SECONDS = 10


def collect_inline_catalog() -> list[dict[str, str]]:
    """Every dispatchable command as ``{name, description}``, first occurrence wins: core gateway commands
    first (menu gating), then plugin + skill commands with ``max_slots=None`` (no cap)."""
    catalog: list[dict[str, str]] = []
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


def filter_catalog(catalog: list[dict[str, str]], term: str) -> list[dict[str, str]]:
    """Rank *catalog* against *term*: prefix > name-substring > description.
    Empty term returns the full catalog in collection order (the "browse" view)."""
    term = (term or "").strip().lower().lstrip("/")
    if not term:
        return list(catalog)
    prefix: list[dict[str, str]] = []
    name_sub: list[dict[str, str]] = []
    desc_sub: list[dict[str, str]] = []
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


def build_inline_results(query: str, offset: str = "", page_size: int = PAGE_SIZE) -> tuple[list[dict[str, Any]], str]:
    """One page of inline results for *query*: first token filters the catalog; the remainder becomes the
    command argument (``@bot plan migrate auth`` → ``/plan migrate auth``). ``next_offset == ""`` = last page."""
    parts = (query or "").strip().split(None, 1)
    term = parts[0] if parts else ""
    args = parts[1].strip() if len(parts) > 1 else ""
    matches = filter_catalog(collect_inline_catalog(), term)
    try:
        # Client-controlled text: negatives must not slice the catalog from the tail.
        start = max(0, int(offset)) if offset else 0
    except (TypeError, ValueError):
        start = 0
    next_offset = str(start + page_size) if len(matches) > start + page_size else ""
    results: list[dict[str, Any]] = []
    for item in matches[start:start + page_size]:
        message_text = f"/{item['name']}" + (f" {args}" if args else "")
        results.append({
            "id": f"{start}:{item['name']}"[:64],  # offset-scoped: unique across pages
            "title": f"/{item['name']}",
            "description": (item.get("description") or "")[:100],
            "message_text": message_text[:4096],
        })
    return results, next_offset


class TelegramChoicePickerMixin:

    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Flat inline-keyboard picker (one tap → one value) for /reasoning, /fast, etc. Each choice dict:
        ``{"value": str, "label": str, "is_current": bool}``."""
        from plugins.platforms.telegram.adapter import InlineKeyboardButton, InlineKeyboardMarkup, SendResult
        def build():
            buttons = []
            for i, choice in enumerate(choices):
                label = str(choice.get("label") or choice.get("value") or "")
                if choice.get("is_current"):
                    label = f"✓ {label}"
                buttons.append(InlineKeyboardButton(label, callback_data=f"cp:{i}"))
            if not buttons:
                return SendResult(success=False, error="No choices")
            rows = [[button] for button in buttons] if (metadata or {}).get("choice_layout") == "vertical" else self._rows_of_two(buttons)
            keyboard = InlineKeyboardMarkup(rows)

            def _remember(msg):
                state_key = (str(chat_id), str(msg.message_id))
                state = {"msg_id": msg.message_id, "choices": choices, "session_key": session_key, "on_choice_selected": on_choice_selected}
                if metadata is not None and "requester_user_id" in metadata:
                    state["owner_user_id"] = str(metadata.get("requester_user_id") or "").strip()
                self._choice_picker_state[state_key] = state
                timeout = (metadata or {}).get("choice_timeout_seconds")
                if timeout is not None:
                    try:
                        timeout = max(0.0, float(timeout))
                    except (TypeError, ValueError):
                        return
                    async def expire():
                        await asyncio.sleep(timeout)
                        if self._choice_picker_state.get(state_key) is state:
                            self._choice_picker_state.pop(state_key, None)
                    task = asyncio.create_task(expire())
                    if not hasattr(self, "_choice_picker_cleanup_tasks"):
                        self._choice_picker_cleanup_tasks = set()
                    self._choice_picker_cleanup_tasks.add(task)
                    task.add_done_callback(self._choice_picker_cleanup_tasks.discard)
            return self.format_message(title), keyboard, _remember
        return await self._send_prompt(
            "send_choice_picker", chat_id, metadata, build, thread_id=metadata.get("thread_id") if metadata else None,
            reply_to_mode=self._reply_to_mode)

    async def _handle_choice_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle choice picker button taps (cp:<index>)."""
        from plugins.platforms.telegram.adapter import _toast, t
        picker_logger = logging.getLogger("plugins.platforms.telegram.adapter")
        state_key = (str(chat_id), str(getattr(getattr(query, "message", None), "message_id", "") or ""))
        state = self._choice_picker_state.get(state_key)
        if not state:
            await query.answer(text=_toast("platform.telegram.picker.expired_rerun"))
            return
        if "owner_user_id" in state:
            actor = str(getattr(query.from_user, "id", "") or "").strip()
            if not actor or actor != state["owner_user_id"]:
                await query.answer(text="⛔ Only the user who opened this picker can use it.")
                return
        try:
            choice = state["choices"][int(data[3:])]
        except (ValueError, IndexError):
            await query.answer(text=_toast("platform.telegram.picker.invalid_selection"))
            return
        callback = state.get("on_choice_selected")
        if not callback:
            await query.answer(text=_toast("platform.telegram.picker.expired"))
            return
        try:
            result_text = await callback(chat_id, str(choice.get("value") or ""))
        except Exception as exc:
            picker_logger.error("Choice picker selection failed: %s", exc, exc_info=True)
            result_text = t("platform.telegram.picker.apply_error", error=str(exc))
        await self._edit_result_text(query, result_text)
        await query.answer()
        self._choice_picker_state.pop(state_key, None)
