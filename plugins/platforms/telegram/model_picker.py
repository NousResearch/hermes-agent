#!/usr/bin/env python3
"""Telegram model-picker presentation logic — PTB-row builders and pricing formatting.

Extracted from ``adapter.py`` (Sept 2026) under the god-file split rule: a 7,488-line
adapter is the signal to move a topic into its own ``<stem>_<topic>`` sibling. These five
helpers were the self-free core of the picker cluster; the callback handlers that still
need adapter state (``_picker_selection``, ``_picker_switch``, ``_handle_*_callback``,
``send_*_picker``) stay in ``adapter.py`` and import from here.

Like ``inline_picker.py``, this module holds no adapter ``self`` state, so the pricing
maths and keyboard-row shapes are unit-testable without constructing a TelegramAdapter
or a bot token.

Why the pricing lookup is ``cached_only=True`` (audit finding P2.11): this runs INSIDE a
PTB callback handler. A network call there blocks the asyncio event loop, so a slow or
dead pricing endpoint stalls every other Telegram update, including inbound messages.
``get_pricing_for_provider(..., cached_only=True)`` reads only the on-disk cache and can
never block; the refresh happens elsewhere, on its own schedule.

Bodies were moved VERBATIM from the adapter, changing only: the leading underscore on
each name, removal of the now-pointless ``@staticmethod``/``@classmethod`` decorators, and
``cls._get_model_pricing_detail`` → ``get_model_pricing_detail`` (it is now a module-level
sibling call, not a class attribute lookup). Do not reformat casually: this file plus the
adapter's diff is the only record of the split surviving a future ``hermes update``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from telegram import InlineKeyboardButton

from agent.i18n import t

# Telegram caps `query.answer(text=...)` payloads at 200 chars; the adapter's
# `_toast` applies the same cut. Mirrored here (rather than imported) so this
# module keeps no adapter dependency — that is what makes it unit-testable.
_TOAST_LIMIT = 200


def _toast(key: str, **kwargs: Any) -> str:
    """``t()`` for ``query.answer(text=...)`` payloads, cut at Telegram's 200-char toast cap."""
    return t(key, **kwargs)[:_TOAST_LIMIT]


def picker_nav_row(page: int, total_pages: int, prefix: str) -> list:
    """``◀ Prev | n/N | Next ▶`` row (``prefix`` = ``mpv``/``mg`` page callback)."""
    nav: list = []
    if page > 0:
        nav.append(InlineKeyboardButton(t("platform.telegram.picker.prev"), callback_data=f"{prefix}:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(t("platform.telegram.picker.next"), callback_data=f"{prefix}:{page + 1}"))
    return nav

def picker_back_cancel_row() -> list:
    return [InlineKeyboardButton(t("platform.telegram.picker.back"), callback_data="mb"), InlineKeyboardButton(t("platform.telegram.picker.cancel"), callback_data="mx")]

def get_model_pricing_detail(provider_slug: str, model_id: str, live_pricing: Optional[dict] = None) -> dict:
    """Extract exact pricing and discount/promo information for a model on a given provider."""
    if live_pricing is None:
        try:
            from hermes_cli.models_pricing import get_pricing_for_provider
            live_pricing = get_pricing_for_provider(provider_slug, cached_only=True) or {}
        except Exception:
            live_pricing = {}

    p = live_pricing.get(model_id) if isinstance(live_pricing, dict) else None
    if p and isinstance(p, dict):
        try:
            from hermes_cli.models_pricing import _format_price_per_mtok, compute_sale_discount
            inp_raw = str(p.get("prompt", "") or "")
            out_raw = str(p.get("completion", "") or "")
            cache_raw = str(p.get("input_cache_read", "") or "")
            inp = _format_price_per_mtok(inp_raw) if inp_raw != "" else ""
            out = _format_price_per_mtok(out_raw) if out_raw != "" else ""
            cache = _format_price_per_mtok(cache_raw) if cache_raw else ""

            is_free = (inp == "free" and out in ("free", "")) or (inp_raw == "0" and out_raw == "0") or (":free" in model_id.lower())
            sale = compute_sale_discount(inp_raw, out_raw, p.get("original"))
            discount_pct = sale[0] if sale else None
            was_inp = _format_price_per_mtok(str(sale[1])) if sale and sale[1] else ""
            was_out = _format_price_per_mtok(str(sale[2])) if sale and sale[2] else ""

            return {
                "has_pricing": bool(inp or out or is_free),
                "is_free": is_free,
                "input": inp,
                "output": out,
                "cache": cache,
                "discount_percent": discount_pct,
                "was_input": was_inp,
                "was_output": was_out,
            }
        except Exception:
            pass

    try:
        from agent.models_dev import fetch_models_dev, _models_dev_id
        from hermes_cli.models_pricing import _format_price_per_mtok
        mdev_prov = _models_dev_id(provider_slug) or provider_slug
        mdev_data = fetch_models_dev(allow_network=False)
        prov_models = mdev_data.get(mdev_prov, {}).get("models", {})
        m_info = prov_models.get(model_id) or prov_models.get(model_id.split("/")[-1])
        if m_info and m_info.get("cost"):
            cost = m_info["cost"]
            inp_cost = cost.get("input")
            out_cost = cost.get("output")
            cache_cost = cost.get("cache_read")

            inp = _format_price_per_mtok(str(inp_cost / 1_000_000)) if inp_cost is not None else ""
            out = _format_price_per_mtok(str(out_cost / 1_000_000)) if out_cost is not None else ""
            cache = _format_price_per_mtok(str(cache_cost / 1_000_000)) if cache_cost is not None else ""
            is_free = (inp == "free" and out in ("free", "")) or (inp_cost == 0 and out_cost == 0) or (":free" in model_id.lower())

            return {
                "has_pricing": True,
                "is_free": is_free,
                "input": inp,
                "output": out,
                "cache": cache,
                "discount_percent": None,
                "was_input": "",
                "was_output": "",
            }
    except Exception:
        pass

    is_free = ":free" in model_id.lower() or model_id.lower().startswith("free-")
    return {
        "has_pricing": is_free,
        "is_free": is_free,
        "input": "free" if is_free else "",
        "output": "free" if is_free else "",
        "cache": "",
        "discount_percent": None,
        "was_input": "",
        "was_output": "",
    }

def format_model_pricing_line(model_id: str, info: dict) -> str:
    """Format a single model's pricing/discount summary line for Telegram message text."""
    short = model_id.split("/")[-1] if "/" in model_id else model_id
    if not info or not info.get("has_pricing"):
        return f"• `{short}`"

    if info.get("is_free"):
        promo_txt = " _(100% OFF promo)_" if info.get("discount_percent") == 100 and info.get("was_input") else ""
        return f"• `{short}` — 🆓 *Free*{promo_txt}"

    parts = []
    if info.get("input") and info.get("output"):
        parts.append(f"📥 {info['input']}/M · 📤 {info['output']}/M")
    elif info.get("input"):
        parts.append(f"📥 {info['input']}/M")
    elif info.get("output"):
        parts.append(f"📤 {info['output']}/M")

    if info.get("cache"):
        parts.append(f"⚡ Cache: {info['cache']}/M")

    price_str = " · ".join(parts) if parts else ""

    disc = info.get("discount_percent")
    if disc and 0 < disc < 100:
        was_str = ""
        if info.get("was_input") or info.get("was_output"):
            was_str = f" _(was {info.get('was_input', '?')}/{info.get('was_output', '?')})_"
        promo = f" 🔥 *{disc}% OFF*{was_str}"
        return f"• `{short}` — {price_str}{promo}"
    elif disc == 100:
        return f"• `{short}` — 🆓 *Free (Promo)*"

    return f"• `{short}` — {price_str}" if price_str else f"• `{short}`"

def sort_models_for_picker(models: list, provider_slug: str, pricing: Optional[dict] = None) -> list:
    """Order models: free models first (from last to first), then regular models (from last to first),
    then GPT and Claude models at the end (from last to first)."""
    def _is_free(mid: str) -> bool:
        if ":free" in mid.lower() or mid.lower().startswith("free-"):
            return True
        info = get_model_pricing_detail(provider_slug, mid, pricing)
        return bool(info.get("is_free"))

    def _is_gpt_or_claude(mid: str) -> bool:
        m = mid.lower()
        return "gpt" in m or "claude" in m or "openai/" in m or "anthropic/" in m

    reversed_models = list(reversed(models))
    free_group = [m for m in reversed_models if _is_free(m)]
    regular_group = [m for m in reversed_models if not _is_free(m) and not _is_gpt_or_claude(m)]
    gpt_claude_group = [m for m in reversed_models if not _is_free(m) and _is_gpt_or_claude(m)]

    return free_group + regular_group + gpt_claude_group


async def picker_selection(query, state: dict, raw_idx: str) -> Optional[tuple]:
    """Resolve ``mm:``/``mc:`` index → ``(idx, model_id, provider_slug, callback)``; answers + None on error."""
    try:
        idx = int(raw_idx)
    except ValueError:
        await query.answer(text=_toast("platform.telegram.picker.invalid_selection"))
        return None
    model_list = state.get("model_list", [])
    if idx < 0 or idx >= len(model_list):
        await query.answer(text=_toast("platform.telegram.picker.invalid_model_index"))
        return None
    callback = state.get("on_model_selected")
    if not callback:
        await query.answer(text=_toast("platform.telegram.picker.expired"))
        return None
    return idx, model_list[idx], state.get("selected_provider", ""), callback
