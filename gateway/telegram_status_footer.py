"""Helpers for Hermes Telegram DM status footers."""

from __future__ import annotations

from typing import Any

from gateway.config import Platform

_FOOTER_SEPARATOR = "────────────"


def _format_token_count(value: int) -> str:
    if value >= 1000:
        return f"{round(value / 1000)}k"
    return str(value)


def _normalize_reasoning_label(reasoning_effort: str | None) -> str:
    label = str(reasoning_effort or "").strip()
    return label or "default"


def _coerce_platform(platform: Any) -> str:
    if isinstance(platform, Platform):
        return platform.value
    return str(platform or "").strip().lower()


def build_telegram_status_footer(
    *,
    model: str | None,
    reasoning_effort: str | None,
    current_tokens: int,
    context_window: int,
) -> str:
    model_label = str(model or "unknown").strip() or "unknown"
    reasoning_label = _normalize_reasoning_label(reasoning_effort)
    current = max(int(current_tokens or 0), 0)
    maximum = max(int(context_window or 0), 0)
    percent = "?"
    if maximum > 0:
        pct = round((current / maximum) * 100)
        pct = max(0, min(100, pct))
        percent = f"{pct}%"
    max_label = _format_token_count(maximum) if maximum > 0 else "?"
    current_label = _format_token_count(current)
    return (
        f"{_FOOTER_SEPARATOR}\n"
        f"🧠 {model_label} · 💨 {reasoning_label}\n"
        f"📊 {current_label} / {max_label} · {percent}"
    )


def maybe_append_telegram_status_footer(
    content: str,
    *,
    platform: Platform | str | None,
    chat_type: str | None,
    model: str | None,
    reasoning_effort: str | None,
    current_tokens: int,
    context_window: int,
) -> str:
    text = content or ""
    if not text.strip():
        return text
    if _FOOTER_SEPARATOR in text:
        return text
    if _coerce_platform(platform) != Platform.TELEGRAM.value:
        return text
    if str(chat_type or "").strip().lower() != "dm":
        return text
    footer = build_telegram_status_footer(
        model=model,
        reasoning_effort=reasoning_effort,
        current_tokens=current_tokens,
        context_window=context_window,
    )
    return f"{text.rstrip()}\n\n{footer}"
