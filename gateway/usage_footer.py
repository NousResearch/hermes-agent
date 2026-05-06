from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

from gateway.config import Platform

FOOTER_SCRIPT = "/Users/davielam/clawd/scripts/usage-footer.sh"
_FOOTER_CACHE_TTL_SECONDS = 60.0
_footer_cache_lock = threading.Lock()
_cached_footer = ""
_cached_footer_at = 0.0


def _is_telegram_adapter(adapter: Any) -> bool:
    platform = getattr(adapter, "platform", None)
    if platform == Platform.TELEGRAM:
        return True
    name = str(getattr(adapter, "name", "") or "").lower()
    return name == "telegram"


def _read_usage_footer() -> str:
    try:
        result = subprocess.run(
            [FOOTER_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _load_usage_footer() -> str:
    global _cached_footer, _cached_footer_at
    now = time.monotonic()
    with _footer_cache_lock:
        if _cached_footer and (now - _cached_footer_at) < _FOOTER_CACHE_TTL_SECONDS:
            return _cached_footer

    footer = _read_usage_footer()
    if not footer:
        return ""

    with _footer_cache_lock:
        _cached_footer = footer
        _cached_footer_at = time.monotonic()
    return footer


def get_usage_footer(adapter: Any, text: str | None = None) -> str:
    if not _is_telegram_adapter(adapter):
        return ""
    footer = _load_usage_footer().strip()
    if not footer:
        return ""
    if text and footer in text:
        return ""
    return footer


def append_usage_footer(text: str, footer: str) -> str:
    clean_text = (text or "").rstrip()
    clean_footer = (footer or "").strip()
    if not clean_footer:
        return clean_text
    if clean_footer in clean_text:
        return clean_text
    if not clean_text:
        return clean_footer
    return f"{clean_text}\n\n{clean_footer}"


def maybe_append_usage_footer(adapter: Any, text: str | None = None) -> str:
    clean_text = (text or "").rstrip()
    footer = get_usage_footer(adapter, clean_text)
    return append_usage_footer(clean_text, footer)


async def send_usage_footer(adapter: Any, chat_id: str, text: str | None = None, metadata: dict | None = None):
    footer = get_usage_footer(adapter, text)
    if not footer:
        return None
    try:
        return await adapter.send(chat_id=chat_id, content=footer, metadata=metadata)
    except Exception:
        return None


def clear_usage_footer_cache() -> None:
    global _cached_footer, _cached_footer_at
    with _footer_cache_lock:
        _cached_footer = ""
        _cached_footer_at = 0.0
