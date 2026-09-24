"""Sticker description cache for Telegram.

Stickers are described via the vision tool once and cached by file_unique_id
(``~/.hermes/sticker_cache.json``) so the same image is never re-analyzed.
"""

import json
import time
from typing import Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

CACHE_PATH = get_hermes_home() / "sticker_cache.json"

# Kept concise to save tokens.
STICKER_VISION_PROMPT = (
    "Describe this sticker in 1-2 sentences. Focus on what it depicts -- "
    "character, action, emotion. Be concise and objective."
)


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    atomic_json_write(CACHE_PATH, cache)


def get_cached_description(file_unique_id: str) -> Optional[dict]:
    """Return ``{description, emoji, set_name, cached_at}`` or None."""
    return _load_cache().get(file_unique_id)


def cache_sticker_description(
    file_unique_id: str, description: str, emoji: str = "", set_name: str = ""
) -> None:
    """Store a vision-generated description under Telegram's stable sticker id."""
    entry = {"description": description, "emoji": emoji, "set_name": set_name,
             "cached_at": time.time()}
    _save_cache({**_load_cache(), file_unique_id: entry})


def build_sticker_injection(description: str, emoji: str = "", set_name: str = "") -> str:
    """Warm-style injection text, e.g.
    ``[The user sent a sticker 😀 from "MyPack"~ It shows: "A cat waving" (=^.w.^=)]``.
    ``set_name`` is only shown together with an emoji."""
    context = f" {emoji}" if emoji else ""
    if set_name and emoji:
        context += f' from "{set_name}"'
    return f'[The user sent a sticker{context}~ It shows: "{description}" (=^.w.^=)]'


def build_animated_sticker_injection(emoji: str = "") -> str:
    """Injection text for animated/video stickers we can't analyze."""
    if emoji:
        return (f"[The user sent an animated sticker {emoji}~ "
                f"I can't see animated ones yet, but the emoji suggests: {emoji}]")
    return "[The user sent an animated sticker~ I can't see animated ones yet]"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
import tempfile  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
