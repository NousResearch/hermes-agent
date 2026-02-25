"""
Sticker description cache for Telegram.

When users send stickers, we describe them via the vision tool and cache
the descriptions keyed by file_unique_id so we don't re-analyze the same
sticker image on every send. Descriptions are concise (1-2 sentences).

Cache location: ~/.hermes/sticker_cache.json
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

CACHE_PATH = Path(os.path.expanduser("~/.hermes/sticker_cache.json"))

# Vision prompt for describing stickers -- kept concise to save tokens
STICKER_VISION_PROMPT = (
    "Describe this sticker in 1-2 sentences. Focus on what it depicts -- "
    "character, action, emotion. Be concise and objective."
)


def _load_cache() -> dict:
    """Load the sticker cache from disk."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Save the sticker cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_cached_description(file_unique_id: str) -> Optional[dict]:
    """
    Look up a cached sticker description.

    Returns:
        dict with keys {description, emoji, set_name, cached_at} or None.
    """
    cache = _load_cache()
    return cache.get(file_unique_id)


def cache_sticker_description(
    file_unique_id: str,
    description: str,
    emoji: str = "",
    set_name: str = "",
) -> None:
    """
    Store a sticker description in the cache.

    Args:
        file_unique_id: Telegram's stable sticker identifier.
        description:    Vision-generated description text.
        emoji:          Associated emoji (e.g. "😀").
        set_name:       Sticker set name if available.
    """
    cache = _load_cache()
    cache[file_unique_id] = {
        "description": description,
        "emoji": emoji,
        "set_name": set_name,
        "cached_at": time.time(),
    }
    _save_cache(cache)


def build_sticker_injection(
    description: str,
    emoji: str = "",
    set_name: str = "",
) -> str:
    """
    Build the warm-style injection text for a sticker description.

    Returns a string like:
      [The user sent a sticker 😀 from "MyPack"~ It shows: "A cat waving" (=^.w.^=)]
    """
    context = ""
    if set_name and emoji:
        context = f' {emoji} from "{set_name}"'
    elif emoji:
        context = f" {emoji}"

    return f'[The user sent a sticker{context}~ It shows: "{description}" (=^.w.^=)]'


def build_animated_sticker_injection(emoji: str = "") -> str:
    """
    Build injection text for animated/video stickers we can't analyze.
    """
    if emoji:
        return (
            f"[The user sent an animated sticker {emoji}~ "
            f"I can't see animated ones yet, but the emoji suggests: {emoji}]"
        )
    return "[The user sent an animated sticker~ I can't see animated ones yet]"
