"""Sticker description cache for Telegram.

Stickers are described via the vision tool once and cached by file_unique_id
(``~/.hermes/sticker_cache.json``) so the same image is never re-analyzed.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

CACHE_PATH = get_hermes_home() / "sticker_cache.json"
_CACHE_PATH_AT_IMPORT = CACHE_PATH


def _resolve_cache_path() -> Path:
    """Active profile's cache file at call time: the patched ``CACHE_PATH`` when a test changed
    it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process serves
    every profile, so the import-time constant would pin every profile to the launch home."""
    return CACHE_PATH if CACHE_PATH != _CACHE_PATH_AT_IMPORT else get_hermes_home() / "sticker_cache.json"

# Kept concise to save tokens.
STICKER_VISION_PROMPT = (
    "Describe this sticker in 1-2 sentences. Focus on what it depicts -- "
    "character, action, emotion. Be concise and objective."
)


def _load_cache() -> dict:
    """Load the sticker cache from disk (utf-8-sig: BOM-tolerant read fix)."""
    path = _resolve_cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):  # OSError covers FileNotFoundError
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    atomic_json_write(_resolve_cache_path(), cache)


# Serializes the read-modify-write in ``cache_sticker_description``. Nothing
# re-acquires it while held (the async wrapper only dispatches the sync form
# to a worker thread), so a plain Lock suffices.
_CACHE_LOCK = threading.Lock()


def get_cached_description(file_unique_id: str) -> Optional[dict]:
    """Return ``{description, emoji, set_name, cached_at}`` or None."""
    return _load_cache().get(file_unique_id)


def cache_sticker_description(
    file_unique_id: str, description: str, emoji: str = "", set_name: str = ""
) -> None:
    """Store a vision-generated description under Telegram's stable sticker id.

    Blocking: ``atomic_json_write`` ends in ``os.replace``. Callers on the event
    loop must use :func:`cache_sticker_description_async`.
    """
    entry = {"description": description, "emoji": emoji, "set_name": set_name,
             "cached_at": time.time()}
    # The lock makes the load/mutate/save triple atomic across worker threads.
    with _CACHE_LOCK:
        _save_cache({**_load_cache(), file_unique_id: entry})


async def cache_sticker_description_async(
    file_unique_id: str, description: str, emoji: str = "", set_name: str = ""
) -> None:
    """Off-loop form of :func:`cache_sticker_description`.

    The write ends in ``os.replace``, whose duration is unbounded under
    filesystem pressure, and the only caller is Telegram's ``_handle_sticker``
    -- an inbound-message coroutine. Paying the rename inline stalls every
    adapter and every in-flight turn in the process for its duration.
    """
    await asyncio.to_thread(
        cache_sticker_description, file_unique_id, description, emoji, set_name
    )


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
