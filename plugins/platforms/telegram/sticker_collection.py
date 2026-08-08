"""
Persistent Telegram sticker collection.

When users send the bot stickers (or the agent imports a pack via
``refresh_from_sets``), we record each sticker's ``file_id`` here so the
agent can later send *native* Telegram stickers back with
``tg_send_sticker``. The collection is keyed by ``file_unique_id``
(Telegram's globally stable sticker identifier) and persisted as JSON at
``~/.hermes/telegram_stickers.json`` (profile-aware via
``get_hermes_home()``).

This module is intentionally separate from ``gateway/sticker_cache.py``:
that file is a vision-description cache keyed the same way, and we only
*read* from it (best-effort description backfill). We never trigger
vision calls ourselves.

Invariants:
- Missing or corrupt JSON is treated as an empty collection (never raises).
- Writes are atomic: ``tempfile.mkstemp`` + ``os.replace`` (mirrors
  ``gateway/sticker_cache.py``).
- Entries missing required fields ("dirty" entries) are skipped on read
  paths and dropped the next time ``record_sticker`` saves (self-heal).
- Capacity is capped at ``MAX_STICKERS``; overflow evicts the oldest
  entries by ``last_seen``.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

COLLECTION_VERSION = 1
MAX_STICKERS = 500

_VALID_KINDS = ("static", "animated", "video")


def _now() -> float:
    """Current epoch seconds (seam for tests)."""
    return time.time()


def _collection_path() -> Path:
    """Resolve the collection file lazily so HERMES_HOME isolation works."""
    return get_hermes_home() / "telegram_stickers.json"


def _empty_collection() -> Dict[str, Any]:
    return {"version": COLLECTION_VERSION, "stickers": {}}


def _valid_entry(entry: Any) -> bool:
    """True when an entry has every required field with a sane type."""
    if not isinstance(entry, dict):
        return False
    if not isinstance(entry.get("file_id"), str) or not entry["file_id"]:
        return False
    for key in ("emoji", "set_name", "kind", "description"):
        if not isinstance(entry.get(key), str):
            return False
    for key in ("first_seen", "last_seen"):
        if not isinstance(entry.get(key), (int, float)):
            return False
    return True


def _load_collection() -> Dict[str, Any]:
    """Load the collection from disk; corrupt/missing data reads as empty."""
    path = _collection_path()
    if not path.exists():
        return _empty_collection()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("[Telegram] Sticker collection unreadable; treating as empty: %s", path)
        return _empty_collection()
    if not isinstance(data, dict) or not isinstance(data.get("stickers"), dict):
        return _empty_collection()
    return data


def _save_collection(collection: Dict[str, Any]) -> None:
    """Persist the collection atomically (mkstemp + os.replace)."""
    path = _collection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _lookup_cached_description(file_unique_id: str) -> str:
    """
    Best-effort description from the vision cache (gateway/sticker_cache.py).

    Never raises, never triggers a vision call — a miss just returns "".
    """
    try:
        from gateway.sticker_cache import get_cached_description

        cached = get_cached_description(file_unique_id)
    except Exception:
        return ""
    if isinstance(cached, dict):
        description = cached.get("description")
        if isinstance(description, str):
            return description
    return ""


def sticker_kind(sticker: Any) -> str:
    """Derive the collection "kind" label from a Telegram Sticker object."""
    if getattr(sticker, "is_video", False):
        return "video"
    if getattr(sticker, "is_animated", False):
        return "animated"
    return "static"


def record_sticker(
    file_unique_id: str,
    file_id: str,
    emoji: str = "",
    set_name: str = "",
    kind: str = "static",
    description: str = "",
) -> bool:
    """
    Upsert a sticker into the collection and refresh its ``last_seen``.

    Returns True iff this call created a *new* entry (callers use this to
    decide whether to tell the model a sticker was just added).

    Description merge rule: a non-empty stored description is never
    overwritten here — explicit curation (``update_description``) is the
    only overwrite channel. When the stored description is empty and the
    caller passes "", we best-effort backfill from the vision cache.
    """
    if not file_unique_id or not file_id:
        logger.warning("[Telegram] record_sticker called without stable identity; skipping")
        return False

    collection = _load_collection()
    # Self-heal: drop dirty entries on every record save.
    stickers = {
        key: entry
        for key, entry in collection["stickers"].items()
        if _valid_entry(entry)
    }
    collection["stickers"] = stickers

    now = _now()
    existing = stickers.get(file_unique_id)
    is_new = existing is None

    if is_new:
        entry: Dict[str, Any] = {
            "file_id": file_id,
            "emoji": emoji or "",
            "set_name": set_name or "",
            "kind": kind if kind in _VALID_KINDS else "static",
            "description": "",
            "first_seen": now,
            "last_seen": now,
        }
        stickers[file_unique_id] = entry
    else:
        entry = existing
        entry["file_id"] = file_id
        if emoji:
            entry["emoji"] = emoji
        if set_name:
            entry["set_name"] = set_name
        if kind in _VALID_KINDS:
            entry["kind"] = kind
        entry["last_seen"] = now

    # Description merge: fill only when the stored value is empty.
    if not entry["description"]:
        new_description = description or _lookup_cached_description(file_unique_id)
        if new_description:
            entry["description"] = new_description

    # Capacity: evict oldest by last_seen, never the just-recorded entry.
    while len(stickers) > MAX_STICKERS:
        oldest_key = min(
            (key for key in stickers if key != file_unique_id),
            key=lambda key: (stickers[key]["last_seen"], key),
            default=None,
        )
        if oldest_key is None:
            break
        del stickers[oldest_key]

    _save_collection(collection)
    return is_new


def update_description(file_unique_id: str, description: str) -> bool:
    """
    Agent curation channel: unconditionally overwrite an entry's description.

    Passing "" clears the annotation. Returns False when the entry is
    unknown (or dirty).
    """
    collection = _load_collection()
    entry = collection["stickers"].get(file_unique_id)
    if not _valid_entry(entry):
        return False
    entry["description"] = description or ""
    _save_collection(collection)
    return True


def remove_sticker(file_unique_id: str) -> bool:
    """Delete an entry. Returns False when it wasn't in the collection."""
    collection = _load_collection()
    if file_unique_id not in collection["stickers"]:
        return False
    del collection["stickers"][file_unique_id]
    _save_collection(collection)
    return True


def _sorted_entries(collection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Valid entries as dicts (with file_unique_id), newest last_seen first."""
    entries = []
    for key, entry in collection["stickers"].items():
        if not _valid_entry(entry):
            continue
        entries.append({"file_unique_id": key, **entry})
    entries.sort(key=lambda e: e["last_seen"], reverse=True)
    return entries


def list_stickers(set_name: str = "", limit: int = 100) -> List[Dict[str, Any]]:
    """
    Read-only listing of the local collection (never touches the Bot API).

    Each dict includes ``file_unique_id`` for use as the selector in
    ``update_description`` / ``remove_sticker``.
    """
    entries = _sorted_entries(_load_collection())
    if set_name:
        entries = [e for e in entries if e["set_name"] == set_name]
    return entries[: max(limit, 0)]


def resolve(query: str, set_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Resolve a user/agent query string to a collection entry.

    Priority: exact ``file_id`` passthrough → exact ``file_unique_id`` →
    "set_name:emoji" → bare emoji. Ties on emoji matches break toward the
    most recent ``last_seen``. Returns None when nothing matches.
    """
    query = (query or "").strip()
    if not query:
        return None

    collection = _load_collection()
    stickers = collection["stickers"]

    for key, entry in stickers.items():
        if _valid_entry(entry) and entry["file_id"] == query:
            return {"file_unique_id": key, **entry}

    entry = stickers.get(query)
    if _valid_entry(entry):
        return {"file_unique_id": query, **entry}

    entries = _sorted_entries(collection)

    if ":" in query:
        set_part, _, emoji_part = query.partition(":")
        for candidate in entries:
            if candidate["set_name"] == set_part and candidate["emoji"] == emoji_part:
                return candidate

    for candidate in entries:
        if candidate["emoji"] != query:
            continue
        if set_name and candidate["set_name"] != set_name:
            continue
        return candidate

    return None


def format_for_prompt(limit: int = 100) -> str:
    """
    Compact one-line-per-sticker listing for prompt injection.

    Lines look like:  - 😀 "a cat waving" (set: MyPack, kind: static)
    Sorted by last_seen, newest first. Empty collection returns "".
    """
    entries = _sorted_entries(_load_collection())
    if not entries:
        return ""
    lines = []
    for entry in entries[: max(limit, 0)]:
        description = f' "{entry["description"]}"' if entry["description"] else ""
        details = []
        if entry["set_name"]:
            details.append(f"set: {entry['set_name']}")
        details.append(f"kind: {entry['kind']}")
        lines.append(f"- {entry['emoji']}{description} ({', '.join(details)})")
    return "\n".join(lines)


async def refresh_from_sets(bot: Any, set_names: List[str]) -> Dict[str, int]:
    """
    Bulk-import sticker packs via the Bot API (``bot.get_sticker_set``).

    Used as a startup seed (config ``telegram.sticker_sets``) and by the
    ``tg_manage_stickers`` add_set action. Per-set failures are logged and
    skipped. Recording goes through ``record_sticker``, so descriptions
    enjoy the same best-effort vision-cache backfill.

    Returns summary counts: {"sets", "sets_failed", "stickers", "new"}.
    """
    summary = {"sets": 0, "sets_failed": 0, "stickers": 0, "new": 0}
    for raw_name in set_names or []:
        name = (raw_name or "").strip()
        if not name:
            continue
        try:
            sticker_set = await bot.get_sticker_set(name)
        except Exception as e:
            logger.warning("[Telegram] Failed to fetch sticker set %r: %s", name, e)
            summary["sets_failed"] += 1
            continue
        summary["sets"] += 1
        for sticker in getattr(sticker_set, "stickers", None) or []:
            try:
                is_new = record_sticker(
                    file_unique_id=sticker.file_unique_id,
                    file_id=sticker.file_id,
                    emoji=getattr(sticker, "emoji", None) or "",
                    set_name=getattr(sticker, "set_name", None) or name,
                    kind=sticker_kind(sticker),
                )
            except Exception as e:
                logger.warning("[Telegram] Failed to record sticker from set %r: %s", name, e)
                continue
            summary["stickers"] += 1
            if is_new:
                summary["new"] += 1
    return summary


def build_sticker_collection_note() -> str:
    """
    Render the first-turn context note describing the sticker collection.

    Injected once per session through the gateway turn-note channel (see
    plan §3); "" when the collection is empty so callers skip injection.
    """
    listing = format_for_prompt()
    if not listing:
        return ""
    return (
        "## Your Telegram Sticker Collection\n"
        f"{listing}\n"
        "To send one, call tg_send_sticker with its emoji (and set name). "
        "You may curate this collection with tg_manage_stickers (annotate "
        "descriptions, remove entries, import a set). Never draw sticker-like "
        "PNGs — that is the wrong path."
    )
