"""Local index of what we've sent (and, for WhatsApp, received) keyed by ``(chat_id, message_id)``.

Telegram does NOT echo a rich message's content back in ``reply_to_message`` (``.text``/``.caption``
empty, ``.api_kwargs`` None), and WhatsApp quotes carry only the quoted message's id (Cloud API) or a
thumbnail stub (Baileys) — never the original bytes. So a reply to something we sent arrives with no
quotable text and no way to re-fetch a quoted attachment. We remember ``message_id -> text`` and
``message_id -> [(local_path, mime)]`` at send/receive time and look them up by ``reply_to_id`` on
inbound. Best-effort and dependency-free: every operation swallows errors and degrades to a no-op /
``None`` / ``[]`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Optional
from utils import atomic_json_write

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
# ``HERMES_RICH_SENT_MAX_ENTRIES`` overrides the cap for long-lived gateways (the default
# keeps roughly a few days of sent messages at Hermes' current outbound volume).
_MAX_ENTRIES_ENV = "HERMES_RICH_SENT_MAX_ENTRIES"


def _max_entries() -> int:
    """Cap on indexed entries; env override wins so a long-lived gateway can widen it."""
    try:
        raw = (os.environ.get(_MAX_ENTRIES_ENV) or "").strip()
        if raw:
            return max(100, int(raw))
    except (TypeError, ValueError):
        pass
    return _MAX_ENTRIES
# ``atomic_json_write`` makes each WRITE atomic, not the load/merge/save triple.
# ``record_async`` runs ``_update`` on worker threads, so two concurrent callers
# (two inbound WhatsApp-Cloud messages, a Telegram send racing an edit) would
# otherwise each load the same pre-state and the later ``os.replace`` drops the
# other key.
_LOCK = threading.Lock()


def _store_path() -> str:
    from hermes_constants import get_hermes_home  # honors the active profile override
    return os.path.join(str(get_hermes_home()), "state", "rich_sent_index.json")


def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):  # missing, unreadable (permissions, a directory) or corrupt
        return {}
    return data if isinstance(data, dict) else {}


def _update(chat_id, message_id, fields: dict, *, require_existing: bool = False, monotonic_ts: bool = False) -> None:
    """Merge ``fields`` into the ``(chat_id, message_id)`` entry. No-op on any failure."""
    path = _store_path()
    with _LOCK:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = _load(path)
            key = f"{chat_id}:{message_id}"
            entry = data.get(key)
            entry = entry if isinstance(entry, dict) else {}
            if require_existing and not entry:
                return
            ts = int(time.time())
            if monotonic_ts:
                # Strictly newer than the previous stamp, so a refreshed entry sorts after
                # same-second neighbours instead of tying with them (ties are broken by
                # insertion order, which would evict the OLDER one first).
                ts = max(ts, int(entry.get("ts", 0)) + 1)
            data[key] = {**entry, **fields, "ts": ts}
            max_entries = _max_entries()
            if len(data) > max_entries:  # trim oldest by timestamp
                for k, _ in sorted(data.items(), key=lambda kv: kv[1].get("ts", 0))[: len(data) - max_entries]:
                    data.pop(k, None)
            atomic_json_write(path, data, indent=None)  # see _LOCK
        except Exception:
            return


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist ``text`` for ``(chat_id, message_id)``. No-op on any failure."""
    if not text or message_id is None or chat_id is None:
        return
    _update(chat_id, message_id, {"t": text[:_MAX_TEXT_CHARS]})


def record_media(chat_id, message_id, media: list[tuple[str, str]]) -> None:
    """Persist local attachment ``(path, mime)`` pairs for ``(chat_id, message_id)``."""
    if not media or message_id is None or chat_id is None:
        return
    _update(chat_id, message_id, {"m": [[str(p), str(mt or "")] for p, mt in media if p]})


def touch(chat_id, message_id) -> None:
    """Refresh an entry's timestamp so a message that gets quoted survives FIFO trimming.

    The index is trimmed oldest-first; a message the user keeps quoting is worth more than
    one that was never replied to. Unknown entries are left alone (no phantom rows).
    """
    if message_id is None or chat_id is None:
        return
    _update(chat_id, message_id, {}, require_existing=True, monotonic_ts=True)


async def record_async(chat_id, message_id, text: Optional[str]) -> None:
    """``record`` for coroutine callers: the read-modify-write + ``os.replace``
    runs on a worker thread so the event loop is not stalled by the filesystem."""
    await asyncio.to_thread(record, chat_id, message_id, text)


async def record_media_async(chat_id, message_id, media: list[tuple[str, str]]) -> None:
    """``record_media`` for coroutine callers; see ``record_async``."""
    await asyncio.to_thread(record_media, chat_id, message_id, media)


def _entry(chat_id, message_id) -> dict:
    if message_id is None or chat_id is None:
        return {}
    entry = _load(_store_path()).get(f"{chat_id}:{message_id}")
    return entry if isinstance(entry, dict) else {}


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    return _entry(chat_id, message_id).get("t") or None


def record_marker(chat_id, message_ids, key: str, value: str) -> None:
    """Tag each of ``message_ids`` (one logical send split into several messages) with ``key=value``."""
    for message_id in dict.fromkeys(str(m) for m in message_ids if m is not None):
        _update(chat_id, message_id, {key: value})


def lookup_marker(chat_id, message_id, key: str) -> Optional[str]:
    """The ``key`` tag :func:`record_marker` stored for ``(chat_id, message_id)``, or ``None``."""
    value = _entry(chat_id, message_id).get(key)
    return value if isinstance(value, str) and value else None


def lookup_media(chat_id, message_id) -> list[tuple[str, str]]:
    """Return stored ``(path, mime)`` pairs whose file still exists (attachments may be temp files)."""
    pairs = _entry(chat_id, message_id).get("m") or []
    return [(p, mt) for p, mt in pairs if isinstance(p, str) and os.path.isfile(p)]
