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

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
_STORE_LOCK = threading.RLock()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only
    msvcrt = None


def _store_path() -> str:
    from hermes_constants import get_hermes_home  # honors the active profile override
    return os.path.join(str(get_hermes_home()), "state", "rich_sent_index.json")


def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@contextmanager
def _locked_store(path: str):
    """Serialize index read-modify-write cycles across gateway processes."""
    with _STORE_LOCK:
        lock_path = f"{path}.lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        if msvcrt and (not os.path.exists(lock_path) or os.path.getsize(lock_path) == 0):
            with open(lock_path, "w", encoding="utf-8") as fh:
                fh.write(" ")
        with open(lock_path, "a+", encoding="utf-8") as fh:
            if fcntl:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            elif msvcrt:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                with suppress(OSError):
                    if fcntl:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    elif msvcrt:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def _update(chat_id, message_id, fields: dict) -> None:
    """Merge ``fields`` into the ``(chat_id, message_id)`` entry. No-op on any failure."""
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _locked_store(path):
            data = _load(path)
            key = f"{chat_id}:{message_id}"
            entry = data.get(key)
            entry = entry if isinstance(entry, dict) else {}
            data[key] = {**entry, **fields, "ts": int(time.time())}
            if len(data) > _MAX_ENTRIES:  # trim oldest by timestamp
                for k, _ in sorted(data.items(), key=lambda kv: kv[1].get("ts", 0))[: len(data) - _MAX_ENTRIES]:
                    data.pop(k, None)
            tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, path)
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


def _entry(chat_id, message_id) -> dict:
    if message_id is None or chat_id is None:
        return {}
    entry = _load(_store_path()).get(f"{chat_id}:{message_id}")
    return entry if isinstance(entry, dict) else {}


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    return _entry(chat_id, message_id).get("t") or None


def lookup_media(chat_id, message_id) -> list[tuple[str, str]]:
    """Return stored ``(path, mime)`` pairs whose file still exists (attachments may be temp files)."""
    pairs = _entry(chat_id, message_id).get("m") or []
    return [(p, mt) for p, mt in pairs if isinstance(p, str) and os.path.isfile(p)]
