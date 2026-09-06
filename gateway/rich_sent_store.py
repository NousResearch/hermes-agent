"""Local index of text we've sent via ``sendRichMessage`` (Bot API 10.1). Telegram does NOT echo a rich
message's content back in ``reply_to_message`` (``.text``/``.caption`` empty, ``.api_kwargs`` None), so
replies to rich sends arrive with no quotable text; we remember ``message_id -> text`` at send time and
look it up by ``reply_to_id`` on inbound. Best-effort and dependency-free: every operation swallows
errors and degrades to a no-op / ``None`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000


def _store_path() -> str:
    from hermes_constants import get_hermes_home  # honors the active profile override
    return os.path.join(str(get_hermes_home()), "state", "rich_sent_index.json")


def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist ``text`` for ``(chat_id, message_id)``. No-op on any failure."""
    if not text or message_id is None or chat_id is None:
        return
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _load(path)
        data[f"{chat_id}:{message_id}"] = {"t": text[:_MAX_TEXT_CHARS], "ts": int(time.time())}
        if len(data) > _MAX_ENTRIES:  # trim oldest by timestamp
            for k, _ in sorted(data.items(), key=lambda kv: kv[1].get("ts", 0))[: len(data) - _MAX_ENTRIES]:
                data.pop(k, None)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)  # atomic; tolerates concurrent writers racing
    except Exception:
        return


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    if message_id is None or chat_id is None:
        return None
    entry = _load(_store_path()).get(f"{chat_id}:{message_id}")
    return (entry.get("t") or None) if isinstance(entry, dict) else None
