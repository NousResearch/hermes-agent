"""Local index of text we've sent via ``sendRichMessage`` (Bot API 10.1).

Telegram does NOT echo a rich message's content back in ``reply_to_message``
when a user replies to it (verified: ``.text``/``.caption`` empty,
``.api_kwargs`` None). So replies to the launchd briefings / any rich send
arrive with no quotable text and the agent is blind to what was referenced.

Fix: remember ``message_id -> text`` at send time, look it up by
``reply_to_id`` on inbound. This module is the single source of truth for that
index.

Best-effort and dependency-free: every operation swallows errors and degrades
to a no-op / ``None`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000


def _store_path() -> str:
    # Resolve through get_hermes_home() so the context-local profile override
    # (set_hermes_home_override) is honored.  Reading os.environ["HERMES_HOME"]
    # directly bypassed the override and leaked every profile's rich-sent index
    # into the launch/default profile in single-process multi-profile runtimes
    # (desktop tui_gateway and the multiplexed gateway).
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return os.path.join(str(home), "state", "rich_sent_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist ``text`` for ``(chat_id, message_id)``. No-op on any failure."""
    if not text or message_id is None or chat_id is None:
        return
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                data = {}
        except (FileNotFoundError, ValueError):
            data = {}
        data[_key(chat_id, message_id)] = {
            "t": text[:_MAX_TEXT_CHARS],
            "ts": int(time.time()),
        }
        # Trim oldest by timestamp when over cap.
        if len(data) > _MAX_ENTRIES:
            for k, _ in sorted(
                data.items(), key=lambda kv: kv[1].get("ts", 0)
            )[: len(data) - _MAX_ENTRIES]:
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
    try:
        with open(_store_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data.get(_key(chat_id, message_id))
        if isinstance(entry, dict):
            return entry.get("t") or None
    except (FileNotFoundError, ValueError, AttributeError):
        return None
    return None
