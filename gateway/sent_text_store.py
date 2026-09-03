"""Local index of recently-sent outbound message text, keyed per platform chat.

Purpose: when a user replies to one of OUR earlier messages whose content
never made it into the active session history — cron deliveries
(#75131), ``/background`` task notifications, out-of-history targets
(#1594) — the inbound reply carries the replied-to message ID but no
quoted text. Platform-side hydration (spectrum API, Bot API echo) fails
exactly for those targets, because they were sent from background
contexts that never touched the conversation transcript.

Fix pattern proven upstream by ``gateway/rich_sent_store.py`` (Telegram
Bot API rich sends): remember ``(chat_id, message_id) -> text`` at SEND
time, look it up by ``reply_to_id`` on INBOUND, and let the gateway's
existing ``[Replying to: "..."]`` disambiguation prefix fire
(``gateway/run.py::_prepare_inbound_message_text``). The gateway keeps
requiring an explicit reply target — there is deliberately no
"guess the last message" fallback; an unresolvable target degrades to
today's behaviour (no injected context).

Privacy/bounds contract (mirrors rich_sent_store):

- Same-chat scoping only: lookup requires the SAME chat id; cross-chat
  leakage is impossible by construction (composite key).
- Bounded: max entry count + per-entry text cap (oldest evicted first).
- Best-effort and dependency-free: every operation swallows errors and
  degrades to a no-op / ``None``, so it can never break a send or an
  inbound dispatch.

This file is ported from @DanBennettUK's PR #96149 into #95687 so the
threaded-reply PR closes its own residual case (background/cron-sent
messages whose reply text can't be hydrated) without depending on a
separate PR landing first.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000


def _store_path() -> str:
    # Resolve via get_hermes_home() so the active profile override is honored.
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return os.path.join(str(home), "state", "sent_text_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def record(chat_id, message_id, text: Optional[str]) -> None:
    """Persist ``text`` for ``(chat_id, message_id)``. No-op on any failure."""
    if not text or not message_id or not chat_id:
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
            "t": str(text)[:_MAX_TEXT_CHARS],
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
    """Return stored text for ``(chat_id, message_id)`` or ``None``.

    Same-chat scoping is enforced by the composite key: a different
    chat's entry can never match, so nothing leaks across conversations.
    """
    if not message_id or not chat_id:
        return None
    try:
        with open(_store_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data.get(_key(chat_id, message_id))
        if isinstance(entry, dict):
            return entry.get("t") or None
    except (FileNotFoundError, ValueError, AttributeError):
        return None
    except Exception:
        return None
    return None
