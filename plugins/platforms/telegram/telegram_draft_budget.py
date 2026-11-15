"""One per-chat budget for Telegram draft frames and typing actions.

Telegram meters ``sendMessageDraft`` and ``sendChatAction`` together, per peer: at most 20 calls in
5 seconds and 40 in 30 seconds (https://core.telegram.org/api/bots/ai). A draft frame or typing
tick that would exceed either window is skipped locally instead of being sent into a FLOOD_WAIT:
a skipped frame is superseded by the next one (or the final send), a skipped typing tick by the
draft itself. The slot is claimed before the request goes out, so concurrent topics of one chat
cannot spend the same slot twice. Ported from @Rinta13795's #133842.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

_WINDOWS = ((5.0, 20), (30.0, 40))  # (seconds, max calls) per peer
_LONGEST = max(seconds for seconds, _ in _WINDOWS)


def claim_ai_action(adapter: Any, chat_id: Any) -> float:
    """Reserve one draft/typing request for ``chat_id``: ``0.0`` when claimed, else the seconds
    until a slot frees (nothing is reserved then)."""
    calls_by_chat: dict[str, deque] = adapter.__dict__.setdefault("_telegram_ai_action_calls", {})
    key = str(normalize_telegram_chat_id(chat_id))
    now = asyncio.get_running_loop().time()
    calls = calls_by_chat.setdefault(key, deque())
    while calls and now - calls[0] >= _LONGEST:
        calls.popleft()
    delay = max((calls[-limit] + seconds - now for seconds, limit in _WINDOWS if len(calls) >= limit),
                default=0.0)
    if delay > 0:
        return delay
    calls.append(now)
    for stale in [k for k, q in calls_by_chat.items() if not q or now - q[-1] >= _LONGEST]:
        del calls_by_chat[stale]  # bounded dict
    return 0.0
