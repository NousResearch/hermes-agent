"""Dispatch-time bot budgets, adapted from 69k4xmdfm2-blip's PR #91483.

Retains the conversation-wide sliding window, not its stateful authorization
check or cooldown: elapsed time must never reset the human-bounded hop count.
State belongs to one runner and is isolated by profile/platform/chat/thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from gateway.config import Platform
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _positive_int(settings: dict, key: str, default: int) -> int:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass
class _Conversation:
    events: deque = field(default_factory=deque)
    seen: set = field(default_factory=set)
    hops: int = 0
    human_id: int = 0


class BotLoopGuard:
    """No stale-bucket eviction: forgetting a hop count reopens a slow loop.

    Only admitted bot IDs in the current chain are retained (bounded by
    max_hops). The human message watermark rejects replays from older chains.
    Restarting the runner clears state; this is not a distributed hop protocol.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._conversations = {}

    def admit(
        self, key: tuple, message_id: str, is_bot: bool, settings: dict,
        *, accept: Callable[[], bool] | None = None, check_only: bool = False,
    ) -> bool:
        """Check, synchronously retain, then charge under one lock.

        ``accept`` must not await or reenter this guard; False means no work was
        retained. Preflight checks never create or mutate conversation state.
        """
        try:
            mid = int(message_id)
        except (TypeError, ValueError):
            return not is_bot
        if mid <= 0:
            return not is_bot
        with self._lock:
            state = self._conversations.get(key)
            if state is None:
                state = _Conversation()
            if not is_bot:
                if not check_only and mid > state.human_id:
                    state.human_id = mid
                    # Delayed human updates must not erase newer accepted peers.
                    state.seen = {peer_mid for peer_mid in state.seen if peer_mid > mid}
                    state.hops = len(state.seen)
                    self._conversations[key] = state
                return True
            if mid <= state.human_id or mid in state.seen:
                return False
            max_hops = _positive_int(settings, "max_hops", 8)
            if state.hops >= max_hops:
                return False
            max_events = _positive_int(settings, "max_events", 20)
            window = _positive_int(settings, "window_seconds", 60)
            now = time.monotonic()
            if sum(timestamp > now - window for timestamp in state.events) >= max_events:
                return False
            if check_only:
                return True
            if accept is not None and not accept():
                return False
            now = time.monotonic()
            while state.events and state.events[0] <= now - window:
                state.events.popleft()
            state.events.append(now)
            state.seen.add(mid)
            state.hops += 1
            self._conversations[key] = state
            if state.hops == max_hops:
                logger.warning("Telegram bot hop limit reached for %s; waiting for an authorized human", key)
            return True


def admit_telegram_bot_turn(
    runner, event, *, accept: Callable[[], bool] | None = None, check_only: bool = False,
) -> bool:
    """Charge only at queue/dispatch acceptance, after auth + native trigger gates.

    Passive observations never enter the runner. Humans and DMs are unmetered;
    bot commands cannot reach administrative or pending-control intercepts.
    """
    source = event.source
    if source.platform != Platform.TELEGRAM:
        return True if check_only or accept is None else bool(accept())
    is_bot = getattr(source, "is_bot", False)
    if is_bot and event.get_command():
        return False
    if source.chat_type not in {"group", "forum", "channel"}:
        return True if check_only or accept is None else bool(accept())
    if not is_bot:
        # Observe mode strips source.user_id; native from_user still proves a
        # human sender. Anonymous admins/channel posts cannot re-arm a chain.
        raw = event.raw_message
        human_id = getattr(getattr(raw, "from_user", None), "id", None) if raw is not None else source.user_id
        if not human_id:
            return True
    adapter = runner._adapter_for_source(source)
    extra = adapter.config.extra if adapter is not None else {}
    settings = extra.get("bot_loop", {})
    if not isinstance(settings, dict):
        settings = {}
    guard = getattr(runner, "_telegram_bot_loop_guard", None)
    if guard is None:
        guard = BotLoopGuard()
        if not check_only:
            runner._telegram_bot_loop_guard = guard
    profile = source.profile or getattr(adapter, "_owner_profile", None) or str(get_hermes_home())
    key = (profile, source.platform.value, str(source.chat_id), str(source.thread_id or ""))
    return guard.admit(key, event.message_id, is_bot, settings, accept=accept, check_only=check_only)
