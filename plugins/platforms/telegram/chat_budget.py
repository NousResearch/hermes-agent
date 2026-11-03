"""One shared per-chat outbound budget for Telegram Bot API calls.

Telegram meters sendMessage, editMessageText, sendChatAction, deleteMessage
(and other chat-scoped calls) against a single envelope. Independent
send/typing throttles each sized at that ceiling sum past it.

Classification uses the chat id sign only — never getChat.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

KIND_COSMETIC = "cosmetic"
KIND_DURABLE = "durable"

WINDOW_SECS = 60.0
PRIVATE_CEILING = 60
GROUP_CEILING = 20
PRIVATE_GAP_SECS = WINDOW_SECS / PRIVATE_CEILING  # 1.0
GROUP_GAP_SECS = WINDOW_SECS / GROUP_CEILING  # 3.0


def chat_budget_key(chat_id: Any) -> Optional[str]:
    """Normalized dict key, or None when chat_id is missing/empty (fail-open)."""
    if chat_id is None:
        return None
    text = str(chat_id).strip()
    if not text:
        return None
    return str(normalize_telegram_chat_id(chat_id))


def classify_chat_rate(chat_id: Any) -> tuple[int, float]:
    """Return ``(ceiling, routine_gap)`` from id sign only.

    numeric > 0 → private 60/60s (1.0s gap); numeric < 0, zero, @username,
    or any non-numeric → group 20/60s (3.0s gap, stricter).
    """
    normalized = normalize_telegram_chat_id(chat_id)
    if isinstance(normalized, int) and normalized > 0:
        return PRIVATE_CEILING, PRIVATE_GAP_SECS
    return GROUP_CEILING, GROUP_GAP_SECS


@dataclass(frozen=True)
class BudgetPlan:
    """Result of :meth:`ChatOutboundBudget.plan`.

    ``skipped`` means fail-open (caller proceeds with no wait). ``shed`` means
    a cosmetic call must be a no-op. Otherwise the caller waits ``wait``
    seconds (0 = proceed now) and the slot is already reserved.
    """

    skipped: bool = False
    shed: bool = False
    wait: float = 0.0


def _skip() -> BudgetPlan:
    return BudgetPlan(skipped=True)


def _shed() -> BudgetPlan:
    return BudgetPlan(shed=True)


@dataclass
class _ChatState:
    next_slot: float = 0.0
    last_slot: float = 0.0
    timestamps: List[float] = field(default_factory=list)
    widened_gap: float = 0.0
    widen_until: float = 0.0


class ChatOutboundBudget:
    """Shared per-chat slot reservation. One instance per adapter."""

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self.clock: Optional[Callable[[], float]] = clock or time.monotonic
        self._states: Dict[str, _ChatState] = {}

    def _now(self) -> Optional[float]:
        clock = self.clock
        if clock is None:
            return None
        try:
            return float(clock())
        except Exception:
            return None

    def _state(self, key: str) -> _ChatState:
        state = self._states.get(key)
        if state is None:
            state = _ChatState()
            self._states[key] = state
        return state

    def _effective_gap(self, state: _ChatState, now: float, routine_gap: float) -> float:
        if state.widened_gap > 0.0 and now < state.widen_until:
            return max(routine_gap, state.widened_gap)
        return routine_gap

    def _prune(self, state: _ChatState, now: float) -> None:
        cutoff = now - WINDOW_SECS
        if state.timestamps:
            state.timestamps = [ts for ts in state.timestamps if ts > cutoff]

    def plan(self, chat_id: Any, kind: str) -> BudgetPlan:
        """Reserve the next slot or report shed/skip. Never raises to callers."""
        try:
            return self._plan(chat_id, kind)
        except Exception:
            return _skip()

    def _plan(self, chat_id: Any, kind: str) -> BudgetPlan:
        key = chat_budget_key(chat_id)
        if key is None:
            return _skip()
        now = self._now()
        if now is None:
            return _skip()
        _ceiling, routine_gap = classify_chat_rate(chat_id)
        state = self._state(key)
        self._prune(state, now)
        gap = self._effective_gap(state, now, routine_gap)
        slot = now if now >= state.next_slot else state.next_slot
        wait = slot - now
        if len(state.timestamps) >= _ceiling:
            oldest = state.timestamps[0]
            window_wait = oldest + WINDOW_SECS - now
            if window_wait > wait:
                wait = window_wait
                slot = now + wait
        if wait > 0.0 and kind == KIND_COSMETIC:
            return _shed()
        state.next_slot = slot + gap
        state.last_slot = slot
        state.timestamps.append(slot)
        return BudgetPlan(wait=wait)

    def note_retry_after(self, chat_id: Any, wait: Any) -> None:
        """Widen this chat's gap after a published retry_after. Unparseable → no-op."""
        try:
            wait_f = float(wait)
        except (TypeError, ValueError):
            return
        if wait_f != wait_f or wait_f <= 0.0:
            return
        key = chat_budget_key(chat_id)
        if key is None:
            return
        now = self._now()
        if now is None:
            return
        try:
            _ceiling, routine_gap = classify_chat_rate(chat_id)
            state = self._state(key)
            widened = max(routine_gap, wait_f)
            if now < state.widen_until:
                state.widened_gap = max(state.widened_gap, widened)
            else:
                state.widened_gap = widened
            state.widen_until = now + WINDOW_SECS
            # Do not resume the routine gap as soon as the inline RetryAfter
            # sleep ends — that snap-back is the escalation bug.
            state.next_slot = max(state.next_slot, now + wait_f + state.widened_gap)
        except Exception:
            return
