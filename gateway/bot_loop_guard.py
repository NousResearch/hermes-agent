"""Pair loop protection for bot-to-bot messaging.

Telegram Bot API 10.0 (2026-05-08) allows bots to receive messages from other
bots.  The platform ships NO loop guard: core.telegram.org/api/bots/bot-to-bot
states plainly that "Bot-to-bot communication can create infinite reply loops.
Bots using this feature must make bot-message handling terminate predictably"
and lists the required safeguards:

  * Deduplicate repeated messages.
  * Apply per-chat and per-bot rate limits.
  * Enforce maximum interaction depth and timeouts, both globally and per
    sender/receiver pair.

This module implements a sliding-window budget per (conversation, bot pair).
The pair is tracked order-independently -- A->B and B->A count as the SAME
pair -- so a two-bot ping-pong burns one shared budget instead of two.

Observed failure this guards against (2026-08-21, profiles medicina + ytmed):
132 messages exchanged in a single Telegram group before a human intervened.
TELEGRAM_ALLOW_BOTS=mentions did NOT stop it: when bot A replies to bot B, the
reply itself satisfies the "mention" test, so each turn re-armed the other bot.

Tunables (env, all optional):
  HERMES_BOT_LOOP_PROTECTION   on|off           (default on)
  HERMES_BOT_LOOP_MAX_EVENTS   int              (default 20)
  HERMES_BOT_LOOP_WINDOW_SEC   int              (default 60)
  HERMES_BOT_LOOP_COOLDOWN_SEC int              (default 60)

Defaults match the OpenClaw reference implementation (20 events / 60 s window /
60 s cooldown), which solves the same problem on Discord, Slack, Matrix,
Feishu and Google Chat.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

__all__ = ["allow_bot_event", "loop_guard_state", "reset_loop_guard"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


def _enabled() -> bool:
    raw = (os.getenv("HERMES_BOT_LOOP_PROTECTION") or "on").strip().lower()
    return raw not in {"off", "false", "0", "no", "disabled"}


_LOCK = threading.Lock()
# key -> (deque of event timestamps, cooldown_until)
_EVENTS: Dict[Tuple[str, str, str], Deque[float]] = {}
_COOLDOWN: Dict[Tuple[str, str, str], float] = {}
_LAST_SWEEP = 0.0


def _key(scope: str, conversation: str, a: str, b: str) -> Tuple[str, str, str]:
    """Order-independent pair key: A->B and B->A collapse to one bucket."""
    lo, hi = sorted([str(a or "?"), str(b or "?")])
    return (str(scope or "-"), str(conversation or "-"), f"{lo}|{hi}")


def _sweep(now: float, window: float) -> None:
    """Drop buckets untouched for 10 windows so memory stays bounded."""
    global _LAST_SWEEP
    if now - _LAST_SWEEP < window:
        return
    _LAST_SWEEP = now
    horizon = now - (window * 10)
    for k in [k for k, dq in _EVENTS.items() if not dq or dq[-1] < horizon]:
        _EVENTS.pop(k, None)
        if _COOLDOWN.get(k, 0.0) < now:
            _COOLDOWN.pop(k, None)


def allow_bot_event(
    scope: str,
    conversation: str,
    sender_bot: str,
    receiver_bot: str,
    *,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """Return (allowed, reason) for one inbound bot-authored message.

    Call ONLY for messages authored by another bot, at the moment the message
    is admitted. Human traffic must never reach this function.
    """
    if not _enabled():
        return True, "guard-disabled"

    max_events = _env_int("HERMES_BOT_LOOP_MAX_EVENTS", 20)
    window = float(_env_int("HERMES_BOT_LOOP_WINDOW_SEC", 60))
    cooldown = float(_env_int("HERMES_BOT_LOOP_COOLDOWN_SEC", 60))

    t = time.monotonic() if now is None else now
    k = _key(scope, conversation, sender_bot, receiver_bot)

    with _LOCK:
        _sweep(t, window)

        until = _COOLDOWN.get(k, 0.0)
        if until > t:
            return False, f"cooldown:{until - t:.0f}s-left"

        dq = _EVENTS.setdefault(k, deque())
        cutoff = t - window
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= max_events:
            _COOLDOWN[k] = t + cooldown
            dq.clear()
            return False, f"budget-exceeded:{max_events}/{window:.0f}s"

        dq.append(t)
        return True, f"ok:{len(dq)}/{max_events}"


def loop_guard_state() -> dict:
    """Snapshot for diagnostics."""
    t = time.monotonic()
    with _LOCK:
        return {
            "enabled": _enabled(),
            "max_events": _env_int("HERMES_BOT_LOOP_MAX_EVENTS", 20),
            "window_seconds": _env_int("HERMES_BOT_LOOP_WINDOW_SEC", 60),
            "cooldown_seconds": _env_int("HERMES_BOT_LOOP_COOLDOWN_SEC", 60),
            "tracked_pairs": len(_EVENTS),
            "pairs": {
                "|".join(k): len(dq) for k, dq in list(_EVENTS.items())[:20]
            },
            "cooling_down": {
                "|".join(k): round(v - t, 1)
                for k, v in _COOLDOWN.items()
                if v > t
            },
        }


def reset_loop_guard() -> None:
    with _LOCK:
        _EVENTS.clear()
        _COOLDOWN.clear()
