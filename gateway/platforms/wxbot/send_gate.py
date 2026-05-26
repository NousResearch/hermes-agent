"""
Thread-safe send gate for iLink Bot sendmessage API.

Controls the rate at which outbound messages are sent to the iLink API,
preventing "one-way burst" behavior that triggers ret=-2 rate limiting.

State machine:

```
                    ┌────────────────────────────────────────┐
                    │                                        │
                    ▼                                        │
            ┌────────────┐   NOTIFY rmn<0 ┌──────────────┐  │
  ┌────────→│  NOTIFY    │───────────────→│ RATE_LIMITED  │──┘
  │         │  (rmn=N)  │                └──────┬───────┘
  │         └─────┬──────┘                       │
  │               │                              │
  │               │ receive msg          receive msg (→ INTERACTIVE)
  │               ▼                              ▼
  │         ┌──────────────┐              ┌──────────────┐
  │         │ INTERACTIVE  │◄─────────────│ RATE_LIMITED  │
  │         └──────┬───────┘              │  + 50min exp │
  │                │                      └──────────────┘
  │                │ send → NOTIFY(rmn=5)
  └────────────────┘

NOTIFY mode: one-way send budget (5 messages). After each send, decrement
  the budget. When budget is exhausted → RATE_LIMITED. Resets to 5 whenever
  the mode transitions TO NOTIFY (from INTERACTIVE or RATE_LIMITED).

Usage:
    gate = SendGate()
    gate.on_user_message()          # inbound → INTERACTIVE
    ok, err = gate.try_acquire()    # request send permission
    if ok:
        ...  # do the send
        gate.on_send_done()         # after successful send
    else:
        ...  # log and abort
"""

import os
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple


class SendMode(Enum):
    """Current send-gate mode."""
    NOTIFY = 0        # Notify mode: N sends budget, exhaust → RATE_LIMITED
    INTERACTIVE = 1   # Interactive mode: user-engaged, allows send, resets to NOTIFY
    RATE_LIMITED = 2  # Rate-limited mode: locked for a 50-minute window


# Maximum window for one-way burst lockout (seconds).
# Empirical value from two rate-limit events:
#   May 15: 47 min (03:28→04:15),  May 17: ~52 min (22:08→23:00)
RATE_LIMIT_WINDOW_SECONDS = 3000  # 50 minutes

# Sends allowed in NOTIFY mode before entering RATE_LIMITED.
# Can be overridden via WEIXIN_NOTIFY_SEND_LIMIT env var.
# Observed: OpenAI's openclaw uses 10; we default to 9 for a conservative buffer.
NOTIFY_SEND_LIMIT = int(os.getenv("WEIXIN_NOTIFY_SEND_LIMIT", "9"))


class SendGate:
    """Thread-safe gate controlling iLink sendmessage permissions.

    All public methods are safe to call from any thread without additional
    locking by the caller.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = SendMode.NOTIFY
        self._rate_limit_start: Optional[datetime] = None
        self._user_msg_counter = 0     # incremented on each on_user_message()
        self._acquire_msg_counter = 0  # snapshot taken at try_acquire()
        self._notify_remaining = NOTIFY_SEND_LIMIT

    # ── public API ──────────────────────────────────────────────────────

    def on_user_message(self) -> None:
        """Report an inbound user message.

        Transitions to *INTERACTIVE* from NOTIFY/RATE_LIMITED — user
        input reopens the gate for interactive replies.
        """
        with self._lock:
            self._mode = SendMode.INTERACTIVE
            self._user_msg_counter += 1

    def try_acquire(self) -> Tuple[bool, Optional[str]]:
        """Try to obtain permission for one outbound send.

        Returns:
            (True, None)          — send is allowed.
            (False, error_msg)   — send is blocked.
        """
        with self._lock:
            now = datetime.now()

            if self._mode == SendMode.INTERACTIVE:
                self._acquire_msg_counter = self._user_msg_counter
                return (True, None)

            if self._mode == SendMode.NOTIFY:
                self._acquire_msg_counter = self._user_msg_counter
                return (True, None)

            # ── RATE_LIMITED mode ───────────────────────────────────
            if self._rate_limit_start is None:
                # Safety fallback — timestamp missing, allow.
                self._acquire_msg_counter = self._user_msg_counter
                return (True, None)

            elapsed = (now - self._rate_limit_start).total_seconds()
            if elapsed >= RATE_LIMIT_WINDOW_SECONDS:
                # Window has elapsed — allow one more send.
                self._acquire_msg_counter = self._user_msg_counter
                return (True, None)

            remaining = int(RATE_LIMIT_WINDOW_SECONDS - elapsed)
            return (False, f"iLink rate limited: ~{remaining}s remaining until next send")

    def on_send_done(self) -> None:
        """Confirm one send completed successfully.

        Advances the state machine — call **after** the API call returns
        without error.

        If a new user message arrived while this send was in flight, the
        gate stays put — the new interactive session owns the state now.
        """
        with self._lock:
            now = datetime.now()

            # Stale-send guard: if a new user message arrived after
            # try_acquire(), don't advance the state machine.  The new
            # interactive session owns the gate.
            if self._acquire_msg_counter != self._user_msg_counter:
                return

            if self._mode == SendMode.INTERACTIVE:
                # INTERACTIVE → NOTIFY  (reset budget to 5)
                self._mode = SendMode.NOTIFY
                self._notify_remaining = NOTIFY_SEND_LIMIT

            elif self._mode == SendMode.NOTIFY:
                # Decrement budget; exhaust → RATE_LIMITED
                self._notify_remaining -= 1
                if self._notify_remaining <= 0:
                    self._mode = SendMode.RATE_LIMITED
                    self._rate_limit_start = now

            elif self._mode == SendMode.RATE_LIMITED:
                # RATE_LIMITED (50 min expired) → stay RATE_LIMITED,
                # reset the window clock.
                self._rate_limit_start = now

    # ── introspection (for logging / debugging) ─────────────────────────

    @property
    def mode(self) -> SendMode:
        with self._lock:
            return self._mode

    def remaining_lockout_seconds(self) -> float:
        """Seconds until the rate-limit window expires (0 if not locked)."""
        with self._lock:
            if self._mode != SendMode.RATE_LIMITED or self._rate_limit_start is None:
                return 0.0
            elapsed = (datetime.now() - self._rate_limit_start).total_seconds()
            return max(0.0, RATE_LIMIT_WINDOW_SECONDS - elapsed)

    @property
    def notify_remaining(self) -> int:
        """Remaining sends in NOTIFY mode before entering RATE_LIMITED."""
        with self._lock:
            return self._notify_remaining
