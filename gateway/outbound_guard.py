"""Outbound chat_id guard — prevents cross-channel Slack misroute.

Production incident (2026-06-19 11:20:58–11:22:32 UTC):
    Inbound from C0AH3RY3DK6 (WorldArchitect) at 11:20:58.
    Orphan reply `1781868147.039389` posted to C0AJQ5M0A0Y at 11:22:27
    (5 seconds BEFORE the correct reply at 11:22:32 to C0AH3RY3DK6).
    No thread_ts on the orphan because the inbound had no parent in C0AJQ5M0A0Y.

Root cause class: the gateway's outbound path used a chat_id that was
NOT derived from the inbound that triggered the response. The chat_id
came from a stale cache, the home channel, or a previous handler's
session source.

This module provides `OutboundGuard`, a task-local context manager that
pins the active inbound chat_id for the lifetime of a handler. Any call
to `verify_send(chat_id)` either confirms the chat_id matches the active
inbound, or — if it doesn't — records a guard violation that the
regression test fails on.

The guard is opt-in: call sites that already pass the correct
`source.chat_id` are unaffected. Call sites that previously sent to the
wrong channel now produce an explicit, observable failure instead of a
silent misroute.

Architecture:
    `_active_chat_id` is a `contextvars.ContextVar` created PER INSTANCE
    in `__post_init__`. Each handler invocation holds its own
    `OutboundGuard` instance, but the *value* of the pinned chat_id is
    stored in the runtime context, which is task-local and is propagated
    through `asyncio.create_task` so that the stream-consumer task spawned
    inside the handler sees the same pin as the handler itself.

    Module-level `pin_inbound()` / `unpin_inbound()` / `verify_outbound()`
    helpers proxy to a single process-wide `_global_guard` singleton. The
    singleton's `_active_chat_id` ContextVar is the canonical pin storage.
    Production call sites (Slack adapter, delivery.py, stream_consumer)
    use these helpers so they all see the same pinned value within a
    given handler turn.
"""

from __future__ import annotations

import contextvars
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)


# Cap on retained violations per guard instance. The guard is long-lived
# (the process-wide singleton `_global_guard` lives for the entire
# gateway process), and a misbehaving handler could in theory accumulate
# thousands of `verify_send()` failures per second if it spins in a loop.
# Retaining only the most recent N keeps memory bounded while preserving
# enough diagnostic context for an operator inspecting a recent incident.
MAX_VIOLATION_HISTORY = 256


# Sentinel used to represent the "no active handler" state in the
# `_active_chat_id` ContextVar. We can't use `None` for that purpose
# because `None` is also a legitimate *pinned* value: when a handler
# fires but `source.chat_id` is missing or unparseable, `pin_inbound(None)`
# is the correct record that "this handler is active but its inbound
# chat_id is unknown" — and we must REFUSE any send through that
# handler (the original incident class includes sends that target the
# wrong channel OR no channel at all because upstream code forgot to
# thread `source.chat_id` through).
#
# Using a unique sentinel for the "truly inactive" state lets the
# guard distinguish:
#   * `_NO_ACTIVE_INBOUND` (no handler is running) → allow the send
#   * `None` (active handler, missing inbound chat_id) → refuse
#   * `"C0AH3RY3DK6"` (active handler with valid chat_id) → enforce
_NO_ACTIVE_INBOUND: object = object()


@dataclass
class OutboundGuard:
    """Tracks the active inbound chat_id and verifies outbound send alignment.

    Each `_handle_message_with_agent` invocation should call
    `self._outbound_guard.enter(chat_id=source.chat_id)` to pin the
    inbound. Any call to `verify_send(chat_id, ...)` while a chat_id
    is pinned either confirms alignment (returns True) or records a
    violation (returns False, emits a WARNING log, appends to `violations`).

    The guard is task-local via `contextvars` so concurrent inbounds in
    different asyncio tasks do not interfere with each other.
    """

    # `_active_chat_id` is intentionally NOT a class-level dataclass field.
    # Declaring `ContextVar` at class scope would create a single shared
    # ContextVar object reused by every instance — that violates the
    # dataclass mutable-default convention and conflates instance identity
    # with state. Each instance creates its own ContextVar in __post_init__
    # so the field is per-instance; the *value* the ContextVar holds is
    # task-local regardless.
    #
    # The value is `object` (not `Optional[str]`) because the guard
    # needs to distinguish THREE states, not two:
    #   * `_NO_ACTIVE_INBOUND` — no handler is running; sends are allowed
    #   * `None`              — handler is running but its inbound chat_id
    #                           is missing/unknown; sends are REFUSED
    #   * `"C..."`            — handler is running with a valid chat_id;
    #                           sends are enforced
    # Using a 2-state (Optional[str]) representation would conflate the
    # first two and silently allow sends through a handler that has no
    # idea what channel to target.
    violations: Deque[dict] = field(
        default_factory=lambda: deque(maxlen=MAX_VIOLATION_HISTORY)
    )
    _active_chat_id: Optional[contextvars.ContextVar[object]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _guard_name: str = field(default="outbound_guard", init=False, repr=False)

    def __post_init__(self) -> None:
        # Create a per-instance ContextVar. The name must be unique across
        # the process for contextvars to disambiguate, so we derive it
        # from id(self) to avoid any collision if multiple guards exist.
        # The default is the `_NO_ACTIVE_INBOUND` sentinel — "truly
        # inactive, no handler running".
        self._active_chat_id = contextvars.ContextVar(
            f"outbound_guard_active_chat_id_{id(self)}",
            default=_NO_ACTIVE_INBOUND,
        )

    def enter(self, chat_id: Optional[str]):
        """Pin `chat_id` as the active inbound for the current task.

        Returns a `ResetToken` (ContextVar token) — call `reset(token)`
        in a `finally` block to restore the previous pinned chat_id.

        Passing `None` is allowed and means "handler is active but its
        inbound chat_id is missing/unknown" — verify_send will then
        refuse any subsequent outbound as unverifiable. To mark
        "no handler is running", leave the guard untouched (the
        default state of the ContextVar is `_NO_ACTIVE_INBOUND`).
        """
        return self._active_chat_id.set(chat_id)

    def reset(self, token) -> None:
        """Restore the previous pinned chat_id (use in `finally`)."""
        self._active_chat_id.reset(token)

    @property
    def active_chat_id(self) -> Optional[str]:
        """The currently-pinned inbound chat_id, or None if either no
        handler is active OR a handler is active but its inbound
        chat_id is missing. Use `is_active` to disambiguate the two.
        """
        value = self._active_chat_id.get()
        if value is _NO_ACTIVE_INBOUND:
            return None
        return value

    @property
    def is_active(self) -> bool:
        """True iff a handler has called `enter()` for the current task.

        Distinguishes "no handler running" (False) from "handler
        running with missing/unknown chat_id" (True, with
        `active_chat_id is None`).
        """
        return self._active_chat_id.get() is not _NO_ACTIVE_INBOUND

    def verify_send(
        self,
        chat_id: Optional[str],
        *,
        operation: str = "adapter.send",
        allowed_extra_destinations: Optional[List[str]] = None,
    ) -> bool:
        """Verify that `chat_id` matches the active inbound chat_id.

        Returns True in any of:
          * No handler is active (default ContextVar state) — startup,
            shutdown, cron, and other non-handler-triggered sends are
            allowed to target the home channel.
          * `chat_id` matches the pinned inbound chat_id.
          * `chat_id` is in `allowed_extra_destinations` (opt-out for
            intentional cross-channel workflows).

        Returns False (records a violation) when:
          * A handler is active and `chat_id` is None (no destination
            to verify) — refused as unverifiable.
          * A handler is active and `chat_id` is non-None but does NOT
            match the pinned inbound — refused as misroute.

        `allowed_extra_destinations` lets specific call sites (like
        the home-channel startup notification) opt out of the check
        even when a handler is active.
        """
        pinned = self._active_chat_id.get()
        if pinned is _NO_ACTIVE_INBOUND:
            # No handler is active — this is a startup/shutdown/cron
            # send. Startup notifications and shutdown pings are
            # allowed to target the home channel because there is no
            # inbound to be misaligned with.
            return True
        if pinned is None:
            # A handler called `enter(None)` — the handler is active
            # but its inbound chat_id is missing. Refuse every outbound
            # because we cannot verify alignment.
            violation = {
                "operation": operation,
                "active_inbound_chat_id": None,
                "outbound_chat_id": chat_id,
                "reason": "handler is active but inbound chat_id is missing",
            }
            self.violations.append(violation)
            logger.warning(
                "Refusing outbound: handler is active but inbound chat_id "
                "is missing (operation=%s outbound=%s)",
                operation, chat_id,
            )
            return False
        if chat_id is None:
            # A handler is active with a valid pinned chat_id, but the
            # send has no destination. Refuse as unverifiable.
            violation = {
                "operation": operation,
                "active_inbound_chat_id": pinned,
                "outbound_chat_id": chat_id,
                "reason": "chat_id is None while inbound is pinned",
            }
            self.violations.append(violation)
            logger.warning(
                "Outbound chat_id missing while inbound is pinned: "
                "operation=%s active_inbound=%s — refusing unverifiable send",
                operation, pinned,
            )
            return False
        if str(chat_id) == str(pinned):
            return True
        if allowed_extra_destinations and str(chat_id) in {
            str(c) for c in allowed_extra_destinations
        }:
            return True
        violation = {
            "operation": operation,
            "active_inbound_chat_id": pinned,
            "outbound_chat_id": chat_id,
        }
        self.violations.append(violation)
        logger.warning(
            "Outbound chat_id misalignment detected: operation=%s "
            "active_inbound=%s outbound=%s — refusing to send to wrong channel",
            operation, pinned, chat_id,
        )
        return False

    def clear_violations(self) -> None:
        """Reset the violation list (used by tests to assert a clean run)."""
        self.violations.clear()

    @property
    def violation_count(self) -> int:
        """Number of violations currently retained (bounded by
        `MAX_VIOLATION_HISTORY`). Exposed for diagnostics and tests.
        """
        return len(self.violations)


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------
#
# Production call sites (SlackAdapter.send, delivery._deliver_to_platform,
# stream_consumer._send_*) need to verify against the same pinned chat_id
# the handler set, but they don't naturally hold a reference to the
# handler's `self._outbound_guard` instance. They import this module and
# call `verify_outbound(chat_id)` instead. The singleton's
# `_active_chat_id` ContextVar is the canonical pin storage — values are
# task-local regardless of which guard instance wrote them, so this works
# correctly as long as the handler also routes through the singleton
# (see run.py `_pin_outbound_for_handler`).

_global_guard: OutboundGuard = OutboundGuard()


def get_global_guard() -> OutboundGuard:
    """Return the process-wide OutboundGuard singleton."""
    return _global_guard


def pin_inbound(chat_id: Optional[str]):
    """Pin `chat_id` as the active inbound for the current task.

    Module-level proxy for `_global_guard.enter(chat_id)`. Returns a
    token that must be passed to `unpin_inbound(token)` in a `finally`
    block to restore the previous pin.
    """
    return _global_guard.enter(chat_id)


def unpin_inbound(token) -> None:
    """Restore the previous pinned chat_id (use in `finally`)."""
    _global_guard.reset(token)


def verify_outbound(
    chat_id: Optional[str],
    *,
    operation: str = "adapter.send",
    allowed_extra_destinations: Optional[List[str]] = None,
) -> bool:
    """Verify `chat_id` against the active inbound pin.

    Module-level proxy for `_global_guard.verify_send(chat_id)`. Returns
    True if the send is aligned (or no inbound is pinned); False if the
    chat_id is misaligned with the pinned inbound — including the case
    where `chat_id` is None while a handler is active.

    Use this from production call sites that don't hold a direct
    reference to a guard instance (SlackAdapter.send, delivery, etc.).
    """
    return _global_guard.verify_send(
        chat_id,
        operation=operation,
        allowed_extra_destinations=allowed_extra_destinations,
    )


__all__ = [
    "OutboundGuard",
    "get_global_guard",
    "pin_inbound",
    "unpin_inbound",
    "verify_outbound",
]
