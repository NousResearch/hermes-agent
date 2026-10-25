"""Turn-scoped deferred delivery ledger (post-delivery callbacks).

Deferred callbacks (goal/loop status lines, background-review release, temporary-bubble
cleanup) belong to the TURN that registered them.  The record below is created per turn,
keyed by the turn's ADMISSION ID in ``authority.turn_deliveries``, carried on that turn's
``MessageEvent``, and consumed by whichever lane actually delivers the turn's reply: the
adapter session task, the authority drain's ``deliver_settled`` send, or the serial-reader
busy observer.

Unlike the legacy session-keyed slot on the adapter, there is no generation tag to match
and no shared dict: the record's identity *is* the ownership proof, so a callback can never
be adopted by a later turn, leaked into the next one, or refused by a ``pop`` whose
generation does not match.  ``gen=None`` stops being meaningful here -- it was only ever a
fingerprint of a lane that never installed the adapter's ``_active_sessions`` guard.

Two invariants the three failure modes this module exists for depend on:

* ``adapter_sends`` is CARRIED on the record, never inferred from ``authority.waiters``
  (which the API, Webhook, Bot, Hosted and Managed lanes also fill).  A lane that cannot
  send -- an is_api turn, a turn that produced no reply -- fires the record itself, so no
  turn's work is stranded in the registry.
* ``drop()`` is TERMINAL: a superseded turn's record is CLOSED, so the post-turn hooks that
  still run after the discard cannot re-arm deferred work on it.  Release/cleanup callbacks
  registered with ``run_on_drop=True`` still RUN at that point -- they free a queue or delete
  per-turn UI, so discarding them leaks what they release.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextvars import ContextVar
from typing import Any, Callable

logger = logging.getLogger(__name__)

_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS = 10.0


def _run_release(callback: Callable[[], Any]) -> None:
    """Run one release/cleanup callback for a dropped turn. They are sync in practice (they
    schedule their own async work); an awaitable result is scheduled on the running loop when
    there is one."""
    try:
        result = callback()
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop().create_task(result)
            except RuntimeError:
                close = getattr(result, "close", None)
                if callable(close):
                    close()
    except Exception:
        logger.debug("Post-delivery release callback failed", exc_info=True)


class TurnDelivery:
    """Deferred callbacks owned by exactly one turn.

    ``fire()`` consumes the callbacks queued at call time, awaiting each with the same
    bounded timeout and per-callback exception isolation the legacy slot used.  Calling it
    again fires callbacks registered afterwards (a queued-follow-up turn delivers twice) and
    never re-fires one already consumed.  A callback registered after the turn's last fire is
    dropped with the record instead of being fired by an unrelated later turn.

    ``drop()`` closes the record instead: later ``register()`` calls are refused, so a
    superseded turn cannot re-arm work that would surface after a newer turn's reply.
    """

    __slots__ = ("_callbacks", "_closed", "_on_drop", "adapter_sends")

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], Any]] = []
        self._on_drop: list[Callable[[], Any]] = []
        self._closed = False
        # Carried ownership bit: a native adapter lifecycle will send this turn's reply and
        # fire this record itself.  Set by the lane that knows; never inferred from
        # ``authority.waiters``.
        self.adapter_sends = False

    @property
    def closed(self) -> bool:
        return self._closed

    def register(self, callback: Callable[[], Any], *, run_on_drop: bool = False) -> None:
        if not callable(callback):
            return
        if self._closed:
            # CLOSED (superseded): reply-ordered work must never fire on this turn; a release
            # callback still has to run NOW, because no delivery will ever fire it.
            if run_on_drop:
                _run_release(callback)
            return
        self._callbacks.append(callback)
        if run_on_drop:
            self._on_drop.append(callback)

    def drop(self) -> None:
        """CLOSED drop: nothing reply-ordered may fire, but release callbacks still run."""
        if self._closed:
            return
        self._closed = True
        release, self._on_drop, self._callbacks = self._on_drop, [], []
        for callback in release:
            _run_release(callback)

    async def fire(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        # Consumed with the callbacks: a release callback must not run a second time.
        self._on_drop = []
        for callback in callbacks:
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(
                        result, timeout=_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS)
            except Exception:
                logger.debug("Post-delivery callback failed", exc_info=True)


# The record of the turn executing on this task.  Set around the handler call by the adapter
# task (``gateway.platforms.base._process_message_background``), the authority drain
# (``gateway.session_ingress.execute_admission``) and the serial-reader busy observer
# (``gateway.session_ingress.dispatch_shared_busy``); reset on exit.  A registration made
# outside any turn falls back to the adapter's legacy session-keyed slot.
current_turn_delivery: ContextVar[Any] = ContextVar("current_turn_delivery", default=None)


def turn_delivery_for_event(event: Any) -> TurnDelivery:
    """The delivery record attached to ``event``, created on first use."""
    record = getattr(event, "_turn_delivery", None)
    if record is None:
        record = TurnDelivery()
        try:
            event._turn_delivery = record
        except Exception:
            # A slotted/immutable stand-in (some tests): callers that thread the returned
            # record explicitly still work, but it cannot be shared across lanes this way.
            logger.debug("Event refused a turn delivery record", exc_info=True)
    return record


def adopt_turn_delivery(source_event: Any, target_event: Any) -> None:
    """Carry the source event's record onto a copy (``dataclasses.replace`` drops it).

    ``dataclasses.replace`` copies the declared fields only, so the record -- an instance
    ATTRIBUTE -- is lost on a ``pre_gateway_dispatch`` rewrite.  Without this the rewritten
    turn registers on the new event's record while the lifecycle that opened the turn fires
    the old one.
    """
    record = getattr(source_event, "_turn_delivery", None)
    if record is not None:
        try:
            target_event._turn_delivery = record
        except Exception:
            logger.debug("Event copy refused a turn delivery record", exc_info=True)
    # The copy drops the crash-marker handoff flag too, and that flag is what makes
    # ``admit_message`` adopt the turn's durable active-turn marker: without it the marker is
    # released before the reply is ledgered, reopening the crash window it exists to close.
    if getattr(source_event, "_turn_marker_handoff", False):
        try:
            target_event._turn_marker_handoff = True
        except Exception:
            logger.debug("Event copy refused the turn marker handoff", exc_info=True)


def drop_current_turn_delivery() -> bool:
    """CLOSED-drop the record of the turn executing on this task; False when there is none."""
    record = current_turn_delivery.get()
    if record is None:
        return False
    record.drop()
    return True


def drop_turn_delivery(event: Any = None) -> None:
    """CLOSED-drop the running turn's record (authoritative), else ``event``'s."""
    record = current_turn_delivery.get() or getattr(event, "_turn_delivery", None)
    if record is not None:
        record.drop()


async def fire_turn_delivery(event: Any) -> None:
    """Deliver ``event``'s deferred callbacks, if it carries a record."""
    record = getattr(event, "_turn_delivery", None)
    if record is not None:
        await record.fire()
