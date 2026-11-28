"""Per-call deadline floors: a tool that declares how long one call may block
(``ToolEntry.deadline_floor``) has the executor's generic deadline raised to clear it.
Also the concurrent batch's start-order gate, whose bounded wait sits under that deadline."""

import logging
import threading

logger = logging.getLogger(__name__)

# A tool that declares a per-call budget (ToolEntry.deadline_floor) gets the generic deadline
# raised to it plus this grace, so the tool's OWN timeout is the one that fires: it returns the
# partial output and exit code it collected, where the generic guard discards both and reports
# only ``tool_timeout``.
_DECLARED_DEADLINE_GRACE_S = 5.0


def declared_deadline_floor(name: str, args: dict) -> float | None:
    """Lower bound this call's deadline must clear, from the tool's own declaration."""
    from tools.registry import registry

    floor = registry.get_deadline_floor(name, args)
    return None if floor is None else floor + _DECLARED_DEADLINE_GRACE_S


def deadline_with_floor(generic: float | None, floor: float | None) -> float | None:
    """Raise *generic* to *floor*. A disabled generic deadline (``None``) stays unbounded —
    a floor only ever lengthens a deadline, it never introduces one."""
    if generic is None or floor is None:
        return generic
    return max(generic, floor)


def batch_deadline(generic: float | None, parsed_calls) -> float | None:
    """One deadline covers a whole concurrent batch, so it has to clear the longest budget any
    member declared — otherwise one slow-but-valid call takes every sibling down with it."""
    for call in parsed_calls:
        if call.parse_error is None:
            generic = deadline_with_floor(generic, declared_deadline_floor(call.name, call.args))
    return generic


class StartOrderGate:
    """Serialize worker dispatch by submit order (prompts appear in call order); ``abandon()``
    releases every parked worker so none dispatches a tool the turn already gave up on."""

    def __init__(self, timeout: float) -> None:
        self._condition = threading.Condition()
        self._next_order = 0
        self._timeout = timeout
        self.abandoned = threading.Event()

    def abandon(self) -> None:
        self.abandoned.set()
        with self._condition:
            self._condition.notify_all()

    def begin_in_order(self, order: int, callback=None, *, tool_name: str = "") -> bool:
        """Wait for ``order``, run ``callback``, advance. Returns False if abandoned."""
        with self._condition:
            # Bounded wait so one wedged dispatch can't starve later-ordered workers; on
            # expiry proceed out of order (interleaved prompts beat starvation). ``>=`` (not
            # ``==``) releases every skipped worker at once; abandoned short-circuits.
            in_order = self._condition.wait_for(
                lambda: self._next_order >= order or self.abandoned.is_set(), timeout=self._timeout,
            )
            if self.abandoned.is_set():
                return False  # the turn already synthesized this result; don't advance
            if not in_order:
                logger.warning(
                    "start-order gate timed out for %s (order=%d next=%d); proceeding out of order",
                    tool_name or "tool", order, self._next_order,
                )
            try:
                if callback is not None:
                    callback()
            finally:
                self._next_order = max(self._next_order, order + 1)
                self._condition.notify_all()
        return True
