"""Post-delivery callbacks: one deferred slot per session key, fired after the turn's reply.

An adapter's own message task fires its slot once the reply is sent
(``BasePlatformAdapter._fire_post_delivery_callback``). A turn the session authority executes
(local Desktop/TUI/CLI sessions, receipt-only ingress such as webhooks) has no such task, so a
registration made while such a turn runs is also recorded on the turn's execution context, and
the authority's drain releases it after delivering the reply (``gateway.session_ingress``).
Without that, every authority-run session kept its last turn's callback (and the turn context
it closes over) for the life of the gateway, and held "Memory updated" notices never went out.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# (adapter, session_key, generation) for each slot a turn registered.
OwedSlot = tuple[Any, str, Optional[int]]


def split_post_delivery_entry(entry: Any) -> tuple[Optional[int], Any]:
    """``(generation, callback)`` from a post-delivery slot; legacy bare callbacks have no
    generation."""
    return entry if isinstance(entry, tuple) and len(entry) == 2 else (None, entry)


_CLOSED = "post_delivery_closed"


def _turn_closed() -> bool:
    """The authority-run turn this code belongs to already ended (an abandoned executor thread
    registering late): nothing would ever release the slot, so it must not be registered."""
    from gateway.session_results import execution_result

    return bool((execution_result.get() or {}).get(_CLOSED))


def _note_owed_slot(adapter: Any, session_key: str, generation: Optional[int]) -> None:
    """Record the slot on the executing admission's result context, when there is one."""
    from gateway.session_results import execution_result

    captured = execution_result.get()
    if captured is not None:
        captured.setdefault("post_delivery", {})[(id(adapter), session_key)] = (adapter, session_key, generation)


def close_owed_slots(captured: dict) -> list[OwedSlot]:
    """End the turn behind *captured*: return the slots it registered; later registrations from
    that turn are refused."""
    captured[_CLOSED] = True
    return list((captured.get("post_delivery") or {}).values())


def drop_slots(slots: Iterable[OwedSlot]) -> None:
    """Pop *slots* from their adapters without running them."""
    for adapter, session_key, generation in slots:
        adapter.pop_post_delivery_callback(session_key, generation=generation)


async def release_owed_slots(slots: Iterable[OwedSlot], *, fire: bool, timeout: float) -> None:
    """Pop each slot (only the generation that registered it) and, when *fire*, run it like the
    adapter's own task would: bounded, errors logged and swallowed."""
    for adapter, session_key, generation in slots:
        callback = adapter.pop_post_delivery_callback(session_key, generation=generation)
        if not fire or not callable(callback):
            continue
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            result = callback()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=timeout)


class PostDeliveryMixin:
    """The per-adapter slot registry (``self._post_delivery_callbacks``, set by ``__init__``)."""

    _post_delivery_callbacks: dict[str, Any]

    def register_post_delivery_callback(
        self, session_key: str, callback: Callable, *, generation: int | None = None) -> None:
        """Register a deferred callback to fire after the main response. Same-key registrations are
        chained (both fire in order, per-callback exception isolation) so independent features
        coexist; ``generation`` ties it to a gateway run — stale generations never overwrite a
        fresher slot."""
        if not session_key or not callable(callback) or _turn_closed():
            return
        existing = self._post_delivery_callbacks.get(session_key)
        if existing is not None:
            existing_gen, existing_cb = split_post_delivery_entry(existing)
            if existing_gen is not None and generation is not None and int(generation) < int(existing_gen):
                return
            # Same-or-newer generation: chain so both fire in registration order.
            if callable(existing_cb) and (
                existing_gen is None or generation is None or int(existing_gen) == int(generation)):
                callback = self._chain_callbacks(existing_cb, callback)
        self._post_delivery_callbacks[session_key] = (
            callback if generation is None else (int(generation), callback))
        _note_owed_slot(self, session_key, None if generation is None else int(generation))

    @staticmethod
    def _chain_callbacks(*callbacks: Callable) -> Callable[[], Awaitable[None]]:
        """Async wrapper running ``callbacks`` in order with per-callback exception isolation;
        async so coroutines returned by async hooks are awaited, not dropped."""
        async def _chained() -> None:
            for _cb in callbacks:
                try:
                    _result = _cb()
                    if inspect.isawaitable(_result):
                        await _result
                except Exception:
                    logger.debug("Post-delivery callback failed", exc_info=True)
        return _chained

    def pop_post_delivery_callback(
        self, session_key: str, *, generation: int | None = None) -> Callable | None:
        """Pop a deferred callback, optionally requiring generation ownership."""
        entry = self._post_delivery_callbacks.get(session_key) if session_key else None
        if entry is None:
            return None
        entry_generation, callback = split_post_delivery_entry(entry)
        if generation is not None and (entry_generation is None or int(entry_generation) != int(generation)):
            return None
        self._post_delivery_callbacks.pop(session_key, None)
        return callback if callable(callback) else None
