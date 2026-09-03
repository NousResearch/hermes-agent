"""Gateway-side dispatch of plugin-triggered session wakes.

``GatewayRunner`` owns the loop, the session store, the authorization check, and
the adapter table; this module owns what to *do* with them when a plugin asks to
wake one exact stored session. ``gateway/run.py`` keeps only a two-line
delegation seam per entry point so injection authority does not regrow inside
the runner (`sharded, never reverted`).

The typed outcome vocabulary lives in :mod:`hermes_cli.plugin_injection`, which
is also what plugins import -- one result algebra, not two.

Scope
-----
This is the *exact-session wake* path: the caller already holds a stored
``session_key`` and the route is read from the session, never from the caller.
It reports on the ``ctx.inject_message(session_key=...)`` seam introduced by
#84929 (preserving the earlier #64436 lineage) and adds nothing to addressing.
Profile/platform/chat *routing* injection, durable cross-gateway relay, and
qualified-identity control-plane work are deliberately out of scope here.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
from typing import Any, Optional, Union

from agent.async_utils import safe_schedule_threadsafe
from hermes_cli.plugin_injection import (
    GatewayInjectionHandle,
    GatewayInjectionResult,
    InjectionDelivery,
)

logger = logging.getLogger(__name__)


def schedule_injection(
    runner: Any,
    *,
    session_key: str,
    content: str,
    plugin_id: str,
    await_dispatch: bool = False,
    correlation_id: Optional[str] = None,
) -> Union[bool, GatewayInjectionHandle]:
    """Schedule a plugin-triggered turn on the live gateway loop.

    Returns ``True``/``False`` for the scheduling outcome by default. With
    ``await_dispatch=True`` it returns a :class:`GatewayInjectionHandle` that
    resolves to what dispatch itself decided, so a caller can tell adoption apart
    from a silent drop.
    """

    def _refused(reason: str) -> Union[bool, GatewayInjectionHandle]:
        if not await_dispatch:
            return False
        return GatewayInjectionHandle.resolved(
            GatewayInjectionResult(
                False,
                reason,
                session_key=session_key,
                correlation_id=correlation_id,
            )
        )

    loop = getattr(runner, "_gateway_loop", None)
    if not getattr(runner, "_running", False) or loop is None or loop.is_closed():
        return _refused("gateway_draining")

    # Shared with the handle so a cancellation or crash can be told apart from a
    # provable refusal once the adapter holds the event.
    delivery = InjectionDelivery()

    # Through the runner's seam, not straight at ``dispatch_injection``: the
    # runner stays the single entry point that owns this coroutine.
    coro = runner._dispatch_plugin_message_injection(
        session_key=session_key,
        content=content,
        plugin_id=plugin_id,
        correlation_id=correlation_id,
        delivery=delivery,
    )
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if current_loop is loop:
        try:
            future = loop.create_task(coro)
        except Exception:
            coro.close()
            logger.warning("Plugin message injection scheduling failed", exc_info=True)
            return _refused("not_scheduled")
        runner._background_tasks.add(future)
        future.add_done_callback(runner._background_tasks.discard)
    else:
        future = safe_schedule_threadsafe(
            coro,
            loop,
            logger=logger,
            log_message="Plugin message injection scheduling failed",
            log_level=logging.WARNING,
        )
        if future is None:
            return _refused("not_scheduled")

    def _log_result(completed) -> None:
        try:
            result = completed.result()
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            return
        except Exception:
            logger.warning(
                "Plugin message injection failed: plugin=%s session=%s",
                plugin_id,
                session_key,
                exc_info=True,
            )
            return
        if not result:
            logger.warning(
                "Plugin message injection was not routed: plugin=%s session=%s "
                "reason=%s",
                plugin_id,
                session_key,
                getattr(result, "reason", "unknown"),
            )

    future.add_done_callback(_log_result)
    if await_dispatch:
        return GatewayInjectionHandle(
            future,
            correlation_id=correlation_id,
            session_key=session_key,
            delivery=delivery,
        )
    return True


async def dispatch_injection(
    runner: Any,
    *,
    session_key: str,
    content: str,
    plugin_id: str,
    correlation_id: Optional[str] = None,
    delivery: Optional[InjectionDelivery] = None,
) -> GatewayInjectionResult:
    """Route a plugin-triggered turn through the session's live adapter.

    Returns a :class:`GatewayInjectionResult`. It is falsy for every refusal, so
    callers that only cared about the boolean still read correctly.

    ``delivery`` is the arbiter between this coroutine and a caller holding the
    handle: everything up to the adapter call is disprovable delivery, the
    adapter call itself is not.
    """
    from gateway.platforms.base import MessageEvent, MessageType

    def _refused(reason: str) -> GatewayInjectionResult:
        return GatewayInjectionResult(
            False,
            reason,
            session_key=session_key,
            correlation_id=correlation_id,
        )

    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        return _refused("gateway_draining")

    entry = await runner.async_session_store.lookup_by_session_key(session_key)
    if entry is None or entry.origin is None:
        return _refused("unknown_session")
    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        return _refused("gateway_draining")

    source = dataclasses.replace(entry.origin)
    try:
        if not runner._is_user_authorized(source, allow_adapter_delegation=False):
            logger.warning(
                "Plugin message injection denied by current gateway authorization: "
                "plugin=%s session=%s",
                plugin_id,
                session_key,
            )
            return _refused("unauthorized")
    except Exception:
        logger.warning(
            "Plugin message injection authorization check failed: plugin=%s session=%s",
            plugin_id,
            session_key,
            exc_info=True,
        )
        return _refused("unauthorized")

    adapter = runner._adapter_for_source(source)
    if adapter is None:
        return _refused("no_adapter")

    metadata = {
        "hermes_plugin_id": plugin_id,
        "hermes_plugin_injection": True,
        "gateway_session_key": session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }
    if correlation_id is not None:
        metadata["hermes_plugin_injection_id"] = correlation_id

    event = MessageEvent(
        text=content,
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata=metadata,
    )
    if delivery is not None and not delivery.enter_adapter():
        # A cancel reserved the outcome first, so non-delivery is still provable.
        return _refused("cancelled")
    await adapter.handle_message(event)
    logger.info(
        "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
        plugin_id,
        session_key,
        entry.session_id,
    )
    return GatewayInjectionResult(
        True,
        "adopted",
        session_id=entry.session_id,
        session_key=session_key,
        correlation_id=correlation_id,
    )
