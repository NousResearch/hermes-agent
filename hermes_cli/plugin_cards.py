"""``PluginContext.post_card``: a card in a conversation, with no model turn.

Callable from any thread, including a timer or the plugin's own background work with no session
bound. The target is ``session_key`` or ``origin`` (the conversations ``inject_message`` reaches),
else the session bound in the calling context. Consent is the ``conversation.cards`` capability,
read from the plugin's own profile. Admission and delivery live in ``gateway/plugin_cards.py``;
this side checks consent and hands the call to the gateway loop. Never raises.

``cancelled``, ``post_failed`` and ``timeout`` are UNCERTAIN results (``certain: False``): the card
may have landed. Do not inject a fallback turn on them. Retry the same card id and unchanged payload
to obtain the durable duplicate receipt without double delivery.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import logging
import time
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.plugins")

CAPABILITY = "conversation.cards"
_TIMEOUT_S = 30.0
_POLL_S = 0.25  # how often a waiting thread checks that the gateway loop is still there


def _err(code: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "error": code, **({"certain": False} if code in {"cancelled", "post_failed", "timeout"} else {}),
            **({"detail": detail} if detail else {})}


def _target(session_key: Optional[str], origin: Optional[Mapping[str, Any]]):
    """``(session_key, origin, None)`` or a refusal."""
    if session_key and origin is not None:
        return None, None, _err("invalid_target", "pass session_key or origin, not both")
    if origin is not None and not isinstance(origin, Mapping):
        return None, None, _err("invalid_target", "origin must be a mapping in the SessionSource.to_dict() shape")
    if not session_key and origin is None:
        from gateway.session_context import bound_session_key
        session_key = bound_session_key()
        if not session_key:
            return None, None, _err("no_conversation", "no session_key or origin, and no session is bound here")
    return session_key or "", dict(origin) if origin is not None else None, None


async def _post(runner: Any, ctx: Any, card: Any, session_key: str, origin) -> dict[str, Any]:
    from gateway import plugin_cards
    try:
        return await plugin_cards.post(runner, card, plugin_id=ctx.plugin_id, plugin_home=ctx._manager.home_path,
                                       session_key=session_key, origin=origin)
    except Exception as exc:  # the plugin gets a result, never the raise
        logger.warning("plugin_card plugin=%s post failed", ctx.plugin_id, exc_info=True)
        return _err("post_failed", str(exc)[:512])


def _gateway_context(runner: Any) -> contextvars.Context:
    """The gateway's own context (``_install_plugin_message_injector``), as ``inject_message`` uses:
    the admission, and the FIFO drain it may start, never inherit the caller's turn scopes (a session
    fork's tool handler, a hook inside a turn)."""
    context = getattr(runner, "_plugin_injection_context", None)
    return contextvars.Context() if context is None else context.copy()


def _prepare(ctx: Any, session_key, origin):
    if not ctx._granted_in_own_home(CAPABILITY):
        return None, None, _err("consent_required", f"grant the {CAPABILITY} capability to this plugin")
    return _target(session_key, origin)


async def apost(ctx: Any, card: Any, *, session_key=None, origin=None) -> dict[str, Any]:
    """``post`` for code already running on the gateway's event loop."""
    key, where, refusal = _prepare(ctx, session_key, origin)
    if refusal is not None:
        return refusal
    from hermes_cli.plugin_messages import _gateway_loop
    runner, loop = _gateway_loop()
    if runner is None or loop is None:
        return _err("gateway_unavailable", "no live gateway is running in this process")
    coro = _post(runner, ctx, card, key, where)
    try:
        task = loop.create_task(coro, context=_gateway_context(runner))
    except RuntimeError:  # the loop is closing under us
        coro.close()
        return _err("gateway_unavailable", "the gateway loop closed before the card was scheduled")
    try:
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():  # the caller's own cancellation, not the gateway's
            raise
        return _err("cancelled", "the gateway stopped before answering; the card may not have landed")


def _schedule(runner: Any, loop: asyncio.AbstractEventLoop, coro) -> Optional[concurrent.futures.Future]:
    """*coro* scheduled on the gateway loop in the gateway's context, or None when the loop closed
    between the lookup and the scheduling (the coroutine is closed, so nothing is left un-awaited).
    The loop runs the scheduling callback, and so the task, in the context it was submitted from."""
    try:
        return _gateway_context(runner).run(asyncio.run_coroutine_threadsafe, coro, loop)
    except RuntimeError:
        coro.close()
        return None


def _wait(future: concurrent.futures.Future, loop: asyncio.AbstractEventLoop, coro) -> dict[str, Any]:
    """The scheduled call's result, or the documented refusal when the loop closes first (the
    scheduled task never ran: its coroutine is closed), the task is cancelled, or the wait times out."""
    deadline = time.monotonic() + _TIMEOUT_S
    while True:
        try:
            return future.result(max(0.0, min(_POLL_S, deadline - time.monotonic())))
        except concurrent.futures.TimeoutError:
            if loop.is_closed():
                future.cancel()
                if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                    coro.close()  # the loop dropped its callbacks without ever starting the task
                return _err("gateway_unavailable", "the gateway loop closed before the card was admitted")
            if time.monotonic() >= deadline:
                return _err("timeout", f"the gateway did not answer within {_TIMEOUT_S:.0f}s; the card may still land")
        except concurrent.futures.CancelledError:
            return _err("cancelled", "the gateway stopped before answering; the card may not have landed")
        except Exception as exc:  # pragma: no cover - _post answers every error with a result
            logger.warning("plugin_card post escaped its result shape", exc_info=True)
            return _err("post_failed", str(exc)[:512])


def post(ctx: Any, card: Any, *, session_key=None, origin=None) -> dict[str, Any]:
    """Post *card* from a worker or plugin thread; waits up to 30 s for the gateway's answer."""
    key, where, refusal = _prepare(ctx, session_key, origin)
    if refusal is not None:
        return refusal
    from hermes_cli.plugin_messages import _gateway_loop, _on_gateway_loop
    runner, loop = _gateway_loop()
    if runner is None or loop is None:
        return _err("gateway_unavailable", "no live gateway is running in this process")
    if _on_gateway_loop():
        return _err("on_event_loop", "called from the gateway event loop: use await ctx.apost_card(...)")
    coro = _post(runner, ctx, card, key, where)
    future = _schedule(runner, loop, coro)
    if future is None:
        return _err("gateway_unavailable", "the gateway loop closed before the card was scheduled")
    return _wait(future, loop, coro)
