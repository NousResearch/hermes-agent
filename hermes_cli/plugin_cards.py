"""``PluginContext.post_card``: a card in a conversation, with no model turn.

Callable from any thread, including a timer or the plugin's own background work with no session
bound. The target is ``session_key`` or ``origin`` (the conversations ``inject_message`` reaches),
else the session bound in the calling context. Consent is the ``conversation.cards`` capability,
read from the plugin's own profile. Admission and delivery live in ``gateway/plugin_cards.py``;
this side checks consent and hands the call to the gateway loop. Never raises.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.plugins")

CAPABILITY = "conversation.cards"
_TIMEOUT_S = 30.0


def _err(code: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "error": code, **({"detail": detail} if detail else {})}


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
    return await loop.create_task(_post(runner, ctx, card, key, where), context=_gateway_context(runner))


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
    # The loop runs the scheduling callback, and so the task, in the context it was submitted from.
    future = _gateway_context(runner).run(asyncio.run_coroutine_threadsafe, _post(runner, ctx, card, key, where), loop)
    try:
        return future.result(_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return _err("timeout", f"the gateway did not answer within {_TIMEOUT_S:.0f}s; the card may still land")
