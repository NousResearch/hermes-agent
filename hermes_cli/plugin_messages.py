"""``PluginContext.post_message``: one message in the current conversation, editable in place.

The conversation is the session bound in the calling context (a tool or hook running in its turn,
its slash command, a hook fired for it): the plugin never names a chat, so it cannot reach another
one. Outside any session (a timer, a plugin thread, cron, a process with no live gateway) the call
is refused with a result, never an exception. Routing and the edit fallback live in
``gateway/plugin_messages.py``; this side only hands work to the gateway loop and back.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.plugins")

_TIMEOUT_S = 30.0


def _err(code: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "error": code, **({"detail": detail} if detail else {})}


class PluginMessage:
    """Handle for one posted message. ``ok`` / ``message_id`` / ``error`` describe the last
    operation; ``edit(text)`` replaces the text in place (or posts anew where the surface cannot
    edit, and the handle then follows the new message). A refused post returns a handle whose
    edits are refused the same way. Operations stay ordered: one the gateway has not finished
    (``error == "timeout"``, ``pending`` set) is waited for before the next runs, and its outcome
    still lands on the handle, so a slow post never leaves an orphan message behind a dead handle."""

    def __init__(self, plugin_id: str, target: Any, result: dict[str, Any]):
        self._plugin_id, self._target = plugin_id, target
        self.result = result
        self.message_id: Optional[str] = result.get("message_id")
        self._pending: Optional[concurrent.futures.Future] = None

    @property
    def ok(self) -> bool:
        return bool(self.result.get("ok"))

    @property
    def error(self) -> Optional[str]:
        return self.result.get("error")

    @property
    def pending(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def __repr__(self) -> str:
        return f"<PluginMessage ok={self.ok} message_id={self.message_id!r} error={self.error!r}>"

    def _adopt(self, other: PluginMessage) -> None:
        self._target, self.result, self.message_id = other._target, other.result, other.message_id

    def _refused(self) -> Optional[dict[str, Any]]:
        if self._target is None:
            return _err(self.error or "no_conversation", self.result.get("detail", ""))
        return None

    def _settle(self, result: dict[str, Any]) -> dict[str, Any]:
        self.result = result
        if result.get("ok") and result.get("message_id"):
            self.message_id = result["message_id"]
        return result

    async def _edit(self, text: str) -> dict[str, Any]:
        refused = self._refused()
        if refused is not None:
            return refused
        from gateway import plugin_messages
        return self._settle(await plugin_messages.edit(self._target, self.message_id, str(text),
                                                       plugin_id=self._plugin_id))

    async def aedit(self, text: str) -> dict[str, Any]:
        """``edit`` for code already running on the gateway's event loop (async hooks/commands)."""
        if self.pending:
            await asyncio.wrap_future(self._pending)
        return await self._edit(text)

    def edit(self, text: str) -> dict[str, Any]:
        """Replace the message text: ``{"ok", "message_id", "edited", ["fallback"]}``. Thread-safe."""
        if self.pending:
            try:
                self._pending.result(_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return _err("timeout", "the previous operation on this message has not finished")
            except Exception:  # its own outcome is already on the handle
                logger.debug("plugin_message: previous operation failed", exc_info=True)
        refused = self._refused()
        if refused is not None:
            return refused
        return _run_on_gateway_loop(self._edit(text), self)


def _gateway_loop():
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except ImportError:
        runner = None
    loop = getattr(runner, "_gateway_loop", None)
    if runner is None or not getattr(runner, "_running", False) or loop is None or loop.is_closed():
        return None, None
    return runner, loop


def _run_on_gateway_loop(coro, handle: PluginMessage) -> dict[str, Any]:
    """Run *coro* on the gateway loop and wait up to ``_TIMEOUT_S``; past that it keeps running as
    the handle's pending operation (its result still lands on the handle)."""
    _runner, loop = _gateway_loop()
    if loop is None:
        coro.close()
        return _err("gateway_unavailable", "no live gateway is running in this process")
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        coro.close()
        return _err("on_event_loop", "called from the gateway event loop: use the async form (await ...)")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    handle._pending = future
    try:
        return future.result(_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        result = _err("timeout", f"the gateway did not answer within {_TIMEOUT_S:.0f}s; the operation "
                                 "is still running and its result will land on this handle")
        handle.result = result
        return result
    except Exception as exc:  # the plugin gets a result, never the raise
        logger.warning("plugin_message: gateway hand-off failed", exc_info=True)
        return _err("send_failed", str(exc)[:512])


_NO_SESSION = ("no session is bound in the calling context (a tool call, slash command or hook "
               "fired for a session)")


async def _post_for(ctx: Any, text: str, session_key: str, route: dict) -> PluginMessage:
    plugin_id = ctx.plugin_id
    runner, _loop = _gateway_loop()
    result: dict[str, Any]
    if not session_key:
        result = _err("no_conversation", _NO_SESSION)
    elif runner is None:
        result = _err("gateway_unavailable", "no live gateway is running in this process")
    else:
        from gateway import plugin_messages
        # The session store read is disk-bound: keep it off the gateway loop.
        target, refusal = await asyncio.to_thread(plugin_messages.resolve, runner, session_key=session_key,
                                                  route=route, plugin_home=ctx._manager.home_path, plugin_id=plugin_id)
        if target is None:
            result = refusal or _err("no_conversation")
        else:
            result = await plugin_messages.post(target, str(text), plugin_id=plugin_id)
            if result.get("ok"):
                return PluginMessage(plugin_id, target, result)
    logger.info("plugin_message plugin=%s op=post refused: %s", plugin_id, result.get("error"))
    return PluginMessage(plugin_id, None, result)


async def apost(ctx: Any, text: str) -> PluginMessage:
    """``post`` for code already running on the gateway's event loop (async hooks/commands)."""
    from gateway.plugin_messages import bound_route
    from gateway.session_context import bound_session_key
    return await _post_for(ctx, text, bound_session_key(), bound_route())


def post(ctx: Any, text: str) -> PluginMessage:
    """Post *text* to the calling session from a tool handler or hook on a worker thread. The session
    is read here, on the caller's own context, then the send runs on the gateway loop."""
    from gateway.session_context import bound_session_key
    session_key = bound_session_key()
    if not session_key:
        return PluginMessage(ctx.plugin_id, None, _err("no_conversation", _NO_SESSION))
    from gateway.plugin_messages import bound_route
    route = bound_route()
    handle = PluginMessage(ctx.plugin_id, None, _err("timeout"))

    async def _post() -> dict[str, Any]:
        handle._adopt(await _post_for(ctx, text, session_key, route))
        return handle.result

    result = _run_on_gateway_loop(_post(), handle)
    if handle._target is None and not handle.pending:
        handle.result = result
    return handle
