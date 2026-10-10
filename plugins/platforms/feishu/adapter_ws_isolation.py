"""Multiplex isolation for the lark_oapi WebSocket client (#73779, #135559).

``lark_oapi.ws.client`` keeps the asyncio loop used by ``Client.start()`` and every coroutine
it spawns in a *module-level global* (``loop``), and Hermes also monkey-patches
``websockets.connect`` on the shared ``websockets`` module to inject per-adapter ping settings.
In multiplex mode every profile runs its own WS client on a dedicated thread, so the N threads
overwrite each other's module globals (last-write-wins): a client ends up scheduling tasks on a
sibling profile's loop ("Future attached to a different loop" crashes) or binds to the wrong
loop at construction time and goes deaf from the start. The fix installs process-wide,
thread-dispatching shims exactly once: * ``ws_client_module.loop`` becomes a proxy that forwards
every attribute access to the loop registered by the *current thread*. All SDK reads of the
global happen on the thread that owns the loop (``start()`` blocks in ``run_until_complete`` and
every ``create_task`` callback runs on the loop's own thread), so each profile transparently
sees its own loop. Threads that never registered one (single-profile installs, CLI) fall back to
the SDK's original module loop. * ``websockets.connect`` becomes a single dispatcher that merges
the per-thread ping overrides registered by the calling profile, so profiles no longer race over
the global patch or restore each other's hooks while a sibling is still connected.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_WS_ISOLATION_LOCK = threading.Lock()
_WS_ISOLATION_INSTALLED = False

# The flag above is per *module namespace*: when this adapter is imported under more than one
# module name (plugin-loader aliases), each copy has its own flag and its own thread-local, so
# every copy installed another wrapper layer on the shared lark module while a worker registered
# its adapter only into its own copy's thread-local — the outermost wrapper then read
# ``adapter=None`` and misclassified a deliberate disconnect (CLOSE 1000) as a dead receive loop
# (#135559). The install marker and the per-WS-thread state below therefore live on the
# ``lark_oapi.ws.client`` module object itself (unique in sys.modules, shared by every copy), and
# the wrappers read them live so all layers converge on the same state even if two copies race
# the very first install.
_LARK_WS_STATE_ATTR = "_hermes_feishu_ws_isolation_state"
_LARK_WS_INSTALLED_ATTR = "_hermes_feishu_ws_isolation_installed"


def _lark_ws_state(ws_client_module: Any) -> threading.local:
    """The process-wide per-WS-thread state (.loop/.connect_kwargs/.on_link_up/.adapter),
    anchored on the lark module so every adapter-namespace copy shares one object."""
    state = getattr(ws_client_module, _LARK_WS_STATE_ATTR, None)
    if state is None:
        state = threading.local()
        setattr(ws_client_module, _LARK_WS_STATE_ATTR, state)
    return state


class _ThreadLocalLoopProxy:
    """Forwards attribute access to the current thread's registered loop."""

    def __init__(self, fallback: Any, ws_client_module: Any) -> None:
        self._fallback = fallback
        self._ws_client_module = ws_client_module

    def _target(self) -> Any:
        return getattr(_lark_ws_state(self._ws_client_module), "loop", None) or self._fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ThreadLocalLoopProxy target={self._target()!r}>"


def _install_lark_ws_isolation(ws_client_module: Any) -> None:
    """Install the thread-dispatching shims once per process (idempotent)."""
    global _WS_ISOLATION_INSTALLED
    with _WS_ISOLATION_LOCK:
        if _WS_ISOLATION_INSTALLED:
            return
        if getattr(ws_client_module, _LARK_WS_INSTALLED_ATTR, False):
            # Another module-namespace copy of this adapter already installed the shims. The
            # wrappers read the lark-anchored shared state live, so this copy's worker
            # registrations are visible without stacking another wrapper layer (#135559).
            _WS_ISOLATION_INSTALLED = True
            return
        ws_client_module.loop = _ThreadLocalLoopProxy(ws_client_module.loop, ws_client_module)
        real_connect = ws_client_module.websockets.connect

        def _dispatch_connect(*args: Any, **kwargs: Any) -> Any:
            overrides = getattr(_lark_ws_state(ws_client_module), "connect_kwargs", None) or {}
            for key, value in overrides.items():
                kwargs.setdefault(key, value)
            return real_connect(*args, **kwargs)

        # Keep inspect.signature(websockets.connect) honest — the SDK probes it for ``proxy`` support.
        _dispatch_connect.__wrapped__ = real_connect
        _dispatch_connect.__name__ = getattr(real_connect, "__name__", "connect")
        ws_client_module.websockets.connect = _dispatch_connect

        original_receive_loop = ws_client_module.Client._receive_message_loop

        async def _receive_message_loop_exit_notify(self: Any) -> None:
            # The SDK schedules this coroutine right after the websocket handshake succeeded, so its
            # entry is the only in-thread proof that a (re)built link is actually up.
            on_link_up = getattr(_lark_ws_state(ws_client_module), "on_link_up", None)
            if on_link_up is not None:
                on_link_up()
            try:
                await original_receive_loop(self)
            except Exception:
                # ``Client.start()`` parks in ``run_until_complete(_select())``, which only returns
                # when this worker loop stops, and the receive loop runs as a bare ``create_task``
                # whose exception nobody retrieves — so every unrecoverable exit (reconnect ladder
                # disabled, or its ``ClientException``/``ServerUnreachableException`` re-raise)
                # left a deaf-but-ESTABLISHED socket whose executor future never completed and the
                # supervisor never rebuilt (#113662). Log the root cause here and stop the loop so
                # ``start()`` raises, the future completes and ``_supervise_websocket_thread`` fires.
                # A *normal* return means the SDK's own ladder already reconnected (it scheduled a
                # fresh receive loop) and must NOT stop the loop. Deliberate disconnects nil
                # ``_ws_client`` first, so the supervisor exits without restarting.
                adapter = getattr(_lark_ws_state(ws_client_module), "adapter", None)
                if adapter is None or getattr(adapter, "_running", True):
                    logger.exception(
                        "[Feishu] lark WS receive loop died; stopping the worker "
                        "loop so the supervisor can rebuild"
                    )
                else:
                    # ``disconnect()`` sent the CLOSE frame itself: the loop ending here is expected.
                    logger.debug("[Feishu] lark WS receive loop ended during disconnect", exc_info=True)
                asyncio.get_running_loop().stop()

        ws_client_module.Client._receive_message_loop = _receive_message_loop_exit_notify
        setattr(ws_client_module, _LARK_WS_INSTALLED_ATTR, True)
        _WS_ISOLATION_INSTALLED = True


def _run_official_feishu_ws_client(ws_client: Any, adapter: Any) -> None:
    """Run the official Lark WS client in its own thread-local loop (see module notes)."""
    import lark_oapi.ws.client as ws_client_module

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    adapter._ws_thread_loop = loop
    original_configure = getattr(ws_client, "_configure", None)

    def _apply_runtime_ws_overrides() -> None:
        try:
            ws_client._reconnect_nonce = adapter._ws_reconnect_nonce
            ws_client._reconnect_interval = adapter._ws_reconnect_interval
            if adapter._ws_ping_interval is not None:
                ws_client._ping_interval = adapter._ws_ping_interval
            # SDK observer (lark-oapi ``Client.on_reconnecting``, fired first thing in ``_reconnect()``):
            # on the live link ``_auto_reconnect`` is on, so the ladder runs *inside* the receive loop
            # and the thread never dies — without this the supervisor's ``retrying`` is never published.
            ws_client.on_reconnecting = _on_reconnecting
        except Exception:
            logger.debug("[Feishu] Failed to apply websocket runtime overrides", exc_info=True)

    connect_overrides = {
        key: value
        for key, value in (("ping_interval", adapter._ws_ping_interval), ("ping_timeout", adapter._ws_ping_timeout))
        if value is not None
    }
    adapter_loop = adapter._loop

    def _on_reconnecting() -> None:
        if adapter_loop is not None and not adapter_loop.is_closed():
            adapter_loop.call_soon_threadsafe(adapter._ws_link_retrying, ws_client)

    def _on_link_up() -> None:
        # Fired on the WS thread when the SDK scheduled a receive loop (handshake done); hop to the
        # adapter loop so the ``connected`` re-stamp after a supervisor rebuild runs where the adapter's
        # state lives.
        if adapter_loop is not None and not adapter_loop.is_closed():
            adapter_loop.call_soon_threadsafe(adapter._ws_link_up, ws_client)

    _install_lark_ws_isolation(ws_client_module)
    ws_state = _lark_ws_state(ws_client_module)
    ws_state.loop = loop
    ws_state.connect_kwargs = connect_overrides
    ws_state.on_link_up = _on_link_up
    ws_state.adapter = adapter

    def _configure_with_overrides(conf: Any) -> Any:
        if original_configure is None:
            raise RuntimeError("Feishu _configure_with_overrides called but original_configure is None")
        result = original_configure(conf)
        _apply_runtime_ws_overrides()
        return result

    if original_configure is not None:
        ws_client._configure = _configure_with_overrides
    _apply_runtime_ws_overrides()
    try:
        ws_client.start()
    except Exception:
        pass
    finally:
        ws_state.loop = None
        ws_state.connect_kwargs = None
        ws_state.on_link_up = None
        ws_state.adapter = None
        if original_configure is not None:
            ws_client._configure = original_configure
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        for closer in (loop.stop, loop.close):
            try:
                closer()
            except Exception:
                pass
        adapter._ws_thread_loop = None
