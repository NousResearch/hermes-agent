"""Connection management: Socket.IO primary, REST polling fallback.

The transport doesn't know about message payloads — it just calls
``on_tick`` whenever something *might* have happened (a WS event fired, a
poll interval elapsed, a reconnect attempt is about to be made). The
caller is responsible for actually fetching messages in response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

try:
    import socketio  # python-socketio[asyncio_client]
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    socketio = None  # type: ignore[assignment]

from .constants import (
    AGENT_NAME_HEADER,
    AGENT_NAME_VALUE,
    DEFAULT_WS_RETRY_INITIAL_MS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)

OnTick = Callable[[], Awaitable[None]]


class Transport:
    """Owns the WS client + polling loop + reconnect scheduler.

    ``mode`` transitions: ``connecting`` → (``websocket`` | ``polling``) →
    ``shutdown``. When the WS drops it falls back to ``polling`` and a
    reconnect task spins until WS comes back, then ``_stop_polling()`` is
    called and we're back to ``websocket``.
    """

    def __init__(
        self,
        base_url: str,
        pat: str,
        poll_interval_s: float,
        ws_retry_max_s: float,
        on_tick: OnTick,
    ):
        self._base_url = base_url.rstrip("/")
        self._pat = pat
        self._poll_interval_s = poll_interval_s
        self._ws_retry_max_s = ws_retry_max_s
        self._on_tick = on_tick

        self._mode: str = "connecting"
        self._sio: Optional["socketio.AsyncClient"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._ws_reconnect_task: Optional[asyncio.Task] = None
        self._ws_retry_backoff_s: float = DEFAULT_WS_RETRY_INITIAL_MS / 1000.0

    @property
    def mode(self) -> str:
        return self._mode

    async def start(self) -> None:
        """Bring the transport up. Tries WS first; falls back to polling.

        When ``python-socketio`` is missing entirely (not just a transient
        WS failure), the adapter logs a prominent warning with the exact
        install command so operators understand the cause of the degraded
        mode — Hermes does not auto-install plugin dependencies (security
        boundary), so the user must install the dep manually.
        """
        if SOCKETIO_AVAILABLE:
            try:
                await self._connect_websocket()
                return
            except Exception as exc:
                logger.warning(
                    "carbonvoice: WS initial connect failed (%s) — using polling",
                    exc,
                )
        else:
            logger.warning(
                "carbonvoice: Carbon Voice realtime websocket support is "
                "unavailable because python-socketio is not installed. "
                "Falling back to REST polling. To enable websocket mode, "
                "install python-socketio[asyncio_client] in the Hermes venv: "
                "python -m pip install 'python-socketio[asyncio_client]>=5'"
            )
        self._mode = "polling"
        self._start_polling()
        if SOCKETIO_AVAILABLE:
            self._schedule_ws_reconnect()

    async def stop(self) -> None:
        self._mode = "shutdown"
        tasks = [
            t for t in (self._poll_task, self._ws_reconnect_task)
            if t is not None and not t.done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._poll_task = None
        self._ws_reconnect_task = None

        if self._sio is not None:
            try:
                await self._sio.disconnect()
            except Exception:
                pass
            self._sio = None

    # ── WebSocket ────────────────────────────────────────────────────────

    async def _connect_websocket(self) -> None:
        if not SOCKETIO_AVAILABLE:
            raise RuntimeError("python-socketio not installed")

        sio = socketio.AsyncClient(reconnection=False)
        self._sio = sio

        @sio.on("connect")
        async def _on_connect():  # noqa: F811
            logger.info("carbonvoice: Socket.IO connected")
            self._mode = "websocket"
            self._ws_retry_backoff_s = DEFAULT_WS_RETRY_INITIAL_MS / 1000.0
            self._stop_polling()

        async def _on_message_event(payload=None):
            if not isinstance(payload, dict):
                return
            # message:created fires before transcription completes;
            # message:updated fires once the transcript is ready.
            if payload.get("status") != "active":
                return
            try:
                await self._on_tick()
            except Exception as exc:
                logger.warning("carbonvoice: tick after WS event failed: %s", exc)

        sio.on("message:created", _on_message_event)
        sio.on("message:updated", _on_message_event)

        @sio.on("disconnect")
        async def _on_disconnect():  # noqa: F811
            if self._mode == "shutdown":
                return
            logger.warning(
                "carbonvoice: Socket.IO disconnected — falling back to polling"
            )
            self._mode = "polling"
            try:
                await self._on_tick()
            except Exception:
                pass
            self._start_polling()
            self._schedule_ws_reconnect()

        await sio.connect(
            self._base_url,
            auth={"authorization": f"Bearer {self._pat}"},
            headers={
                AGENT_NAME_HEADER: AGENT_NAME_VALUE,
                "user-agent": USER_AGENT,
            },
            transports=["websocket"],
        )

    def _schedule_ws_reconnect(self) -> None:
        if self._ws_reconnect_task and not self._ws_reconnect_task.done():
            return
        if self._mode == "shutdown" or not SOCKETIO_AVAILABLE:
            return

        async def _reconnect():
            try:
                while self._mode not in ("shutdown", "websocket"):
                    await asyncio.sleep(self._ws_retry_backoff_s)
                    if self._mode == "shutdown":
                        return
                    logger.info(
                        "carbonvoice: attempting WS reconnect (backoff %.1fs)",
                        self._ws_retry_backoff_s,
                    )
                    try:
                        await self._on_tick()
                        await self._connect_websocket()
                        return
                    except Exception as exc:
                        logger.debug("carbonvoice: WS reconnect failed: %s", exc)
                        self._ws_retry_backoff_s = min(
                            self._ws_retry_backoff_s * 2,
                            self._ws_retry_max_s,
                        )
            except asyncio.CancelledError:
                pass

        self._ws_reconnect_task = asyncio.create_task(_reconnect())

    # ── Polling ──────────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        if self._poll_task and not self._poll_task.done():
            return
        logger.info("carbonvoice: polling every %.1fs", self._poll_interval_s)

        async def _tick():
            try:
                while self._mode not in ("shutdown", "websocket"):
                    try:
                        await self._on_tick()
                    except Exception as exc:
                        logger.warning("carbonvoice: poll tick failed: %s", exc)
                    await asyncio.sleep(self._poll_interval_s)
            except asyncio.CancelledError:
                pass

        self._poll_task = asyncio.create_task(_tick())

    def _stop_polling(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            logger.info("carbonvoice: polling stopped (WS active)")
        self._poll_task = None
