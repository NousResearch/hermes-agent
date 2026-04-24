# -*- coding: utf-8 -*-
"""WebSocket lifecycle management for QQ Bot.

Decoupled from the adapter via :class:`WSCallbacks` — the adapter provides
all state-mutating callbacks at construction time so this class carries zero
hermes dependencies and is independently testable.

Architecture::

    QQWebSocket
      ├── open()              — connect to the gateway URL
      ├── start_listeners()   — create listen + heartbeat asyncio tasks
      ├── stop()              — cancel tasks and clean up
      │
      ├── _listen_loop()      — outer reconnect loop  (CC ≤ 4)
      │     └── _read_events()    — raw frame reader
      │           └── _dispatch_payload()
      │                 ├── _handle_hello()          (CC ≤ 3)
      │                 ├── _handle_dispatch()        (CC ≤ 4)
      │                 └── _handle_heartbeat_ack()   (CC ≤ 1)
      │
      ├── _handle_ws_error()  — classify + act on reconnect   (CC ≤ 6)
      └── _reconnect()        — exponential backoff reconnect  (CC ≤ 3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from .constants import (
    CONNECT_TIMEOUT_SECONDS,
    DEDUP_MAX_SIZE,
    DEDUP_WINDOW_SECONDS,
    MAX_QUICK_DISCONNECT_COUNT,
    MAX_RECONNECT_ATTEMPTS,
    QUICK_DISCONNECT_THRESHOLD,
    RATE_LIMIT_DELAY,
    RECONNECT_BACKOFF,
)
from .dto import (
    CloseAction,
    DEFAULT_INTENTS,
    EventType,
    INTERACTION_EVENT_TYPES,
    MESSAGE_EVENT_TYPES,
    OPCode,
    classify_close_code,
    parse_hello,
    parse_ready,
    parse_ws_payload,
)
from .utils import build_user_agent

logger = logging.getLogger(__name__)


# ── WSCallbacks ───────────────────────────────────────────────────────

@dataclass
class WSCallbacks:
    """Dependency-injection contract between :class:`QQWebSocket` and the host.

    The host (adapter) provides all state-mutating callbacks at construction
    time.  :class:`QQWebSocket` never holds a reference to the adapter.

    :param on_message_event: Called for each inbound user message event.
    :param on_connected: Called when the connection is established / resumed.
    :param on_disconnected: Called when the connection drops.
    :param on_fatal_error: Called on non-retryable errors.
        Signature: ``(error_code: str, message: str, retryable: bool) -> None``.
    :param get_token: Async callable returning a valid access token string.
    :param get_session: Returns ``(session_id, last_seq)`` for resume.
    :param set_session: Stores ``(session_id, last_seq)`` after READY/RESUME.
    :param set_heartbeat_interval: Updates the heartbeat interval (seconds).
    :param clear_token: Invalidates the cached token (called on 4004).
    :param fail_pending: Fails all pending response futures with a reason string.
    """

    on_message_event: Callable[[str, dict], Awaitable[None]]
    on_connected: Callable[[], None]
    on_disconnected: Callable[[], None]
    on_fatal_error: Callable[[str, str, bool], None]
    get_token: Callable[[], str]
    """Synchronous — returns a valid access token string.

    Called from the WS thread loop; must not perform async I/O.
    Use ``QQApiClient.ensure_token_sync`` as the implementation.
    """
    get_session: Callable[[], Tuple[Optional[str], Optional[int]]]
    set_session: Callable[[Optional[str], Optional[int]], None]
    set_heartbeat_interval: Callable[[float], None]
    clear_token: Callable[[], None]
    fail_pending: Callable[[str], None]
    get_gateway_url: Callable[[], str]
    """Synchronous — returns the WebSocket gateway URL string.

    Called from the WS thread loop; must not perform async I/O.
    Use ``QQApiClient.get_gateway_url_sync`` as the implementation.
    """
    on_interaction_event: Optional[Callable[[str, dict], Awaitable[None]]] = None
    """Called for INTERACTION_CREATE events (button clicks).

    When ``None``, interaction events are silently discarded.
    Signature: ``(event_type: str, data: dict) -> None``.
    """

    on_ready: Optional[Callable] = None
    """Called after READY with the parsed :class:`~dto.WSReadyData`.

    Allows the host to capture bot identity (username, id) for persistence.
    Signature: ``(ready: WSReadyData) -> None``.
    """

    on_heartbeat_ack: Optional[Callable[[], None]] = None
    """Called on each Heartbeat ACK (op 11).

    The adapter can use this to flush dirty session state to disk,
    piggy-backing on the heartbeat interval instead of writing on every
    dispatch.
    """


# ── QQCloseError ──────────────────────────────────────────────────────

class QQCloseError(Exception):
    """Raised when the WebSocket closes with a specific code."""

    def __init__(self, code: Any, reason: str = "") -> None:
        self.code = int(code) if code else None
        self.reason = str(reason) if reason else ""
        super().__init__(
            f"WebSocket closed (code={self.code}, reason={self.reason})"
        )


# ── QQWebSocket ───────────────────────────────────────────────────────

class QQWebSocket:
    """WebSocket lifecycle manager for the QQ Bot gateway.

    Runs its own asyncio event loop in a dedicated daemon thread so that
    network I/O, reconnect backoff, and heartbeat never block the caller's
    event loop (e.g. the gateway main loop shared with Discord/Feishu).

    Inbound message callbacks that need to run on the *caller's* loop are
    dispatched via :func:`asyncio.run_coroutine_threadsafe` using the
    ``main_loop`` supplied to :meth:`start`.

    Architecture::

        Caller loop (main)          WS thread loop
        ──────────────────          ───────────────
        start()  ──────────────────► _run_ws_thread()
                                        open()
                                        _listen_loop()
                                        _heartbeat_loop()
        stop()   ──────────────────► _stop_async()  (via run_coroutine_threadsafe)

        on_message_event ◄──────────── run_coroutine_threadsafe(main_loop)
        on_interaction_event ◄───────── run_coroutine_threadsafe(main_loop)

    All other :class:`WSCallbacks` callbacks (sync) are called directly
    from the WS thread; they must be thread-safe (update in-memory state
    only, no asyncio operations).

    Usage::

        ws = QQWebSocket(callbacks=WSCallbacks(...), log_tag="QQBot:12345")
        main_loop = asyncio.get_running_loop()
        ws.start(gateway_url, aio_session, main_loop)
        # ... later ...
        await ws.async_stop()
    """

    def __init__(self, callbacks: WSCallbacks, log_tag: str = "QQBot") -> None:
        self._cb = callbacks
        self._log_tag = log_tag

        self._ws: Any = None            # aiohttp.ClientWebSocketResponse
        self._session: Any = None       # aiohttp.ClientSession
        self._running = False
        self._stop_requested = False   # Set True on fatal errors / user stop

        self._heartbeat_interval: float = 30.0
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Dedicated thread + loop
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Message deduplication: msg_id → received_at timestamp.
        self._seen_msg_ids: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public lifecycle API (called from the main/caller loop)
    # ------------------------------------------------------------------

    def start(
        self,
        gateway_url: str,
        main_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Start the WebSocket in a dedicated daemon thread.

        :param gateway_url: WebSocket URL.
        :param main_loop: The caller's event loop — used to dispatch
            ``on_message_event`` and ``on_interaction_event`` callbacks.
        """
        self._main_loop = main_loop
        self._ws_thread = threading.Thread(
            target=self._run_ws_thread,
            args=(gateway_url,),
            name=f"qqbot-ws-{self._log_tag}",
            daemon=True,
        )
        self._ws_thread.start()

    async def async_stop(self) -> None:
        """Stop the WebSocket thread and wait for it to exit.

        Must be called from the main loop (not from the WS thread).
        """
        ws_loop = self._ws_loop
        if ws_loop is not None and not ws_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self._stop_async(), ws_loop)
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, future.result, 10.0
                )
            except Exception:
                pass

        thread = self._ws_thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        self._ws_thread = None
        self._ws_loop = None

    # ------------------------------------------------------------------
    # Legacy API (kept for backward compat with tests / one-shot callers)
    # ------------------------------------------------------------------

    async def open(self, gateway_url: str, aio_session: Any) -> None:
        """Open a WebSocket connection (runs in the *current* loop).

        Used internally during reconnect and by tests.
        """
        if self._ws and not self._ws.closed:
            await self._ws.close()

        self._session = aio_session
        self._ws = await aio_session.ws_connect(
            gateway_url,
            headers={"User-Agent": build_user_agent()},
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
        logger.info("[%s] WebSocket connected to %s", self._log_tag, gateway_url)

    def start_listeners(self) -> None:
        """Create listen and heartbeat tasks in the *current* loop.

        Used by :meth:`_run_ws_thread` and legacy callers / tests.
        """
        self._running = True
        self._stop_requested = False
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Cancel listeners and close the WebSocket (runs in current loop)."""
        await self._stop_async()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def _run_ws_thread(self, gateway_url: str) -> None:
        """Entry point for the dedicated WS thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ws_loop = loop
        try:
            loop.run_until_complete(self._run_ws_async(gateway_url))
        except Exception as exc:
            logger.error("[%s] WS thread exited with error: %s", self._log_tag, exc, exc_info=True)
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._ws_loop = None

    async def _run_ws_async(self, gateway_url: str) -> None:
        """Open the connection and start listeners inside the WS loop.

        Creates its own aiohttp.ClientSession so nothing from the main loop
        bleeds into the WS thread loop.  Token refresh uses the synchronous
        ``get_token`` callback (httpx.Client), which is safe from any thread.
        """
        import aiohttp as _aiohttp

        async with _aiohttp.ClientSession() as aio_session:
            await self.open(gateway_url, aio_session)
            self.start_listeners()
            tasks = [t for t in (self._listen_task, self._heartbeat_task) if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _stop_async(self) -> None:
        """Cancel listeners and close the WebSocket (runs inside WS loop)."""
        self._running = False
        self._stop_requested = True
        for task in (self._listen_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._listen_task = None
        self._heartbeat_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

    # ------------------------------------------------------------------
    # Listen loop
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Outer reconnect loop — reads events and reconnects on errors."""
        backoff_idx = 0      # index into RECONNECT_BACKOFF (caps at len-1)
        attempt_count = 0    # total reconnect attempts (checked against MAX)
        quick_count = 0
        connect_time = 0.0

        while self._running:
            try:
                connect_time = time.monotonic()
                await self._read_events()
                # _read_events returned normally (ws closed cleanly or _running=False)
                backoff_idx = 0
                attempt_count = 0
                quick_count = 0
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running or self._stop_requested:
                    return

                quick_count = self._update_quick_count(connect_time, quick_count)
                if quick_count < 0:
                    return  # too many rapid disconnects

                # Reconnect loop with exponential backoff — keep retrying until
                # success or fatal error, never falling back to _read_events
                # with a broken/None ws.
                reconnected = False
                first_attempt = True
                while self._running and not reconnected:
                    reconnected = await self._handle_ws_error(exc, backoff_idx, notify_disconnect=first_attempt)
                    first_attempt = False
                    if self._stop_requested:
                        # Fatal close code reached — stop retrying.
                        return
                    if reconnected:
                        backoff_idx = 0
                        attempt_count = 0
                        quick_count = 0
                    else:
                        attempt_count += 1
                        backoff_idx = min(backoff_idx + 1, len(RECONNECT_BACKOFF) - 1)
                        if attempt_count >= MAX_RECONNECT_ATTEMPTS:
                            logger.error("[%s] Max reconnect attempts reached", self._log_tag)
                            self._cb.on_fatal_error(
                                "qq_max_reconnect",
                                "Max reconnect attempts reached",
                                True,
                            )
                            return
                        # Use exc=RuntimeError on subsequent retries so
                        # _handle_ws_error skips close-code analysis.
                        exc = RuntimeError("reconnect retry")

    def _update_quick_count(self, connect_time: float, count: int) -> int:
        """Increment quick-disconnect counter; return -1 if fatal threshold hit."""
        duration = time.monotonic() - connect_time
        if connect_time <= 0 or duration >= QUICK_DISCONNECT_THRESHOLD:
            return 0
        count += 1
        logger.info(
            "[%s] Quick disconnect (%.1fs), count: %d",
            self._log_tag, duration, count,
        )
        if count >= MAX_QUICK_DISCONNECT_COUNT:
            logger.error(
                "[%s] Too many quick disconnects — check bot permissions on QQ Open Platform",
                self._log_tag,
            )
            self._stop_requested = True
            self._cb.on_fatal_error(
                "qq_quick_disconnect",
                "Too many quick disconnects — check bot permissions",
                True,
            )
            return -1
        return count

    async def _handle_ws_error(self, exc: Exception, backoff_idx: int, notify_disconnect: bool = True) -> bool:
        """Classify the error and attempt reconnection. Returns True on success."""
        if notify_disconnect:
            self._cb.on_disconnected()
            self._cb.fail_pending("Connection interrupted")

        if isinstance(exc, QQCloseError):
            logger.warning(
                "[%s] WebSocket closed: code=%s reason=%s",
                self._log_tag, exc.code, exc.reason,
            )
            if not await self._apply_close_action(exc.code, backoff_idx):
                return False
        else:
            logger.warning("[%s] WebSocket error: %s", self._log_tag, exc)

        return await self._reconnect(backoff_idx)

    async def _apply_close_action(self, code: Optional[int], backoff_idx: int) -> bool:
        """Apply the strategy for *code*. Returns False if should stop entirely."""
        action = classify_close_code(code)

        if action == CloseAction.STOP:
            desc = {
                4914: "offline/sandbox-only",
                4915: "banned",
                4013: "invalid intent",
                4014: "intent not authorized",
            }.get(code, f"fatal error (code={code})")
            logger.error("[%s] Bot is %s. Check QQ Open Platform.", self._log_tag, desc)
            self._stop_requested = True
            self._cb.on_fatal_error(f"qq_{desc}", f"Bot is {desc}", False)
            return False

        if action == CloseAction.RATE_LIMIT:
            logger.info("[%s] Rate limited (4008), waiting %ds", self._log_tag, RATE_LIMIT_DELAY)
            if backoff_idx >= MAX_RECONNECT_ATTEMPTS:
                return False
            await asyncio.sleep(RATE_LIMIT_DELAY)

        elif action == CloseAction.IDENTIFY_ONLY:
            logger.info(
                "[%s] Session invalid (code=%s), clearing session for re-identify",
                self._log_tag, code,
            )
            self._cb.set_session(None, None)

        elif action == CloseAction.RESUME_OK:
            logger.info(
                "[%s] Connection expired (code=%s), will resume",
                self._log_tag, code,
            )
            # Keep session_id + seq intact for Resume.

        return True

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect(self, backoff_idx: int) -> bool:
        """Wait and attempt to reconnect. Returns True on success."""
        delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
        logger.info(
            "[%s] Reconnecting in %ds (attempt %d)…",
            self._log_tag, delay, backoff_idx + 1,
        )
        await asyncio.sleep(delay)

        self._heartbeat_interval = 30.0
        try:
            async def _do_reconnect() -> None:
                gateway_url = await asyncio.to_thread(self._cb.get_gateway_url)
                await self.open(gateway_url, self._session)

            await asyncio.wait_for(_do_reconnect(), timeout=CONNECT_TIMEOUT_SECONDS * 3)
            logger.info("[%s] Reconnected", self._log_tag)
            return True
        except asyncio.TimeoutError:
            logger.warning("[%s] Reconnect timed out", self._log_tag)
            return False
        except Exception as exc:
            logger.warning("[%s] Reconnect failed: %s", self._log_tag, exc)
            return False

    # ------------------------------------------------------------------
    # Event reading
    # ------------------------------------------------------------------

    async def _read_events(self) -> None:
        """Read WebSocket frames until the connection closes."""
        import aiohttp as _aiohttp

        ws = self._ws
        if not ws:
            raise RuntimeError("WebSocket not connected")

        while self._running and ws and not ws.closed:
            msg = await ws.receive()
            if msg.type == _aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if payload:
                    self._dispatch_payload(payload)
            elif msg.type == _aiohttp.WSMsgType.CLOSE:
                raise QQCloseError(msg.data, msg.extra)
            elif msg.type in (
                _aiohttp.WSMsgType.CLOSED,
                _aiohttp.WSMsgType.CLOSING,
                _aiohttp.WSMsgType.ERROR,
            ):
                raise RuntimeError("WebSocket closed unexpectedly")

    @staticmethod
    def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        """Parse a JSON string into a dict, returning None on failure."""
        try:
            payload = json.loads(raw)
        except Exception:
            logger.warning("[QQBot] Failed to parse JSON: %r", raw)
            return None
        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Payload dispatch (split into per-op methods for low CC)
    # ------------------------------------------------------------------

    def _dispatch_payload(self, raw: Dict[str, Any]) -> None:
        """Route inbound WebSocket payloads to op-specific handlers."""
        payload = parse_ws_payload(raw)

        # Update sequence number.
        session_id, last_seq = self._cb.get_session()
        if isinstance(payload.s, int) and (last_seq is None or payload.s > last_seq):
            self._cb.set_session(session_id, payload.s)

        if payload.op == OPCode.HELLO:
            self._handle_hello(payload.d)
        elif payload.op == OPCode.DISPATCH:
            self._handle_dispatch(payload.t, payload.d or {})
        elif payload.op == OPCode.HEARTBEAT_ACK:
            self._handle_heartbeat_ack()
        elif payload.op == OPCode.RECONNECT:
            # Server requests reconnect — close and let the listen loop
            # reconnect with Resume.
            logger.info("[%s] Server requested reconnect (op 7)", self._log_tag)
            self._close_ws_async()
        elif payload.op == OPCode.INVALID_SESSION:
            # Session is invalid — clear session and re-identify.
            # payload.d is True if the session is resumable (rare).
            resumable = payload.d is True
            if resumable:
                logger.info("[%s] Invalid session (op 9, resumable=True), will resume", self._log_tag)
            else:
                logger.info("[%s] Invalid session (op 9), clearing session for re-identify", self._log_tag)
                self._cb.set_session(None, None)
            self._close_ws_async()
        else:
            logger.debug("[%s] Unknown op: %s", self._log_tag, payload.op)

    def _handle_hello(self, data: Any) -> None:
        """Process op 10 Hello — schedule Identify or Resume."""
        hello = parse_hello(data)
        interval = hello.heartbeat_interval / 1000.0 * 0.8
        self._heartbeat_interval = interval
        self._cb.set_heartbeat_interval(interval)
        logger.debug(
            "[%s] Hello: heartbeat every %.1fs",
            self._log_tag, interval,
        )
        session_id, last_seq = self._cb.get_session()
        if session_id and last_seq is not None:
            self._create_task(self._send_resume())
        else:
            self._create_task(self._send_identify())

    def _handle_dispatch(self, event_type: str, data: dict) -> None:
        """Process op 0 Dispatch events."""
        if event_type == EventType.READY:
            ready = parse_ready(data)
            self._cb.set_session(ready.session_id, None)
            self._cb.on_connected()
            if self._cb.on_ready is not None:
                self._cb.on_ready(ready)
            bot_info = ""
            if ready.user:
                bot_info = f" bot={ready.user.username}"
            logger.info(
                "[%s] Ready, session_id=%s%s",
                self._log_tag, ready.session_id, bot_info,
            )
        elif event_type == EventType.RESUMED:
            self._cb.on_connected()
            session_id, last_seq = self._cb.get_session()
            logger.info(
                "[%s] Session resumed, session_id=%s seq=%s",
                self._log_tag, session_id, last_seq,
            )
        elif event_type in MESSAGE_EVENT_TYPES:
            msg_id = str(data.get("id", ""))
            if msg_id and self._is_duplicate(msg_id):
                logger.debug("[%s] Duplicate message id dropped: %s", self._log_tag, msg_id)
                return
            self._dispatch_to_main(self._cb.on_message_event(event_type, data))
        elif event_type in INTERACTION_EVENT_TYPES:
            if self._cb.on_interaction_event is not None:
                self._dispatch_to_main(self._cb.on_interaction_event(event_type, data))
            else:
                logger.debug("[%s] Unhandled interaction (no callback): %s", self._log_tag, event_type)
        else:
            logger.debug("[%s] Unhandled dispatch: %s", self._log_tag, event_type)

    def _is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was seen within the dedup window.

        Evicts stale entries when the cache exceeds :data:`DEDUP_MAX_SIZE`.
        """
        now = time.time()
        if len(self._seen_msg_ids) >= DEDUP_MAX_SIZE:
            self._evict_seen_ids(now)
        if msg_id in self._seen_msg_ids:
            return True
        self._seen_msg_ids[msg_id] = now
        return False

    def _evict_seen_ids(self, now: float) -> None:
        """Remove entries older than DEDUP_WINDOW_SECONDS."""
        cutoff = now - DEDUP_WINDOW_SECONDS
        self._seen_msg_ids = {
            k: v for k, v in self._seen_msg_ids.items() if v > cutoff
        }

    def _handle_heartbeat_ack(self) -> None:
        """Process op 11 Heartbeat ACK."""
        if self._cb.on_heartbeat_ack is not None:
            self._cb.on_heartbeat_ack()

    def _close_ws_async(self) -> None:
        """Schedule a graceful WebSocket close.

        Called from the synchronous ``_dispatch_payload`` when op 7 or op 9
        requires the connection to be closed.  The close triggers an exception
        in ``_read_events`` → the outer listen loop handles reconnect.
        """
        async def _do_close():
            if self._ws and not self._ws.closed:
                await self._ws.close()

        self._create_task(_do_close())

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat frames."""
        try:
            while self._running:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._ws or self._ws.closed:
                    continue
                __, last_seq = self._cb.get_session()
                try:
                    await self._ws.send_json({"op": OPCode.HEARTBEAT, "d": last_seq})
                except Exception as exc:
                    logger.debug("[%s] Heartbeat failed: %s", self._log_tag, exc)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Identify / Resume
    # ------------------------------------------------------------------

    async def _send_identify(self) -> None:
        """Send op 2 Identify."""
        token = await asyncio.to_thread(self._cb.get_token)
        payload = {
            "op": OPCode.IDENTIFY,
            "d": {
                "token": f"QQBot {token}",
                "intents": int(DEFAULT_INTENTS),
                "shard": [0, 1],
                "properties": {
                    "$os": "linux",
                    "$browser": "qqbot-sdk",
                    "$device": "qqbot-sdk",
                },
            },
        }
        await self._send_ws_json(payload, "Identify")

    async def _send_resume(self) -> None:
        """Send op 6 Resume."""
        token = await asyncio.to_thread(self._cb.get_token)
        session_id, last_seq = self._cb.get_session()
        payload = {
            "op": OPCode.RESUME,
            "d": {
                "token": f"QQBot {token}",
                "session_id": session_id,
                "seq": last_seq,
            },
        }
        success = await self._send_ws_json(payload, "Resume")
        if not success:
            # Reset session so next hello triggers Identify instead.
            self._cb.set_session(None, None)

    async def _send_ws_json(self, payload: dict, label: str) -> bool:
        """Send a JSON payload over the WebSocket. Returns True on success."""
        if not self._ws or self._ws.closed:
            logger.warning(
                "[%s] Cannot send %s: WebSocket not connected",
                self._log_tag, label,
            )
            return False
        try:
            await self._ws.send_json(payload)
            logger.info("[%s] %s sent", self._log_tag, label)
            return True
        except Exception as exc:
            logger.error("[%s] Failed to send %s: %s", self._log_tag, label, exc)
            return False

    # ------------------------------------------------------------------
    # Task helper
    # ------------------------------------------------------------------

    @staticmethod
    def _create_task(coro: Awaitable) -> Optional[asyncio.Task]:
        """Schedule a coroutine, silently skipping if no event loop is running."""
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()  # type: ignore[attr-defined]
            return None

    def _dispatch_to_main(self, coro: Awaitable) -> None:
        """Dispatch a coroutine to the main (caller) event loop.

        Used for callbacks that belong to the gateway framework
        (``on_message_event``, ``on_interaction_event``) — they must run on
        the main loop, not the WS-thread loop.

        Falls back to scheduling on the current loop when no main loop has
        been set (e.g. in tests or legacy single-loop usage).
        """
        main_loop = self._main_loop
        if main_loop is not None and not main_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(coro, main_loop)  # type: ignore[arg-type]
                return
            except RuntimeError:
                # Main loop may have just closed (race with shutdown).
                pass
        self._create_task(coro)
