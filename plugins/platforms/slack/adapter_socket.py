"""Socket Mode lifecycle for ``SlackAdapter``: handler teardown, the watchdog and app-level liveness.

Websocket ping/pong only proves that Slack's edge holds the socket. A session Slack never registered
for event routing keeps answering pings while no event ever arrives (seen after Slack's real-time
connection disruptions: sockets reconnected mid-incident answered pings for hours, delivered nothing,
and got none of Slack's periodic refreshes until the process restarted). Slack sends ``hello`` once
it registers a session and every event is a text frame, so the watchdog restarts a session that never
got ``hello`` and renews, make-before-break, one that has been silent for too long.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

logger = logging.getLogger(__name__)


# SocketModeClient's background tasks (getattr-looked-up so an SDK rename degrades to a no-op).
_SOCKET_CLIENT_TASK_ATTRS = ("current_session_monitor", "message_processor", "message_receiver")
# Teardown wait cap: a task wedged in a network call must not hold up shutdown.
_SOCKET_TASK_CANCEL_TIMEOUT_S = 3.0
# ``close_async()`` runs under the reconnect lock; a close hung on a dead network must not strand it.
_SOCKET_CLOSE_TIMEOUT_S = 10.0
# Slack control frames (``hello``/``disconnect``) are a few hundred bytes. Larger frames are events:
# they only refresh the liveness clock and are never parsed a second time.
_SOCKET_CONTROL_FRAME_MAX = 8192


@dataclass
class _SocketLiveness:
    """App-level liveness of one Socket Mode client: the Slack frames it has seen (module docstring)."""

    last_frame: float  # monotonic time of the last Slack text frame (hello, disconnect or event)
    observable: bool = False  # False when the SDK offers no frame hook: liveness checks stay off
    registered_session: Any = None  # session whose ``hello`` arrived (strong ref, so no id reuse)
    seen_session: Any = None  # session the watchdog last observed, and when it first saw it
    seen_at: float = 0.0
    connections: Any = None  # ``num_connections`` from the latest ``hello``
    registered: asyncio.Event = field(default_factory=asyncio.Event)  # set by the first ``hello``


def _slack_socket_seconds(env_name: str, default: float) -> float:
    """Socket Mode liveness knob in seconds from ``env_name`` (``0`` disables); invalid → ``default``."""
    raw = _get_scoped_secret(env_name, "")
    if raw:
        try:
            value = float(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
        logger.warning(
            "[Slack] Invalid %s=%r (expected seconds >= 0, 0 disables); using %.0fs", env_name, raw,
            default)
    return default


async def _cancel_socket_tasks(tasks: Any) -> None:
    """Cancel Socket Mode tasks and await them (bounded); unawaited cancel still races the work."""
    live = [
        task
        for task in tasks
        if task is not None
        and callable(getattr(task, "cancel", None))
        and not (callable(getattr(task, "done", None)) and task.done())]
    for task in live:
        task.cancel()
    pending = set(live)
    if not pending:
        return
    done, still_running = await asyncio.wait(pending, timeout=_SOCKET_TASK_CANCEL_TIMEOUT_S)
    for task in done:
        if task.cancelled():
            continue
        if task.exception() is not None:  # pragma: no cover - defensive logging
            logger.debug("[Slack] Socket Mode task failed while stopping", exc_info=True)
    if still_running:  # pragma: no cover - defensive logging
        logger.warning(
            "[Slack] %d Socket Mode task(s) did not stop within %.1fs", len(still_running),
            _SOCKET_TASK_CANCEL_TIMEOUT_S)


class SlackSocketModeMixin:
    """Socket Mode lifecycle mixed into ``SlackAdapter``. It reads the adapter's ``_app``,
    ``_app_token``, ``_running``, ``_handler`` and ``_socket_mode_task``, and asks the adapter for a
    started handler through ``_new_socket_mode_handler()``."""

    # Watchdog: poll interval; reconnect after N ping_intervals of silence (Slack pings idle
    # sockets, so silence = wedged transport); grace after (re)connect for the first ping/pong.
    _socket_watchdog_interval_s = 15.0
    _socket_ping_stale_factor = 4
    _socket_first_ping_grace_s = 60.0
    # App-level liveness. A session with no ``hello`` this long after the watchdog first sees it was
    # accepted by Slack's edge but never registered, so it is restarted
    # (override: ``SLACK_SOCKET_HELLO_TIMEOUT_SECONDS``, 0 = off).
    _socket_hello_timeout_s = 60.0
    # No Slack frame at all for this long → renew make-before-break: open a second session, swap it in
    # once Slack registers it, then close the old one. Idle apps get no frames, so this bounds a silent
    # stall without a delivery gap (override: ``SLACK_SOCKET_SILENCE_RENEW_SECONDS``, 0 = off).
    _socket_silence_renew_s = 1800.0
    # After a renewal whose new session Slack never registered, keep the old one this long first.
    _socket_renew_backoff_s = 300.0

    def _init_socket_mode_state(self) -> None:
        """Socket Mode self-healing state; the monotonic start time is the grace window for the first
        ping/pong."""
        self._socket_watchdog_task: Optional[asyncio.Task] = None
        self._socket_reconnect_lock = asyncio.Lock()
        self._socket_handler_started_monotonic: Optional[float] = None
        self._socket_liveness: Optional[_SocketLiveness] = None
        self._socket_disconnected_samples = 0
        self._socket_renew_not_before = 0.0
        self._socket_liveness_unavailable_logged = False
        self._socket_hello_timeout_s = _slack_socket_seconds(
            "SLACK_SOCKET_HELLO_TIMEOUT_SECONDS", self._socket_hello_timeout_s)
        self._socket_silence_renew_s = _slack_socket_seconds(
            "SLACK_SOCKET_SILENCE_RENEW_SECONDS", self._socket_silence_renew_s)

    def _adopt_socket_handler(self, handler: Any, task: asyncio.Task, liveness: _SocketLiveness) -> None:
        """Make a started handler the live one."""
        self._handler, self._socket_mode_task, self._socket_liveness = handler, task, liveness
        self._socket_handler_started_monotonic = time.monotonic()

    def _attach_socket_liveness(self, client: Any) -> _SocketLiveness:
        """Hook the SDK receive loop so Slack text frames feed this client's liveness. Without the hook
        the checks that need it stay off; a check that can never pass would restart forever."""
        liveness = _SocketLiveness(last_frame=time.monotonic())
        listeners = getattr(client, "on_message_listeners", None)
        if isinstance(listeners, list):
            listeners.append(self._socket_frame_listener(client, liveness))
            liveness.observable = True
        elif not self._socket_liveness_unavailable_logged:
            self._socket_liveness_unavailable_logged = True
            logger.info(
                "[Slack] Socket Mode client has no on_message_listeners; only transport checks run")
        return liveness

    def _socket_frame_listener(
            self, client: Any, liveness: _SocketLiveness) -> Callable[[Any], Awaitable[None]]:
        """``on_message_listeners`` hook. It runs inside the SDK receive loop for every text frame, so it
        must never raise: an exception there counts as a receive failure and backs the loop off."""

        async def _on_frame(message: Any) -> None:
            try:
                liveness.last_frame = time.monotonic()
                data = getattr(message, "data", None)
                if not isinstance(data, str) or len(data) > _SOCKET_CONTROL_FRAME_MAX:
                    return
                if '"hello"' not in data and '"disconnect"' not in data:
                    return
                frame = json.loads(data)
                if not isinstance(frame, dict):
                    return
                if frame.get("type") == "hello":
                    self._note_socket_hello(client, liveness, frame)
                elif frame.get("type") == "disconnect":
                    self._note_socket_disconnect(frame)
            except Exception:  # pragma: no cover - bookkeeping must not break the receive loop
                logger.debug("[Slack] Socket Mode frame bookkeeping failed", exc_info=True)

        return _on_frame

    def _note_socket_hello(self, client: Any, liveness: _SocketLiveness, frame: dict) -> None:
        """Slack registered the session that delivered this ``hello``. ``num_connections`` counts every
        registered connection of the app, which exposes a second consumer or a leaked socket."""
        liveness.registered_session = getattr(client, "current_session", None)
        liveness.registered.set()
        connections = frame.get("num_connections")
        liveness.connections = connections
        # A hello on a handler that is not current is a renewal candidate: the live one should count too.
        renewing = liveness is not self._socket_liveness
        expected = 2 if renewing else 1
        if not renewing:
            debug_info = frame.get("debug_info") if isinstance(frame.get("debug_info"), dict) else {}
            logger.info(
                "[Slack] Socket Mode session registered (connections=%s, host=%s, refresh≈%ss)",
                connections, debug_info.get("host", "?"),
                debug_info.get("approximate_connection_time", "?"))
        if isinstance(connections, int) and connections > expected:
            logger.warning(
                "[Slack] Slack counts %d Socket Mode connections for this app (expected %d): Slack "
                "splits events across them, so another process using this SLACK_APP_TOKEN gets a "
                "share; stop it, or look for a leaked socket", connections, expected)
        elif renewing and connections == 1:
            logger.warning(
                "[Slack] Slack counts only the new Socket Mode connection: the session being replaced "
                "was not registered, so events were not reaching it")

    @staticmethod
    def _note_socket_disconnect(frame: dict) -> None:
        """Slack announces a reconnect; the SDK opens the new session itself."""
        reason = str(frame.get("reason") or "unknown")
        if reason == "link_disabled":
            logger.warning(
                "[Slack] Slack disconnected Socket Mode (link_disabled): Socket Mode is turned off in "
                "this app's settings at api.slack.com/apps; no events arrive until it is re-enabled")
        elif reason == "too_many_websockets":
            logger.warning(
                "[Slack] Slack disconnected Socket Mode (too_many_websockets): look for leaked sockets "
                "or another process using this app token")
        else:
            logger.info("[Slack] Slack requested a Socket Mode reconnect (reason=%s)", reason)

    async def _stop_socket_mode_handler(self) -> None:
        """Stop Socket Mode handler and task. Order matters: ``SocketModeClient.connect()`` is a
        ``while True`` retry loop that never checks ``closed``, so anything inside it when
        ``close_async()`` drops the session retries forever. Cancel every task that can reach
        ``connect()`` BEFORE closing (it rebinds task attrs on success, so a mid-close snapshot
        races a moving target).

        Everything that can reach ``connect()`` therefore has to be stopped first.
        ``monitor_current_session()`` and ``receive_messages()`` each get there on their own, and
        ``connect()`` rebinds the client's task attributes on success, so the set of live tasks changes
        across the awaits inside ``close()``. Cancelling from a snapshot taken partway through that would
        race a moving target. See slackapi/python-slack-sdk#1913.
        """
        handler, task = self._handler, self._socket_mode_task
        self._handler = self._socket_mode_task = self._socket_liveness = None
        await self._close_socket_mode_handler(handler, task)

    @staticmethod
    async def _close_socket_mode_handler(handler: Any, task: Any) -> None:
        """Stop one handler's task and client tasks first, then close it (see above for the order)."""
        client = getattr(handler, "client", None)
        await _cancel_socket_tasks(
            [task] + [getattr(client, attr, None) for attr in _SOCKET_CLIENT_TASK_ATTRS])
        if handler is not None:
            try:
                await asyncio.wait_for(handler.close_async(), timeout=_SOCKET_CLOSE_TIMEOUT_S)
            except TimeoutError:
                logger.warning(
                    "[Slack] Socket Mode handler did not close within %.0fs; continuing",
                    _SOCKET_CLOSE_TIMEOUT_S)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(
                    "[Slack] Error while closing Socket Mode handler: %s", e, exc_info=True)

    async def _socket_transport_connected(self) -> Optional[bool]:
        """Best-effort check of current Socket Mode transport state."""
        state = getattr(getattr(self._handler, "client", None), "is_connected", None)
        if state is None:
            return None
        try:
            value = state() if callable(state) else state
            if asyncio.iscoroutine(value):
                value = await value
            return bool(value)
        except Exception:  # pragma: no cover - optional client API
            logger.debug("[Slack] Could not inspect Socket Mode transport state", exc_info=True)
            return None

    def _socket_ping_pong_stale(self) -> bool:
        """No recent ping/pong on the transport. Slack pings every ``ping_interval`` even when idle,
        and a client stuck on a closed session can still report ``is_connected()``, so staleness is
        the reliable "wedged" signal. Non-numeric attrs (mocked clients) never reconnect."""
        client = getattr(self._handler, "client", None)
        if client is None:
            return False
        ping_interval = getattr(client, "ping_interval", None)
        if not isinstance(ping_interval, (int, float)) or ping_interval <= 0:
            return False
        last = getattr(client, "last_ping_pong_time", None)
        if last is None:
            # No ping yet: healthy right after (re)connect until the grace window elapses.
            started = self._socket_handler_started_monotonic
            grace = max(self._socket_first_ping_grace_s, ping_interval * 2)
            return started is not None and (time.monotonic() - started) > grace
        if not isinstance(last, (int, float)):
            return False
        return (time.time() - last) > (ping_interval * self._socket_ping_stale_factor)

    def _socket_processor_stopped(self) -> bool:
        """The SDK creates its dispatch task once per client and never restarts it. If it ends, the
        receive loop (and its pongs) carries on, so frames are read but never handled."""
        processor = getattr(getattr(self._handler, "client", None), "message_processor", None)
        return isinstance(processor, asyncio.Future) and processor.done()

    def _socket_unregistered(self) -> bool:
        """The current session got no ``hello`` within the timeout: Slack's edge holds the socket and
        answers pings, but Slack never registered it for event routing, so nothing arrives on it."""
        liveness, timeout = self._socket_liveness, self._socket_hello_timeout_s
        if liveness is None or not liveness.observable or timeout <= 0:
            return False
        session = getattr(getattr(self._handler, "client", None), "current_session", None)
        if session is None or getattr(session, "closed", True) is not False:
            return False  # a missing or closed session is the transport checks' job
        now = time.monotonic()
        if session is not liveness.seen_session:
            liveness.seen_session, liveness.seen_at = session, now
        return liveness.registered_session is not session and now - liveness.seen_at >= timeout

    def _socket_silent(self) -> bool:
        """No Slack frame for the renewal window (and no renewal backoff pending)."""
        liveness, window = self._socket_liveness, self._socket_silence_renew_s
        if liveness is None or not liveness.observable or window <= 0:
            return False
        now = time.monotonic()
        return now - liveness.last_frame >= window and now >= self._socket_renew_not_before

    async def _await_socket_registration(
            self, handler: Any, task: asyncio.Task, liveness: _SocketLiveness) -> bool:
        """Wait until Slack registers ``handler``'s session (``hello``) or its task ends, bounded."""
        registered = asyncio.ensure_future(liveness.registered.wait())
        try:
            await asyncio.wait(
                {registered, task}, timeout=self._socket_hello_timeout_s or 60.0,
                return_when=asyncio.FIRST_COMPLETED)
        finally:
            registered.cancel()
        session = getattr(getattr(handler, "client", None), "current_session", None)
        return (self._running and session is not None and session is liveness.registered_session
                and getattr(session, "closed", True) is False)

    async def _renew_socket_mode(self) -> None:
        """Replace a silent session make-before-break. An idle but healthy socket is renewed without a
        delivery gap, and a deaf one (accepted by the edge, never routed) is replaced. If Slack does
        not register the new session, keep the current one and back off."""
        if not self._running:
            return
        async with self._socket_reconnect_lock:
            current = self._socket_liveness
            if not self._running or not self._app or not self._app_token or current is None:
                return
            silent_for = time.monotonic() - current.last_frame
            try:
                handler, task, liveness = self._new_socket_mode_handler()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._socket_renew_not_before = time.monotonic() + self._socket_renew_backoff_s
                logger.error("[Slack] Socket Mode renewal could not start: %s", exc, exc_info=True)
                return
            swapped = False
            try:
                if await self._await_socket_registration(handler, task, liveness):
                    old_handler, old_task = self._handler, self._socket_mode_task
                    self._adopt_socket_handler(handler, task, liveness)
                    self._socket_renew_not_before = 0.0
                    swapped = True
                    logger.info(
                        "[Slack] Socket Mode renewed after %.0fs without Slack frames "
                        "(connections=%s)", silent_for, liveness.connections)
                    await self._close_socket_mode_handler(old_handler, old_task)
                elif self._running:
                    self._socket_renew_not_before = time.monotonic() + self._socket_renew_backoff_s
                    logger.warning(
                        "[Slack] Socket Mode renewal: Slack did not register a new session; keeping "
                        "the current one and retrying in %.0fs", self._socket_renew_backoff_s)
            finally:
                if not swapped:
                    await self._close_socket_mode_handler(handler, task)

    async def _restart_socket_mode(self, reason: str) -> None:
        """Reconnect Socket Mode without rebuilding adapter state."""
        if not self._running:
            return
        handler = self._handler
        async with self._socket_reconnect_lock:
            if not self._running or not self._app or not self._app_token:
                return
            if self._handler is not handler:
                return  # replaced while we waited (watchdog vs. task done-callback): already healed
            logger.warning("[Slack] Socket Mode unhealthy (%s); reconnecting", reason)
            await self._stop_socket_mode_handler()
            self._socket_disconnected_samples = 0
            try:
                self._start_socket_mode_handler()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("[Slack] Socket Mode reconnect failed: %s", exc, exc_info=True)

    async def _socket_watchdog_loop(self) -> None:
        """Monitor Socket Mode and reconnect if the task/transport dies.
        Broad except so a transient probe/restart bug can't kill self-healing."""
        while self._running:
            try:
                await asyncio.sleep(self._socket_watchdog_interval_s)
                if not self._running:
                    break
                await self._socket_watchdog_tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "[Slack] Socket Mode watchdog iteration failed; continuing", exc_info=True)

    async def _socket_watchdog_tick(self) -> None:
        """One watchdog check, cheapest signal first: task, transport, pings, dispatch, then the
        app-level checks that only Slack frames can answer."""
        task = self._socket_mode_task
        if task is None:
            await self._restart_socket_mode("socket task missing")
            return
        if task.done():
            await self._restart_socket_mode("socket task stopped")
            return
        connected = await self._socket_transport_connected()
        if connected is False:
            # One sample can land inside the SDK's own reconnect (Slack refreshes sessions every
            # few hours), and restarting then only races it. Act on two in a row.
            self._socket_disconnected_samples += 1
            if self._socket_disconnected_samples >= 2:
                await self._restart_socket_mode("transport disconnected")
            return
        self._socket_disconnected_samples = 0
        if self._socket_ping_pong_stale():
            # is_connected() can lie on a closed session; staleness catches the zombie.
            await self._restart_socket_mode("ping/pong stale")
        elif self._socket_processor_stopped():
            await self._restart_socket_mode("message processor stopped")
        elif self._socket_unregistered():
            # Pings prove only the transport; without hello the session gets no events.
            await self._restart_socket_mode(
                f"no hello from Slack within {self._socket_hello_timeout_s:.0f}s, "
                "session not registered")
        elif self._socket_silent():
            await self._renew_socket_mode()

    def _on_socket_watchdog_done(self, task: asyncio.Task) -> None:
        if task is not self._socket_watchdog_task:
            return
        if task.cancelled() or not self._running:
            return
        try:
            exc = task.exception()
        except (asyncio.CancelledError, Exception):  # pragma: no cover
            exc = None
        if exc is not None:
            logger.warning(
                "[Slack] Socket Mode watchdog exited with error; restarting: %s", exc, exc_info=True
            )
        else:
            logger.warning("[Slack] Socket Mode watchdog exited; restarting")
        self._socket_watchdog_task = None
        self._ensure_socket_watchdog()

    async def _cancel_socket_watchdog(self, failure_msg: str) -> None:
        """Cancel and await the watchdog task (if any); exceptions are debug-logged."""
        watchdog_task = self._socket_watchdog_task
        self._socket_watchdog_task = None
        if watchdog_task is None or watchdog_task.done():
            return
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(failure_msg, exc_info=True)

    def _ensure_socket_watchdog(self) -> None:
        if self._socket_watchdog_task is None or self._socket_watchdog_task.done():
            task = asyncio.create_task(self._socket_watchdog_loop())
            self._socket_watchdog_task = task
            task.add_done_callback(self._on_socket_watchdog_done)

    def _on_socket_mode_task_done(self, task: asyncio.Task) -> None:
        # Ignore stale tasks from intentional reconnect/shutdown.
        if task is not self._socket_mode_task or task.cancelled() or not self._running:
            return
        exc = None
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("[Slack] Could not inspect Socket Mode task exception", exc_info=True)
        if exc is not None:
            logger.warning("[Slack] Socket Mode task exited with error: %s", exc, exc_info=True)
        else:
            logger.warning("[Slack] Socket Mode task exited unexpectedly")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._restart_socket_mode("socket task exited"))
