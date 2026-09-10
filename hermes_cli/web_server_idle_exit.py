"""Idle-exit for Desktop-owned ``hermes serve --isolated`` backends reached over SSH (#101626).

That backend is deliberately detached (``setsid``/``nohup``, PPID 1) so it survives the SSH channel
closing, and every teardown path lives on the CLIENT. A laptop that sleeps mid-session (dark wake
reconnects the tunnel, spawns a backend, sleeps again) therefore leaves a backend behind every
cycle — each one an extra writer on ``state.db``. The server needs its own liveness signal.

Two pieces, both scoped to the SSH-isolated case (a session token was handed over via
``--ssh-session-token-file``):

* An ASGI wrapper counts accepted WebSocket connections (every dashboard WS route: /api/ws,
  /api/pty, /api/console, /api/pub, /api/events, /api/audio/speak-stream) without touching the
  handlers. When the count has been zero for the grace window and no agent turn is running, the
  watchdog asks uvicorn to exit gracefully (WAL checkpoint, exit 0). An indeterminate turn probe
  fails closed: the backend stays up.
* Loopback normally disables uvicorn's WS ping (a dead local client sends FIN/RST). Across an SSH
  tunnel the local socket is healthy while the far end is asleep, so pings are the only way to notice
  a half-open tunnel; the isolated backend keeps a slow ping with a long timeout so a GIL-holding
  turn cannot trip it.

Design and the client-count/turn-probe/fail-closed shape are from #101678 by @StanleyStetson; this
is the slim redo on the decomposed web server.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

_log = logging.getLogger(__name__)

DEFAULT_IDLE_GRACE_S = 900.0
# Slow enough that a long GIL-holding turn (minutes) cannot trip it, fast enough that a sleeping
# laptop's half-open tunnel is noticed well inside the idle grace window.
TUNNEL_WS_PING_INTERVAL_S = 60.0
TUNNEL_WS_PING_TIMEOUT_S = 600.0


class IdleClientTracker:
    """Live accepted-WebSocket count plus the moment the last client left."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._live = 0
        self._last_client_at = now()

    def on_open(self) -> None:
        with self._lock:
            self._live += 1
            self._last_client_at = self._now()

    def on_close(self) -> None:
        with self._lock:
            self._live = max(0, self._live - 1)
            self._last_client_at = self._now()

    def live_count(self) -> int:
        with self._lock:
            return self._live

    def idle_for(self) -> float:
        with self._lock:
            return 0.0 if self._live else self._now() - self._last_client_at


def wrap_asgi_with_ws_tracking(app, tracker: IdleClientTracker):
    """Count WebSocket sessions at the ASGI boundary: open on the ``websocket.accept`` send, close
    when the scope ends. Handlers stay untouched, so a new WS route is tracked automatically."""

    async def _app(scope, receive, send):
        if scope.get("type") != "websocket":
            return await app(scope, receive, send)
        accepted = False

        async def _send(message):
            nonlocal accepted
            if message.get("type") == "websocket.accept" and not accepted:
                accepted = True
                tracker.on_open()
            await send(message)

        try:
            await app(scope, receive, _send)
        finally:
            if accepted:
                tracker.on_close()

    return _app


_probe_failure_logged = False


def turn_in_flight() -> Optional[bool]:
    """True/False from the gateway's running-session table; None when it cannot be read. The table
    lives on ``tui_gateway.server`` (the voice mixin's helper is bound into that namespace). None
    keeps the backend alive forever, so the cause is logged once — a silent never-exits would be
    the original bug with a new face."""
    global _probe_failure_logged
    try:
        import tui_gateway.server as gateway
        with gateway._sessions_lock:
            return any(s.get("running") for s in gateway._sessions.values())
    except Exception:
        if not _probe_failure_logged:
            _probe_failure_logged = True
            _log.warning("idle-exit turn probe unavailable; this backend will not self-retire", exc_info=True)
        return None


def should_exit_idle(tracker: IdleClientTracker, grace_s: float,
                     probe: Callable[[], Optional[bool]] = turn_in_flight) -> bool:
    """Exit only when no client has been connected for ``grace_s`` AND no turn is provably running.
    A probe that cannot answer keeps the process (fail closed)."""
    return tracker.idle_for() >= grace_s and probe() is False  # idle_for() is 0 while a client is connected


def start_idle_watchdog(server, tracker: IdleClientTracker, *, grace_s: float = DEFAULT_IDLE_GRACE_S,
                        poll_s: float = 15.0, probe: Callable[[], Optional[bool]] = turn_in_flight) -> threading.Thread:
    """Daemon thread that sets ``server.should_exit`` once :func:`should_exit_idle` holds."""

    poll_s = min(poll_s, max(0.5, grace_s / 4))

    def _loop() -> None:
        while not getattr(server, "should_exit", False):
            if should_exit_idle(tracker, grace_s, probe):
                _log.warning("SSH-isolated backend idle for %.0fs with no client and no running turn; exiting.",
                             tracker.idle_for())
                server.should_exit = True
                return
            time.sleep(poll_s)

    thread = threading.Thread(target=_loop, daemon=True, name="ssh-isolated-idle-watchdog")
    thread.start()
    return thread
