"""Keepalive strike policy for non-loopback WebSocket listeners (C2a, t_e15288d4).

The measured production churn: laptop-app sockets die on a 40 s grid (p50 40.0 s; 99 % of
short-lived closes land within 0.25 s of a 20 s multiple), and the Step-1 harness reproduces it
through the production listener — a client whose process the OS suspends (App Nap, sleep,
SIGSTOP-class freeze) cannot answer pongs, and uvicorn's sansio keepalive fails it at exactly
``ws_ping_interval + ws_ping_timeout`` (the 20/20 non-loopback default: 40.0 s, code 1005).

One missed pong cycle is inconclusive for a backgrounded renderer: the kernel keeps ACKing while
the process is frozen, and a waking client answers the buffered ping within milliseconds. So
require two consecutive misses before failing the connection — but only when the whole detection
budget still fits under the proxy idle cap the non-loopback bind sits behind (docs cite the
Cloudflare Tunnel ~100 s). Half-open detection must stay inside that cap; where one longer
cycle already owns the budget (the SSH-tunnel path's 60/600), the policy leaves the transport's
own single-strike arithmetic alone.

The policy is an override of the sansio keepalive, so it applies only while uvicorn's own
``ws="auto"`` transport is a sansio that HAS that keepalive: on older uvicorn (the dependency
floor) there is nothing to extend, and the listener keeps uvicorn's transport untouched rather
than a subclass whose ping never fires.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)

#: Non-loopback binds sit behind proxies that drop idle connections at roughly this idle time;
#: a genuinely half-open connection must be detected before the proxy does.
WS_KEEPALIVE_DETECT_CAP_S = 100.0

#: Consecutive missed pong cycles that count as evidence the peer is gone.
DEFAULT_WS_PING_STRIKES = 2


def ws_ping_strikes(ping_interval: Optional[float], ping_timeout: Optional[float]) -> int:
    """Missed pongs to tolerate for a listener with this cadence.

    Two cycles only buy rescue time if detection still lands under ``WS_KEEPALIVE_DETECT_CAP_S``
    (defaults 20/20 → kill at 60 s ✓). A cadence that cannot fit two cycles under the cap is one
    the operator chose for a slow link; keep its single long cycle (60/600 → 660 s) rather than
    doubling it past every watchdog that depends on the detection.
    """
    if ping_interval is None or ping_timeout is None:  # pings off (loopback bind)
        return 1
    if ping_interval + DEFAULT_WS_PING_STRIKES * ping_timeout <= WS_KEEPALIVE_DETECT_CAP_S:
        return DEFAULT_WS_PING_STRIKES
    return 1


def _make_strike_protocol(base: type, strikes: int) -> type:
    class KeepaliveStrikeProtocol(base):  # type: ignore[valid-type]
        """``WebSocketsSansIOProtocol`` failing a connection only after N consecutive missed pongs.

        A first timeout re-pings immediately (the second strike's pong deadline is re-armed by
        ``send_keepalive_ping``); an in-flight pong clears the counter through ``handle_pong``,
        so a client that wakes inside the grace keeps its socket. The final strike defers to the
        transport's own failure path — the close report (1005 to the app, per
        ``tests/tui_gateway/test_ws_close_initiated.py``) is unchanged, only its timing moves.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._missed_pongs = 0

        def handle_pong(self, event) -> None:
            # Match against the pending payload BEFORE super() clears it: a matching pong means
            # the peer is alive however many strikes we have already counted.
            answered = (self.pending_ping_payload is not None
                        and bytes(event.data) == self.pending_ping_payload)
            super().handle_pong(event)
            if answered:
                self._missed_pongs = 0

        def keepalive_timeout(self) -> None:
            if (self._missed_pongs + 1 < strikes
                    and not self.close_sent and not self.transport.is_closing()):
                self._missed_pongs += 1
                self.pong_timer = None
                self.pending_ping_payload = None
                self.logger.debug("WebSocket keepalive ping timeout (strike %d/%d)",
                                  self._missed_pongs, strikes)
                self.send_keepalive_ping()  # arms the next pong deadline; handle_pong chains the rest
                return
            super().keepalive_timeout()

    return KeepaliveStrikeProtocol


_STRIKE_PROTOCOLS: dict[int, type[Any]] = {}

#: The three hooks ``_make_strike_protocol`` overrides. They arrived together with the sansio
#: keepalive, so their presence is the capability probe — no version table to keep current.
_KEEPALIVE_HOOKS = ("send_keepalive_ping", "keepalive_timeout", "handle_pong")

#: Warn-once latch for the fallback below: every listener build would otherwise repeat one
#: static verdict about this uvicorn install.
_no_base_fallback_logged = False


def _sansio_strike_base() -> Optional[Any]:
    """The transport the strike policy may extend, or ``None`` when this uvicorn has none.

    The base is uvicorn's OWN ``ws="auto"`` class, and only while that class carries the
    keepalive hooks the policy overrides. Both halves matter: subclassing a class uvicorn would
    not have run swaps the listener's transport, and subclassing a sansio that predates the
    keepalive (the ``pyproject`` floor's 0.41: ``auto`` resolves to the LEGACY
    ``websockets_impl.WebSocketProtocol``, whose own ping does the killing, and the sansio class
    has no ping at all) produced a transport that never pings — half-open detection silently
    OFF. ``None`` makes the callers hand uvicorn ``"auto"``, i.e. that uvicorn's own behavior,
    single-strike and intact.

    ``Any``, not ``type``: uvicorn annotates the ``auto`` alias as a callable, and the class it
    resolves to at import time is what we subclass.
    """
    try:
        from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol
    except ImportError:  # pragma: no cover -- a uvicorn build without websocket protocols
        return None
    if not all(callable(getattr(AutoWebSocketsProtocol, hook, None)) for hook in _KEEPALIVE_HOOKS):
        return None
    return AutoWebSocketsProtocol


def strike_keepalive_protocol(ws_ping: tuple[Optional[float], Optional[float]]) -> Any:
    """The ``uvicorn.Config(ws=...)`` value for a listener with cadence *ws_ping* (interval, timeout);
    ``"auto"`` keeps the transport untouched.

    "auto" means "no change to the transport": pings off, a single-strike policy (identical to
    upstream), or a uvicorn whose auto-resolved transport has no sansio keepalive to extend
    (the pyproject floor, or no ``websockets`` package installed → uvicorn falls back to wsproto,
    whose keepalive is different code) — in every one of those the transport uvicorn picks keeps
    its own detection working. Returns ``Any`` because the value IS the sansio subclass at
    runtime; uvicorn's own annotation wants ``type[Protocol]``, which a dynamic ``class X(base)``
    is not statically.
    """
    strikes = ws_ping_strikes(ws_ping[0], ws_ping[1])
    if strikes <= 1:
        return "auto"
    cached = _STRIKE_PROTOCOLS.get(strikes)
    if cached is not None:
        return cached
    base = _sansio_strike_base()
    if base is None:
        global _no_base_fallback_logged
        if not _no_base_fallback_logged:
            _no_base_fallback_logged = True
            import uvicorn  # hard dependency; its version names which floor we are on

            _log.warning(
                "uvicorn %s has no sansio keepalive to extend, so the two-strike tolerance is "
                "not applied to non-loopback listeners: a suspended peer still fails after ONE "
                "missed pong (half-open detection itself is unchanged — uvicorn's own transport "
                "runs). Upgrade uvicorn to get the rescue band.", uvicorn.__version__)
        return "auto"
    protocol = _make_strike_protocol(base, strikes)
    _STRIKE_PROTOCOLS[strikes] = protocol
    return protocol
