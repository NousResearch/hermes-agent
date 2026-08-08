"""Bounded in-memory diagnostics ring for gateway-side hitch capture (U3/KTD2).

The gateway already detects the two boundary signals that matter for the
desktop hitching investigation — the CF-1 loop heartbeat ("event loop stalled
Ns") in :mod:`hermes_cli.web_server` and the WS write timeout ("ws write slow")
in :mod:`tui_gateway.ws` — but both only reach a log file, at thresholds tuned
for *incidents* (5s / 10s). A 400ms hitch never leaves a trace anywhere.

This module adds the machine-readable sink those detectors lack (KTD4), under
an explicit arm/disarm capture so the cost is paid only while someone is
looking:

* **Disarmed** (the normal state) the whole module is one global bool read on
  the two hot paths. No thread, no timer change, no allocation.
* **Armed** the heartbeat tightens to :data:`_ARMED_HB_INTERVAL_S` and every
  drift sample above :data:`_CAPTURE_FLOOR_S` becomes a ring event, a watchdog
  *thread* (deliberately off the event loop — the loop is exactly what is
  blocked) samples the stalled loop thread's stack for attribution, and WS
  write-slow events join the ring with their peer identifier HMAC'd under a
  random per-capture key.

The ring is pulled over the gateway's existing authenticated JSON-RPC surface
(``/api/ws`` → ``tui_gateway.server.dispatch``) via the three methods installed
by :func:`install_rpc_methods` — there is no new listener and no new auth path.
"""

from __future__ import annotations

import hmac
import os
import secrets
import sys
import threading
import time
from collections import deque
from hashlib import sha256
from typing import Any

# Fast-path flag read by the heartbeat and the WS write timeout. Kept as a
# plain module global (not a function call, not an object attribute) so the
# disarmed cost is a single LOAD_GLOBAL on paths that run per tick / per frame.
ARMED = False

# Ring bounds (KTD2: fixed-size, ~300s). Whichever limit is hit first evicts
# the oldest event and increments the dropped counter, so a capture left armed
# overnight costs a constant amount of memory.
_MAX_EVENTS = 4096
_MAX_AGE_S = 300.0

# Events below this drift never reach the ring: at 20Hz a healthy loop produces
# a sample every 50ms and none of them are interesting. 250ms is the floor the
# plan sets — below the perceptible-hitch threshold, far below the 5s log line.
_CAPTURE_FLOOR_S = 0.25

# Heartbeat tick while armed. The production 2.0s tick cannot resolve a 400ms
# block at all (its drift measurement is `elapsed - interval`, so a sub-interval
# block is invisible or under-reported); 20Hz makes every block above the floor
# observable while costing ~20 trivial loop wakeups per second for the length of
# a capture only.
_ARMED_HB_INTERVAL_S = 0.05

# Watchdog cadence. Same order as the armed tick so a stall is noticed within
# ~one floor of starting, while the sampling itself is capped at one stack grab
# per stall episode.
_WATCHDOG_POLL_S = 0.05

# Depth cap for a sanitized stack summary. Deep enough to cross the asyncio
# frames and name real work, shallow enough to stay a small dict list.
_MAX_FRAMES = 16

# Truncated HMAC width. 16 hex chars (64 bits) is plenty to correlate peers
# inside one capture and keeps the exported bundle readable.
_TOKEN_HEX = 16
_TOKEN_CACHE_MAX = 512

# Repo root, used to turn an absolute source path into a package-relative
# dotted module name. Frame summaries must never carry absolute paths.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_lock = threading.RLock()
_capture: "_Capture | None" = None


class _Capture:
    """One armed capture window. Guarded by the module ``_lock``."""

    __slots__ = (
        "capture_id",
        "monotonic_anchor",
        "wall_clock_anchor_ms",
        "events",
        "dropped",
        "key",
        "tokens",
        "last_tick",
        "loop_thread_id",
        "pending_frames",
        "sampled_episode",
        "stop",
        "thread",
    )

    def __init__(self, capture_id: str, wall_clock_anchor_ms: float | None) -> None:
        self.capture_id = capture_id
        self.monotonic_anchor = time.monotonic()
        self.wall_clock_anchor_ms = wall_clock_anchor_ms
        self.events: deque[dict[str, Any]] = deque()
        self.dropped = 0
        # Random per-capture HMAC key. Never returned by any method, never
        # written anywhere, discarded on disarm — identifiers correlate inside
        # one capture and are not reversible outside it.
        self.key = secrets.token_bytes(32)
        self.tokens: dict[str, str] = {}
        self.last_tick = self.monotonic_anchor
        self.loop_thread_id: int | None = None
        self.pending_frames: list[dict[str, Any]] | None = None
        self.sampled_episode = False
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None


# ── Sanitization ─────────────────────────────────────────────────────


def _module_name(filename: str) -> str:
    """Dotted, package-relative module name for a frame's source file.

    Absolute path components outside the source tree are dropped entirely: a
    stdlib or site-packages frame collapses to its bare module name. Nothing
    that leaves this function can leak a home directory or a checkout path.
    """
    try:
        path = os.path.abspath(filename)
    except Exception:
        return "<unknown>"
    if path.startswith(_SRC_ROOT + os.sep):
        path = path[len(_SRC_ROOT) + 1:]
    else:
        path = os.path.basename(path)
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace(os.sep, ".").replace("/", ".")


def _sample_frames(thread_id: int) -> list[dict[str, Any]] | None:
    """Sanitized stack summary for *thread_id* — module/function/line only.

    No arguments, no locals, no source text: a frame summary is three scalars
    per level, which is what makes the stall attribution safe to ship inside a
    capture bundle.
    """
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return None
    out: list[dict[str, Any]] = []
    while frame is not None and len(out) < _MAX_FRAMES:
        code = frame.f_code
        out.append({
            "module": _module_name(code.co_filename),
            "function": code.co_name,
            "line": frame.f_lineno,
        })
        frame = frame.f_back
    return out


# ── Ring writes ──────────────────────────────────────────────────────


def _append(cap: _Capture, event: dict[str, Any]) -> None:
    """Append under ``_lock``, evicting by count and by age."""
    cap.events.append(event)
    cutoff = event["t_monotonic"] - _MAX_AGE_S
    while cap.events and (
        len(cap.events) > _MAX_EVENTS or cap.events[0]["t_monotonic"] < cutoff
    ):
        cap.events.popleft()
        cap.dropped += 1


def _token(cap: _Capture, value: str) -> str:
    """HMAC-SHA256 a peer/stream identifier under the per-capture key."""
    cached = cap.tokens.get(value)
    if cached is not None:
        return cached
    digest = hmac.new(cap.key, value.encode("utf-8", "replace"), sha256).hexdigest()
    token = digest[:_TOKEN_HEX]
    if len(cap.tokens) >= _TOKEN_CACHE_MAX:
        cap.tokens.clear()
    cap.tokens[value] = token
    return token


# ── Watchdog thread ──────────────────────────────────────────────────


def _watchdog(cap: _Capture) -> None:
    """Off-loop stall detector + stack sampler.

    Runs in its own thread precisely because the event loop is the thing that
    stalls: a loop-scheduled sampler cannot observe its own block. It watches
    the gap since the last heartbeat tick and, once per stall episode, grabs a
    sanitized stack of the loop thread. The heartbeat attaches that summary to
    the drift event it records when the loop finally breathes.
    """
    while not cap.stop.wait(_WATCHDOG_POLL_S):
        with _lock:
            tid = cap.loop_thread_id
            gap = time.monotonic() - cap.last_tick
            if gap <= _CAPTURE_FLOOR_S:
                # Episode over — re-arm sampling for the next one.
                cap.sampled_episode = False
                continue
            if cap.sampled_episode or tid is None:
                continue
            cap.sampled_episode = True
        # Sample outside the lock: sys._current_frames() walks every thread and
        # we do not want the blocked loop's next tick queuing behind us.
        frames = _sample_frames(tid)
        if frames is None:
            continue
        with _lock:
            if cap.stop.is_set():
                return
            cap.pending_frames = frames


# ── Arm / disarm / collect ───────────────────────────────────────────


def arm(capture_id: str, wall_clock_anchor_ms: float | None = None) -> dict[str, Any]:
    """Start (or re-confirm) a capture window. Idempotent per capture_id.

    Returns the monotonic anchor the exporter aligns this stream on (KTD3).
    Re-arming the same capture_id is a no-op that returns the original anchor;
    arming a different capture_id replaces the window and drops the old ring.
    """
    global ARMED, _capture
    if not capture_id or not isinstance(capture_id, str):
        raise ValueError("capture_id must be a non-empty string")
    with _lock:
        cap = _capture
        if cap is not None and cap.capture_id == capture_id:
            return {"monotonic_anchor_ms": cap.monotonic_anchor * 1000.0}
        if cap is not None:
            _stop_locked(cap)
        cap = _Capture(capture_id, wall_clock_anchor_ms)
        cap.thread = threading.Thread(
            target=_watchdog, args=(cap,), daemon=True,
            name=f"hermes-diag-watchdog-{capture_id[:8]}",
        )
        _capture = cap
        ARMED = True
        cap.thread.start()
        return {"monotonic_anchor_ms": cap.monotonic_anchor * 1000.0}


def _stop_locked(cap: _Capture) -> None:
    """Signal the watchdog and forget the per-capture key. Caller holds _lock."""
    cap.stop.set()
    cap.key = b""
    cap.tokens.clear()


def disarm() -> dict[str, Any]:
    """End the capture window. Idempotent; safe to call when never armed."""
    global ARMED, _capture
    with _lock:
        cap = _capture
        _capture = None
        ARMED = False
        if cap is None:
            return {"armed": False}
        _stop_locked(cap)
        return {"armed": False, "capture_id": cap.capture_id}


def collect(capture_id: str) -> dict[str, Any]:
    """Snapshot the ring for *capture_id*.

    Non-draining: the desktop capture controller pulls once at export, and a
    retried pull must not come back empty. Raises :class:`LookupError` when no
    capture is armed under that id.
    """
    with _lock:
        cap = _capture
        if cap is None or cap.capture_id != capture_id:
            raise LookupError(f"no armed capture for capture_id={capture_id!r}")
        # Age out before snapshotting so a long-idle capture reports the same
        # bounded window it would have reported on the next write.
        cutoff = time.monotonic() - _MAX_AGE_S
        while cap.events and cap.events[0]["t_monotonic"] < cutoff:
            cap.events.popleft()
            cap.dropped += 1
        return {
            "capture_id": cap.capture_id,
            "monotonic_anchor_ms": cap.monotonic_anchor * 1000.0,
            "wall_clock_anchor_ms": cap.wall_clock_anchor_ms,
            "events": list(cap.events),
            "dropped": cap.dropped,
        }


def is_armed() -> bool:
    """Test/introspection helper — hot paths read :data:`ARMED` directly."""
    return ARMED


def armed_heartbeat_interval() -> float:
    """Heartbeat interval to use while a capture is armed."""
    return _ARMED_HB_INTERVAL_S


# ── Detector entry points ────────────────────────────────────────────


def note_loop_tick() -> None:
    """Record loop liveness for the watchdog. Called from the heartbeat tick."""
    with _lock:
        cap = _capture
        if cap is None:
            return
        cap.last_tick = time.monotonic()
        cap.loop_thread_id = threading.get_ident()


def record_loop_drift(drift_s: float) -> dict[str, Any] | None:
    """Record a heartbeat drift sample above the capture floor.

    Returns the event (for tests) or None when the sample is below the floor.
    The caller's 5s log threshold is unrelated and unchanged — this is a
    strictly additional, strictly in-memory sink.
    """
    if drift_s < _CAPTURE_FLOOR_S:
        return None
    with _lock:
        cap = _capture
        if cap is None:
            return None
        event: dict[str, Any] = {
            "kind": "loop_drift",
            "t_monotonic": time.monotonic(),
            "drift_s": round(float(drift_s), 6),
        }
        frames = cap.pending_frames
        if frames is not None:
            cap.pending_frames = None
            event["frames"] = frames
        _append(cap, event)
        return event


def record_ws_write_slow(peer: Any, timeout_s: float, stream: Any = None) -> None:
    """Record a ``ws write slow`` / write-timeout event.

    ``peer`` (and ``stream``, when given) are HMAC'd under the per-capture key,
    never stored in the clear: an exported bundle correlates the frames of one
    connection without ever naming the address they went to.
    """
    with _lock:
        cap = _capture
        if cap is None:
            return
        event: dict[str, Any] = {
            "kind": "ws_write_slow",
            "t_monotonic": time.monotonic(),
            "timeout_s": round(float(timeout_s), 6),
            "peer": _token(cap, str(peer)),
        }
        if stream is not None:
            event["stream"] = _token(cap, str(stream))
        _append(cap, event)


# ── JSON-RPC surface ─────────────────────────────────────────────────

# The desktop capture controller drove these as ``diagnostics/arm`` etc.; the
# gateway's own convention is dotted (``session.create``, ``config.get``). Both
# spellings are registered onto the same handlers so neither side has to guess.
_METHOD_ALIASES = {
    "arm": ("diagnostics.arm", "diagnostics/arm"),
    "disarm": ("diagnostics.disarm", "diagnostics/disarm"),
    "collect": ("diagnostics.collect", "diagnostics/collect"),
}

# JSON-RPC error code for a bad/absent capture, in the gateway's 5xxx band.
_ERR_NO_CAPTURE = 5091


def install_rpc_methods(server) -> None:
    """Register arm/disarm/collect on ``tui_gateway.server``'s method table.

    Deliberately *only* the existing surface: these ride ``/api/ws``, which is
    gated by ``web_server._ws_auth_ok`` before a single frame is dispatched, so
    an unauthenticated caller never reaches a handler. No HTTP route, no second
    listener, no per-capture bearer token to leak (KTD4 / U3 step 4).
    """

    def _arm(rid, params: dict):
        try:
            result = arm(
                params.get("capture_id"),
                params.get("wall_clock_anchor_ms"),
            )
        except ValueError as exc:
            return server._err(rid, _ERR_NO_CAPTURE, str(exc))
        return server._ok(rid, result)

    def _disarm(rid, params: dict):
        return server._ok(rid, disarm())

    def _collect(rid, params: dict):
        try:
            return server._ok(rid, collect(params.get("capture_id")))
        except LookupError as exc:
            return server._err(rid, _ERR_NO_CAPTURE, str(exc))

    handlers = {"arm": _arm, "disarm": _disarm, "collect": _collect}
    for key, names in _METHOD_ALIASES.items():
        for name in names:
            server._methods[name] = handlers[key]


# ── Test-only loop block (U5) ────────────────────────────────────────
#
# The proof harness has to show that a *gateway* stall is attributed to the
# gateway, which means it needs to cause one on demand. Nothing in normal
# operation can be asked to block the loop, so this hook exists — and it is
# dangerous by construction, so it is not reachable at all unless the process
# was started with :data:`_TEST_HOOKS_ENV` set to ``1``. The guard is checked
# once, at *registration* time: without the env var the route is never mounted,
# so a normally started gateway answers 404 rather than "refused".

_TEST_HOOKS_ENV = "HERMES_DIAGNOSTICS_TEST_HOOKS"

# The harness blocks for ~1-6s; anything longer is a mistake, not a scenario.
_MAX_TEST_BLOCK_S = 30.0

_TEST_BLOCK_PATH = "/api/diagnostics/test/block-loop"


def test_hooks_enabled() -> bool:
    """True only when this process opted into the diagnostics test hooks."""
    return os.environ.get(_TEST_HOOKS_ENV) == "1"


def block_event_loop(seconds: Any) -> dict[str, Any]:
    """Hold the calling thread for *seconds* — test hook only.

    Called from an ``async def`` route it holds the *event loop*, which is
    exactly the condition the CF-1 heartbeat measures as drift and the armed
    ring records as a ``loop_drift`` event.
    """
    try:
        duration = float(seconds)
    except (TypeError, ValueError):
        raise ValueError("seconds must be a number")
    if not 0 < duration <= _MAX_TEST_BLOCK_S:
        raise ValueError(f"seconds must be within (0, {_MAX_TEST_BLOCK_S}]")
    started = time.monotonic()
    time.sleep(duration)
    return {"blocked_s": round(time.monotonic() - started, 6)}


def install_test_routes(app) -> bool:
    """Mount the guarded test hook on *app*. Returns whether it was mounted.

    A no-op — and therefore an unmounted, unreachable path — unless
    :func:`test_hooks_enabled`. The route sits under ``/api/`` like the rest of
    the ring, so when it *is* mounted it still rides ``auth_middleware``.
    """
    if not test_hooks_enabled():
        return False

    # Imported here, not at module scope: this module is loaded by the TUI
    # gateway too, and nothing outside this branch needs FastAPI. `seconds` is
    # taken as an embedded body field rather than through a `Request` so the
    # signature stays resolvable under `from __future__ import annotations`.
    from fastapi import Body, HTTPException

    @app.post(_TEST_BLOCK_PATH)
    async def diagnostics_test_block_loop(seconds: float = Body(embed=True)):
        try:
            return block_event_loop(seconds)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return True
