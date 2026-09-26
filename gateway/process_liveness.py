"""Gateway event-loop liveness probes and bounded escalation."""
from __future__ import annotations

import contextlib
import json
import socket
import time
from pathlib import Path

from gateway.status import terminate_pid

GATEWAY_LOOP_ALIVE = "alive"
GATEWAY_LOOP_WEDGED = "wedged"
GATEWAY_LOOP_UNKNOWN = "unknown"
DEFAULT_LOOP_LIVENESS_STALE_AFTER_S = 90.0
_LOOP_TICK_ABSENT = object()

def _wait_for_pid_exit(pid: int, timeout: float, *, on_progress=None) -> bool:
    """Wait up to ``timeout``s for ``pid`` to exit; True once gone. (``launchctl bootstrap`` fails EIO
    while the previous instance still drains, so teardown callers must wait for the real exit.)"""
    if pid <= 0:
        return True
    # ``os.kill(pid, 0)`` hard-kills on Windows (TerminateProcess); use _pid_exists instead.
    from gateway.status import _pid_exists
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if not _pid_exists(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        if on_progress is not None:
            on_progress()
        time.sleep(0.5)

def _probe_loop_tick_socket(pid: int, home: Path | None, timeout: float = 1.0) -> bool | None:
    """Ping the loop-tick witness socket: True answered, False node present but silent, None no node (not evidence)."""
    try:
        from gateway.shutdown_watchdog import get_loop_tick_socket_path
        path = get_loop_tick_socket_path(home, pid)
        if not path.is_socket():
            return None
    except Exception:
        return None
    return _ping_loop_tick_witness(socket.AF_UNIX, str(path), timeout)

def _ping_loop_tick_witness(family: int, address, timeout: float) -> bool:
    """Connect to a loop-tick witness and expect one byte ``"1"``; False on refusal/timeout/any error."""
    sock = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(max(float(timeout), 0.0))
        sock.connect(address)
        return sock.recv(1) == b"1"
    except Exception:
        return False
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()

def _probe_loop_tick_tcp(port: int, timeout: float = 1.0) -> bool | None:
    """TCP-loopback variant of the tick probe for Windows (no AF_UNIX in asyncio); same semantics, None
    on invalid port."""
    try:
        port_num = int(port)
        if port_num <= 0 or port_num > 65535:
            return None
    except (TypeError, ValueError):
        return None
    return _ping_loop_tick_witness(socket.AF_INET, ("127.0.0.1", port_num), timeout)

def _probe_loop_tick_socket_sustained(
    pid: int, home: Path | None, *, timeout: float = 1.0, strikes: int = 3, gap_s: float = 0.2,
    tcp_port: int | None = None,
) -> bool | None:
    """Probe the tick socket up to ``strikes`` times, ``gap_s`` apart: True once answered, False if a node
    stayed silent the whole window, None if the node vanished (not evidence). One silent probe is not
    destructive evidence — a transient synchronous stall can outlast one recv timeout.

    A single silent probe is NOT destructive evidence: the loop may be in a short transient synchronous
    stall (a reconnect storm, a heavy synchronous callback, scheduler delay) that outlasts one recv timeout.
    Killing a gateway on that would be a false wedge — the exact class of false positive #90502 exists to
    prevent. Destructive authority therefore requires the loop to fail to answer across a bounded window of
    ``strikes`` consecutive misses, ``gap_s`` apart; any answer inside the window proves the loop is
    dispatching and returns ``True``.
    """
    total = max(int(strikes), 0)
    for attempt in range(total):
        if tcp_port is not None:
            result = _probe_loop_tick_tcp(tcp_port, timeout=timeout)
        else:
            result = _probe_loop_tick_socket(pid, home, timeout=timeout)
        if result is True:
            return True
        if result is None:
            # No node: ambiguity, never a wedge — absence is not a miss.
            return None
        if attempt < total - 1 and gap_s > 0:
            time.sleep(gap_s)
    return False

def probe_gateway_loop_liveness(
    pid: int, *, stale_after: float = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S, home: Path | None = None,
    tick_timeout: float = 1.0, tick_strikes: int = 3, tick_gap_s: float = 0.2,
) -> str:
    """Classify a gateway PID's event loop as alive / wedged / unknown (see block comment above).
    Stale heartbeat is ``wedged`` only when the payload declares the tick socket armed AND it stays
    silent across ``tick_strikes`` misses; any answer is ``alive``; ambiguity is ``unknown``.

    - the loop-tick socket (``state/gateway.loop-tick.<pid>.sock``): answered by the gateway loop itself, so
    a reply is direct proof that the loop is dispatching. It is never refreshed by the heartbeat executor
    thread and never stalled by a filesystem that is slow to fsync. - the heartbeat file
    (``state/gateway.heartbeat``): rewritten every 30s on a thread since #90502, so freshness alone is no
    longer proof of loop schedulability — a stalled write (measured at 112.6s max on the incident box) or a
    saturated executor can age the file while the loop runs, and a write can land after the loop froze.
    """
    try:
        stale_budget = max(float(stale_after), 0.0)
    except (TypeError, ValueError):
        stale_budget = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path
        path = get_loop_heartbeat_path(home)
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
        heartbeat_pid = int(payload.get("pid", 0))
    except Exception:
        return GATEWAY_LOOP_UNKNOWN
    if heartbeat_pid <= 0 or int(pid) <= 0 or heartbeat_pid != int(pid):
        # Heartbeat is not this process's (old version, starting up, stale file): not evidence.
        return GATEWAY_LOOP_UNKNOWN

    # TCP loopback witness (Windows) takes priority when published; else the AF_UNIX socket.
    tcp_port = payload.get("loop_tick_tcp_port")
    try:
        tcp_port_int = int(tcp_port) if tcp_port is not None else None
    except (TypeError, ValueError):
        tcp_port_int = None

    if tcp_port_int is not None and tcp_port_int > 0:
        witness = _probe_loop_tick_tcp(tcp_port_int, timeout=tick_timeout)
        tick_armed = True
    else:
        witness = _probe_loop_tick_socket(pid, home, timeout=tick_timeout)
        tick_armed = payload.get("loop_tick_socket", _LOOP_TICK_ABSENT)
    if witness is True:
        # Loop answered: a stale file is a stalled write, not a wedge.
        return GATEWAY_LOOP_ALIVE
    # The loop answered a ping — it is dispatching right now. See #90502.
    age = time.time() - mtime
    if age <= stale_budget:
        if witness is False:
            # Fresh file but silent loop: an off-loop write can land after the loop froze.
            return GATEWAY_LOOP_UNKNOWN
        return GATEWAY_LOOP_ALIVE

    # Stale past the budget; the verdict depends on what the producer promised about its witness.
    if tick_armed is _LOOP_TICK_ABSENT:
        # Legacy on-loop writer: staleness proves the loop stopped scheduling.
        return GATEWAY_LOOP_WEDGED
    if tick_armed is not True:
        # Witness could not be armed (bind failed); off-loop write means staleness is not proof.
        return GATEWAY_LOOP_UNKNOWN
    if witness is False:
        # First miss. The probe above is miss #1, so ``tick_strikes - 1`` more attempts follow.
        # One silent probe is NOT destructive authority: a short transient synchronous stall can outlast a
        # single recv timeout, and killing a live gateway on it would be the exact false wedge #90502 exists
        # to prevent.
        sustained = _probe_loop_tick_socket_sustained(
            pid, home, timeout=tick_timeout, strikes=tick_strikes - 1, gap_s=tick_gap_s, tcp_port=tcp_port_int
        )
        if sustained is False:
            return GATEWAY_LOOP_WEDGED
        if sustained is True:
            return GATEWAY_LOOP_ALIVE  # Transient stall, not a wedge.
        return GATEWAY_LOOP_UNKNOWN  # Witness vanished mid-window: ambiguity — never kill on it.
    return GATEWAY_LOOP_UNKNOWN  # Armed but unreachable socket: ambiguity — never kill on it.

def _escalate_wedged_gateway(pid: int, *, term_grace: float = 5.0, kill_wait: float = 5.0) -> bool:
    """Bounded stop (SIGTERM, ``term_grace``, SIGKILL, ``kill_wait``) for a provably dead loop; True once gone.
    Callers MUST have classified ``GATEWAY_LOOP_WEDGED`` first: escalating a merely busy gateway
    bypasses the cron drain floor and SIGKILLs live work.

    See #86684.
    """
    from runtime.process_identity import get_process_start_time
    expected_start_time = get_process_start_time(pid)
    try:
        terminate_pid(pid, force=False)
    except (ProcessLookupError, PermissionError, OSError):
        return _wait_for_pid_exit(pid, 1.0)
    if _wait_for_pid_exit(pid, max(float(term_grace), 0.0)):
        return True
    try:
        terminate_pid(pid, force=True, expected_start_time=expected_start_time)
        print(f"⚠ Gateway PID {pid} unresponsive to SIGTERM; sent SIGKILL")
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return _wait_for_pid_exit(pid, max(float(kill_wait), 0.0))
