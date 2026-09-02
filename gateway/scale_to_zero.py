"""Scale-to-zero idle detection + dormant-quiesce for the gateway (Phase 0).

This is the gateway-side BEHAVIOUR layer that consumes the relay scale-to-zero
PRIMITIVES (gateway-gateway Phase 5: the buffered-flip, the durable per-instance
buffer, the wakeUrl poke, the reconnect supervisor). It owns the *decision* to go
idle, drives the relay transport's ``go_dormant()`` (D12), and then SUSPENDS the
machine itself through the local Fly Machines API socket. Wake stays platform-side:
autostart-on-wakeUrl (decisions.md Q3=C′).

Why the gateway self-suspends instead of relying on ``autostop:"suspend"``: Fly
Proxy judges idle exclusively on INBOUND proxied connections — it cannot see an
in-flight agent turn (outbound-only LLM traffic) and there is no way for the app
to signal "not ready to suspend". Mid-2026 the proxy also stopped treating open
OUTBOUND sockets as activity, so the relay WebSocket no longer masks the race:
Fly would suspend a machine mid-job, and could suspend BEFORE ``go_dormant()``
flipped the relay destination (the buffered-event black hole). Owning the suspend
call closes both: it only ever fires after the idle predicate (no running agents,
no live background work, inbound-quiet) holds AND the dormant quiesce completed.

Design constraints (decisions.md):
  - Per-instance enable is gated SOLELY by the NAS "Labs" toggle, carried to the
    gateway as the ``HERMES_SCALE_TO_ZERO`` env stamp (D11/Q8=A). NOT a user
    config key; ``scale_to_zero.idle_timeout_minutes`` IS config.yaml (D2).
  - Arm only when messaging is relay-only or absent (D1/F6) AND a wakeUrl is
    registered (§3.4(1)) AND the flag is set.
  - Idle = no in-flight agent turn AND no inbound for N min AND no live
    background work (D2/D3/F7).
  - The quiesce uses ``go_dormant()`` (socket closed + supervisor preserved),
    NEVER the stop/restart drain or ``disconnect()`` (F12/F14). The process stays
    alive; Fly freezes+resumes it.
  - ``mark_resume_pending`` is deliberately NOT called here (D13 — suspend
    preserves RAM; revive only if we move to autostop:"stop" or see kills).

The pure helpers (``parse_idle_timeout_seconds``, ``scale_to_zero_enabled``,
``messaging_is_relay_only_or_absent``, ``is_idle``, ``should_arm``) take plain
inputs so they unit-test without a live gateway.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Env flag stamped by NAS when the scaleToZero Labs toggle is on (D11/Q8=A),
# mirroring how the `relay` feature stamps GATEWAY_RELAY_URL. Truthy values only.
SCALE_TO_ZERO_ENV = "HERMES_SCALE_TO_ZERO"

# Fly-injected machine identity (present on every Fly machine). Used by the
# self-suspend call; both must be present for self_suspend_available().
FLY_APP_NAME_ENV = "FLY_APP_NAME"
FLY_MACHINE_ID_ENV = "FLY_MACHINE_ID"

# The local flaps (Fly Machines API) unix socket, available inside every Fly
# machine. A POST to /v1/apps/{app}/machines/{id}/suspend snapshots RAM and
# suspends THIS machine — the Fly-endorsed replacement for proxy autostop when
# the app must own the idle decision (https://fly.io/docs/reference/suspend-resume/).
FLY_API_SOCKET = "/.fly/api"


# config.yaml default (D2). Behavioural setting -> config, not env.
# 2 minutes: with the gateway owning the suspend (idle predicate covers agent
# turns, cron, API runs, and background work; the relay drains + flips before
# the freeze), a short window is safe — real work always blocks the suspend and
# resume-from-suspend is sub-second, so the only cost of waking "too eagerly"
# after a quiet spell is a Fly-proxied poke away. Longer windows just bill idle
# RAM. Raise per-instance via gateway.scale_to_zero.idle_timeout_minutes.
DEFAULT_IDLE_TIMEOUT_MINUTES = 2

_TRUTHY = {"1", "true", "yes", "on"}


def scale_to_zero_enabled(environ: Optional[dict] = None) -> bool:
    """Whether the per-instance Labs toggle is on (the HERMES_SCALE_TO_ZERO stamp).

    D11/Q8=A: this env flag is the SOLE per-instance enable signal reaching the
    gateway. Absent/blank/falsey -> disabled (fail-safe default off).
    """
    env = environ if environ is not None else os.environ
    return str(env.get(SCALE_TO_ZERO_ENV, "")).strip().lower() in _TRUTHY


def parse_idle_timeout_seconds(
    cfg_value: Any, default_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES
) -> float:
    """Coerce ``scale_to_zero.idle_timeout_minutes`` (config.yaml, D2) to seconds.

    Degrades to the default on any non-numeric / non-positive value (never raises,
    never returns <= 0 — a zero/negative timeout would make the gateway go dormant
    instantly, which is never the intent).
    """
    try:
        minutes = float(cfg_value)
    except (TypeError, ValueError):
        minutes = float(default_minutes)
    if minutes <= 0:
        minutes = float(default_minutes)
    return minutes * 60.0


def messaging_is_relay_only_or_absent(platforms: Iterable[Any]) -> bool:
    """True iff the only connected messaging platform is RELAY, or there is none
    (a Chronos-only / no-platform agent) — the F6/D1 structural precondition.

    A directly-connected platform (Discord/Telegram/Slack/...) holds a live
    socket and cannot scale to zero, so its presence disarms the feature. We
    compare by the platform's ``.value``/name to avoid importing the enum here
    (keeps this module import-light and unit-testable).
    """
    names = {_platform_name(p) for p in platforms}
    names.discard("relay")
    return len(names) == 0


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value).strip().lower()


def should_arm(
    *,
    enabled: bool,
    relay_only_or_absent: bool,
    wake_url: Optional[str],
) -> bool:
    """Whether to start the idle watcher at all (D1/D11/§3.4(1)).

    ALL must hold: the Labs flag is on, messaging is relay-only/absent, and a
    wakeUrl is registered (a suspended instance with no reachable wake target is
    a black hole — §3.4(1)). Any unmet -> the watcher never starts (no idle
    timer, no dormancy), so a non-opted instance behaves exactly as today.
    """
    return bool(enabled) and bool(relay_only_or_absent) and bool(wake_url)


def is_idle(
    *,
    active_work_count: int,
    seconds_since_last_inbound: float,
    idle_timeout_seconds: float,
    has_live_background_work: bool,
) -> bool:
    """The idle predicate (D2/D3/F7). Pure — composes the three conjuncts.

    Idle iff: no counted active work (in-flight agent turns + cron jobs +
    API-server runs — the caller aggregates every foreground work source),
    no inbound within the timeout window, and no live background work
    (backgrounded delegate_task / kanban / bg terminal). Any active work
    keeps the gateway awake — suspending mid-flight would lose it.

    ``active_work_count`` deliberately names the BROAD aggregate, not just
    agents: a caller passing only ``len(_running_agents)`` reopens the
    mid-cron-job suspend hole. Callers that cannot read a work source must
    fail AWAKE (pass a positive sentinel), never fail to 0.
    """
    if active_work_count > 0:
        return False
    if has_live_background_work:
        return False
    return seconds_since_last_inbound >= idle_timeout_seconds


# Dashboard-client liveness marker. The dashboard process (tui_gateway/ws.py,
# a DIFFERENT process from the gateway on hosted instances) touches this file
# on every /api/ws connect and inbound frame — the desktop app, web dashboard
# and TUI all send `gateway.ping` every 15s (apps/shared/src/json-rpc-gateway.ts,
# ui-tui/src/gatewayClient.ts). The gateway folds the mtime into its inbound
# clock, so an open client holds the box awake exactly like a chat message does
# and gets the same idle_timeout grace after it disconnects. Without this the
# box suspends under the open client, the client's reconnect loop re-pokes the
# Fly-proxied hostname, autostart resumes it, and the instance flaps every ~60s
# (13 of 72 active opted-in prod instances, 2026-09-02).
#
# There is deliberately NO staleness cutoff here: the mtime is a timestamp of
# real inbound, and is_idle already decides whether it is recent enough. A
# lingering marker cannot pin the box — once it is older than idle_timeout it
# no longer counts, exactly like an old _last_inbound_at.
DASHBOARD_CLIENT_HEARTBEAT_REL = os.path.join("state", "dashboard_clients.heartbeat")


def dashboard_client_heartbeat_path(hermes_home: Optional[os.PathLike | str] = None):
    """Path of the dashboard-client liveness marker under HERMES_HOME."""
    from pathlib import Path

    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    return Path(hermes_home) / DASHBOARD_CLIENT_HEARTBEAT_REL


def touch_dashboard_client_heartbeat(path: Optional[os.PathLike | str] = None) -> bool:
    """Mark "a dashboard client is attached right now". Best-effort, never raises."""
    try:
        p = dashboard_client_heartbeat_path() if path is None else path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8"):
            pass
        os.utime(p, None)
        return True
    except Exception:  # noqa: BLE001 - liveness garnish must never break the WS
        logger.debug("scale-to-zero: dashboard heartbeat touch failed", exc_info=True)
        return False


def dashboard_client_last_seen(
    path: Optional[os.PathLike | str] = None,
    *,
    now: Optional[float] = None,
) -> Optional[float]:
    """Epoch seconds a dashboard client last sent a WS frame, or None if never.

    Missing marker -> None (the steady state on a box nobody has the dashboard
    open on — NOT fail-awake, or every instance would never sleep). An
    unreadable marker -> ``now`` (fail-awake: an unreadable source counts as
    activity, same rule as the work counters in ``is_idle``).
    """
    import time

    p = dashboard_client_heartbeat_path() if path is None else path
    try:
        return os.stat(p).st_mtime
    except FileNotFoundError:
        return None
    except OSError:
        return time.time() if now is None else now


def self_suspend_available(environ: Optional[dict] = None) -> bool:
    """Whether this process can suspend its own machine via the flaps socket.

    True iff the Fly-injected machine identity is present AND the local Machines
    API socket exists. Off-Fly (local dev, Azure ACA, tests) this is False and
    the watcher skips the quiesce entirely: the platform owns the freeze, so
    the gateway stays connected until it lands.
    """
    env = environ if environ is not None else os.environ
    return bool(
        str(env.get(FLY_APP_NAME_ENV, "")).strip()
        and str(env.get(FLY_MACHINE_ID_ENV, "")).strip()
        and os.path.exists(FLY_API_SOCKET)
    )


def suspend_self(
    environ: Optional[dict] = None,
    *,
    socket_path: str = FLY_API_SOCKET,
    timeout: float = 10.0,
) -> bool:
    """POST /v1/apps/{app}/machines/{id}/suspend on the local flaps socket.

    Fly's in-machine Machines API needs no token — the socket itself is the
    credential. Equivalent to:
        curl --unix-socket /.fly/api -X POST \\
          http://flaps/v1/apps/$FLY_APP_NAME/machines/$FLY_MACHINE_ID/suspend

    Returns True when flaps accepted the request (2xx). The caller should treat
    this as fire-and-forget: on success the kernel freezes this process shortly
    after, so there may be nothing meaningful to run afterwards. Never raises —
    a failed suspend just leaves the machine running (fail-awake, never
    fail-frozen), which costs money but loses no work.

    stdlib-only on purpose: a plain unix-socket HTTP/1.1 request, no httpx/
    requests dependency in the hot path and no async plumbing to freeze
    mid-await.
    """
    env = environ if environ is not None else os.environ
    app = str(env.get(FLY_APP_NAME_ENV, "")).strip()
    machine_id = str(env.get(FLY_MACHINE_ID_ENV, "")).strip()
    if not app or not machine_id:
        logger.warning("scale-to-zero: suspend_self called without Fly machine identity")
        return False
    request = (
        f"POST /v1/apps/{app}/machines/{machine_id}/suspend HTTP/1.1\r\n"
        "Host: flaps\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            sock.sendall(request.encode("ascii"))
            response = b""
            while len(response) < 65536:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
    except OSError as exc:
        logger.warning("scale-to-zero: flaps suspend request failed: %s", exc)
        return False
    status_line = response.split(b"\r\n", 1)[0].decode("ascii", "replace")
    parts = status_line.split()
    ok = len(parts) >= 2 and parts[1].isdigit() and 200 <= int(parts[1]) < 300
    if ok:
        logger.info("scale-to-zero: machine suspend accepted by flaps (%s)", status_line)
    else:
        body = response.split(b"\r\n\r\n", 1)[-1][:500].decode("utf-8", "replace")
        logger.warning(
            "scale-to-zero: flaps suspend rejected: %s %s",
            status_line,
            json.dumps(body)[:500],
        )
    return ok
