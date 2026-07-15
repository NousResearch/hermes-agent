"""Shared gateway restart constants and supervisor detection helpers."""

import math
import os
from collections.abc import Mapping

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG from sysexits.h — fatal configuration error (e.g. token
# collision, no messaging platforms).  The s6 finish script translates
# this into exit 125 (permanent failure) so the supervisor stops
# restarting the gateway.  See #51228.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

# Set by ``hermes gateway run --external-supervisor``. Unlike systemd's
# INVOCATION_ID and launchd's XPC_SERVICE_NAME, this survives wrappers that
# intentionally replace the child environment (for example ``sudo env -i``).
EXTERNAL_GATEWAY_SUPERVISOR_ENV = "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)
DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT = float(
    DEFAULT_CONFIG["gateway"]["signal_interrupt_grace_timeout"]
)
DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT = 5.0

# In-band restart (``/restart``, SIGUSR1, self-restart from a child CLI)
# waits for active turns to finish *before* ``stop()`` begins. Distinct
# from ``restart_drain_timeout``, which is the force-interrupt budget
# once ``stop()`` is running (and must stay short under systemd
# TimeoutStopSec). See #77184.
DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_after_turn_timeout"]
)

# Cron-only floor under the ``stop()`` drain. ``restart_drain_timeout``
# defaults to 0 because interrupting a *chat* turn is cheap and recoverable:
# the user is told the gateway is restarting and the session is pre-marked
# resume_pending. An interrupted *cron* run has neither property — nobody is
# waiting on it, it lands in jobs.json as a permanent failure, and a recurring
# job just waits for its next schedule — so a zero-second drain silently
# destroys work. See #82161.
DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["cron_drain_timeout"]
)

# Seconds of the shutdown watchdog leash held back for the work that still has
# to happen after the drain returns: interrupt agents, kill tool subprocesses,
# mark in-flight jobs interrupted, disconnect adapters. Waiting for cron past
# that point trades a job that is killed *and recorded* for one that is
# SIGKILLed mid-write and stays wedged at ``last_status=running`` forever.
CRON_DRAIN_CLEANUP_RESERVE_S = 10.0

# systemd TimeoutStopSec headroom after the stop-path drain budget, and the
# floor used when that budget is still the default immediate (0s) chat drain.
# Keep these in lockstep with generate_systemd_unit() / #94759.
SYSTEMD_STOP_HEADROOM_S = 30.0
SYSTEMD_TIMEOUT_STOP_SEC_FLOOR = 60.0


def is_gateway_supervisor_process(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this gateway process is owned by a supervisor."""
    env = os.environ if environ is None else environ
    if env.get("INVOCATION_ID"):
        return True
    if env.get("HERMES_S6_SUPERVISED_CHILD"):
        return True
    xpc_service = env.get("XPC_SERVICE_NAME", "")
    if xpc_service and xpc_service != "0":
        return True
    return str(env.get(EXTERNAL_GATEWAY_SUPERVISOR_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_container_restart_context() -> bool:
    """Return whether the gateway is running inside a container for restart
    routing purposes (Docker/Podman ⇒ the detached setsid path dies with the
    cgroup; exit-75 service restart is the only viable path).

    Extracted from the inline probe in the /restart handler so tests can mock
    container detection hermetically — a real ``/.dockerenv`` on a
    containerized CI runner otherwise flips the routing under the test.
    """
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def parse_restart_after_turn_timeout(raw: object) -> float:
    """Parse the after-turn wait cap for in-band restart, falling back to default.

    ``0`` is a deliberate disable (legacy immediate drain) and must not fall
    through to the default — unlike empty/missing input.
    """
    if raw is None:
        return DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    if isinstance(raw, str) and not raw.strip():
        return DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    return max(0.0, value)


def parse_cron_drain_timeout(raw: object) -> float:
    """Parse the cron-only drain floor, falling back to the shared default.

    ``0`` is a deliberate opt-out — cron work is then interrupted on the same
    budget as chat work, the pre-#82161 behaviour — and must not fall through
    to the default, unlike empty/missing input.
    """
    if raw is None:
        return DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
    if isinstance(raw, str) and not raw.strip():
        return DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
    return max(0.0, value)


def resolve_cron_drain_budget(
    drain_timeout: float,
    cron_drain_timeout: float,
    *,
    watchdog_delay: float,
    elapsed: float = 0.0,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S,
) -> float:
    """Seconds the shutdown drain may spend waiting on in-flight cron work.

    The configured floor is clamped to what this process can actually honour.
    The shutdown watchdog hard-exits at ``watchdog_delay`` and the service
    manager's ``TimeoutStopSec`` is sized from the full stop budget (drain
    vs cron floor + cleanup reserve, plus headroom — see
    ``resolve_systemd_timeout_stop_sec``), so waiting past that leash
    (minus ``cleanup_reserve_s`` for the teardown that follows the drain)
    would swap a cleanly-interrupted job for a SIGKILL that leaves it
    wedged mid-run — strictly worse than the bug being fixed.

    Never returns less than ``drain_timeout``: the cron floor only ever
    extends the wait, so an operator who deliberately configured a long
    ``restart_drain_timeout`` keeps it.
    """

    def _seconds(value: object, fallback: float = 0.0) -> float:
        try:
            return max(float(value), 0.0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return fallback

    drain = _seconds(drain_timeout)
    floor = _seconds(cron_drain_timeout)
    if floor <= 0.0:
        return drain
    ceiling = (
        _seconds(watchdog_delay)
        - _seconds(elapsed)
        - _seconds(cleanup_reserve_s, CRON_DRAIN_CLEANUP_RESERVE_S)
    )
    return max(drain, min(floor, ceiling))


def resolve_systemd_timeout_stop_sec(
    drain_timeout: float,
    cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    *,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S,
    headroom_s: float = SYSTEMD_STOP_HEADROOM_S,
    floor_s: float = SYSTEMD_TIMEOUT_STOP_SEC_FLOOR,
) -> int:
    """Seconds systemd ``TimeoutStopSec`` must cover the full stop budget.

    ``restart_drain_timeout`` is only the chat-turn interrupt budget (default
    0). The stop path may wait longer for in-flight cron work —
    ``cron_drain_timeout`` plus ``cleanup_reserve_s`` — before it even starts
    interrupting. Sizing the unit from drain alone lets systemd SIGKILL an
    in-budget drain (#94759).

    A zero ``cron_drain_timeout`` is a deliberate opt-out and does not extend
    the budget. Non-numeric inputs degrade to 0 rather than raising.
    """

    def _seconds(value: object) -> float:
        try:
            return max(float(value), 0.0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    drain = _seconds(drain_timeout)
    cron = _seconds(cron_drain_timeout)
    reserve = _seconds(cleanup_reserve_s)
    headroom = _seconds(headroom_s)
    floor = _seconds(floor_s)
    cron_budget = (cron + reserve) if cron > 0.0 else 0.0
    stop_budget = max(drain, cron_budget)
    return int(max(floor, stop_budget + headroom))


def resolve_restart_exit_wait_budget(
    drain_timeout: float,
    after_turn_timeout: float,
    *,
    headroom: float = 15.0,
) -> float:
    """Seconds a CLI should wait for the gateway PID to exit after SIGUSR1.

    In-band restart may defer ``stop()`` until active turns finish
    (``after_turn_timeout``) and then spend up to ``drain_timeout`` inside
    ``stop()``. Callers that fall back to a hard kill on wait expiry must
    cover both phases or they reintroduce #77184.
    """
    try:
        drain = max(float(drain_timeout), 0.0)
    except (TypeError, ValueError):
        drain = 0.0
    try:
        after_turn = max(float(after_turn_timeout), 0.0)
    except (TypeError, ValueError):
        after_turn = 0.0
    try:
        margin = max(float(headroom), 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    return drain + after_turn + margin


def parse_signal_interrupt_grace_timeout(raw: object) -> float:
    """Parse the unexpected-signal post-interrupt grace timeout."""
    try:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            value = DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
        else:
            value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    if not math.isfinite(value):
        return DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    return max(0.0, value)
