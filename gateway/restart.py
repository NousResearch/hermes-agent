"""Shared gateway restart constants and supervisor detection helpers."""

import math
import os
from collections.abc import Mapping

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL (sysexits.h): ask the service manager to restart after a graceful drain/reload.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75
# EX_CONFIG (sysexits.h): fatal configuration error (token collision, no platforms);
# the s6 finish script maps it to exit 125 so the supervisor stops restarting.
# See #51228.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

# Set by ``hermes gateway run --external-supervisor``. Unlike systemd's INVOCATION_ID
# and launchd's XPC_SERVICE_NAME, this survives wrappers that replace the child
# environment (e.g. ``sudo env -i``).
EXTERNAL_GATEWAY_SUPERVISOR_ENV = "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(DEFAULT_CONFIG["agent"]["restart_drain_timeout"])
DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT = float(DEFAULT_CONFIG["gateway"]["signal_interrupt_grace_timeout"])
DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT = 5.0

# In-band restart waits for active turns to finish *before* ``stop()`` begins; distinct from
# ``restart_drain_timeout``, the force-interrupt budget once ``stop()`` runs (short under TimeoutStopSec).
DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT = float(DEFAULT_CONFIG["agent"]["restart_after_turn_timeout"])

# Cron-only floor under the ``stop()`` drain. ``restart_drain_timeout`` defaults to 0 because
# interrupting a *chat* turn is cheap and recoverable (user told, session resume_pending); an
# interrupted *cron* run is a permanent failure in jobs.json — a 0s drain silently destroys work.
DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT = float(DEFAULT_CONFIG["agent"]["cron_drain_timeout"])
# Watchdog leash held back for post-drain work (interrupt agents, kill subprocesses, mark jobs,
# disconnect adapters). Waiting for cron past that trades a job killed *and recorded* for one
# SIGKILLed mid-write and wedged at ``last_status=running`` forever.
CRON_DRAIN_CLEANUP_RESERVE_S = 10.0
# systemd TimeoutStopSec headroom after the stop-path drain budget, and the floor when that
# budget is still the default immediate (0s) chat drain. Keep in lockstep with generate_systemd_unit().
# See #94759.
SYSTEMD_STOP_HEADROOM_S = 30.0
SYSTEMD_TIMEOUT_STOP_SEC_FLOOR = 60.0

_TRUTHY = {"1", "true", "yes", "on"}


def is_global_startup_conflict(error_code: str | None) -> bool:
    """True when an adapter's fatal error is a single-writer ownership conflict.

    Adapters emit ``{scope}_lock`` with ``retryable=True`` so a *mid-run* reconnect can
    recover; at startup a live foreign holder is a configuration conflict (two gateways
    cannot poll one token), not a transient blip.  Matches by error CODE only, never text.

    ``BasePlatformAdapter._acquire_platform_lock`` emits ``{scope}_lock`` with ``retryable=True`` on
    purpose: a *mid-run* reconnect must be able to recover once the live holder exits or a stale record is
    cleared (#54167). This matches by error CODE only (the ``{scope}_lock`` / ``lock_conflict`` families
    every adapter emits for scoped-lock and identity conflicts), never by message text.
    """
    code = (error_code or "").strip().lower()
    return bool(code) and (code == "lock_conflict" or code.endswith("_lock"))


def is_gateway_supervisor_process(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this gateway process is owned by a supervisor."""
    env = os.environ if environ is None else environ
    xpc_service = env.get("XPC_SERVICE_NAME", "")
    return bool(env.get("INVOCATION_ID") or env.get("HERMES_S6_SUPERVISED_CHILD") or (xpc_service and xpc_service != "0")
                or str(env.get(EXTERNAL_GATEWAY_SUPERVISOR_ENV, "")).strip().lower() in _TRUTHY)


def is_container_restart_context() -> bool:
    """In a container (Docker/Podman) the detached setsid restart path dies with the cgroup,
    so exit-75 service restart is the only viable path.  Own function so tests can mock it."""
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _seconds(value: object, fallback: float = 0.0) -> float:
    """Non-negative float, or ``fallback`` on non-numeric input."""
    try:
        return max(float(value), 0.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _parse_timeout_keeping_zero(raw: object, default: float, *, finite: bool = False) -> float:
    """Parse a timeout where ``0`` is a deliberate disable (must NOT fall through
    to ``default``), unlike None / blank / non-numeric (/ non-finite) input."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if finite and not math.isfinite(value) else max(0.0, value)


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout; falsy (incl. ``0``) falls back to the shared default."""
    return _parse_timeout_keeping_zero(raw or None, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT)


def parse_restart_after_turn_timeout(raw: object) -> float:
    """Parse the after-turn wait cap for in-band restart (``0`` = legacy immediate drain)."""
    return _parse_timeout_keeping_zero(raw, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT)


def parse_cron_drain_timeout(raw: object) -> float:
    """Parse the cron-only drain floor (``0`` = opt out; cron interrupted on the chat budget).

    ``0`` is a deliberate opt-out — cron work is then interrupted on the same budget as chat work, the
    pre-#82161 behaviour — and must not fall through to the default, unlike empty/missing input.
    """
    return _parse_timeout_keeping_zero(raw, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT)


def parse_signal_interrupt_grace_timeout(raw: object) -> float:
    """Parse the unexpected-signal post-interrupt grace timeout."""
    return _parse_timeout_keeping_zero(raw, DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT, finite=True)


def resolve_cron_drain_budget(
    drain_timeout: float, cron_drain_timeout: float, *, watchdog_delay: float, elapsed: float = 0.0,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S,
) -> float:
    """Seconds the stop drain may wait on in-flight cron work.

    Clamped to what this process can honour: the watchdog hard-exits at ``watchdog_delay``,
    so waiting past that leash minus ``cleanup_reserve_s`` swaps a cleanly-interrupted job
    for a SIGKILL that leaves it wedged.  Never less than ``drain_timeout`` (only extends).
    """
    drain = _seconds(drain_timeout)
    floor = _seconds(cron_drain_timeout)
    if floor <= 0.0:
        return drain
    ceiling = _seconds(watchdog_delay) - _seconds(elapsed) - _seconds(cleanup_reserve_s, CRON_DRAIN_CLEANUP_RESERVE_S)
    return max(drain, min(floor, ceiling))


def resolve_systemd_timeout_stop_sec(
    drain_timeout: float, cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, *,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S, headroom_s: float = SYSTEMD_STOP_HEADROOM_S,
    floor_s: float = SYSTEMD_TIMEOUT_STOP_SEC_FLOOR,
) -> int:
    """Seconds systemd ``TimeoutStopSec`` must cover: the stop path may first wait
    ``cron_drain_timeout`` + ``cleanup_reserve_s`` for cron work, so sizing from the chat drain
    alone lets systemd SIGKILL an in-budget drain.  A zero cron timeout is an opt-out.

    ``restart_drain_timeout`` is only the chat-turn interrupt budget (default 0). See #94759.
    """
    drain = _seconds(drain_timeout)
    cron = _seconds(cron_drain_timeout)
    cron_budget = (cron + _seconds(cleanup_reserve_s)) if cron > 0.0 else 0.0
    return int(max(_seconds(floor_s), max(drain, cron_budget) + _seconds(headroom_s)))


def resolve_restart_exit_wait_budget(drain_timeout: float, after_turn_timeout: float, *, headroom: float = 15.0) -> float:
    """Seconds a CLI should wait for the gateway PID to exit after SIGUSR1: in-band restart may
    defer ``stop()`` until turns finish, then spend ``drain_timeout`` inside it — cover both."""
    return _seconds(drain_timeout) + _seconds(after_turn_timeout) + _seconds(headroom)
