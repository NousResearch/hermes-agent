"""Shared gateway restart constants and supervisor detection helpers."""

import os
from collections.abc import Mapping

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG from sysexits.h — fatal configuration error (e.g. token
# collision / single-writer lock conflict).  The s6 finish script
# translates this into exit 125 (permanent failure) so the supervisor
# stops restarting the gateway.  See #51228.
#
# Platform-local auth/config failures (missing membership, unpaired
# bridge, missing adapter credentials) are NOT EX_CONFIG: they park the
# affected adapter and leave cron/Kanban/other platforms running.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78


def _message_looks_like_polling_conflict(message: str) -> bool:
    """Return True when *message* matches a Telegram getUpdates single-writer 409.

    Production Telegram connect can surface a live foreign poller as a generic
    ``telegram_connect_error`` whose body still carries the Bot API 409 text
    (``Conflict: terminated by other getUpdates request``). Classification must
    recognize that payload even when the structured code was not upgraded to
    ``telegram_polling_conflict``.
    """
    msg = (message or "").lower()
    if not msg:
        return False
    if "terminated by other getupdates request" in msg:
        return True
    if "another bot instance is running" in msg:
        return True
    if "another getupdates request" in msg:
        return True
    # Narrow 409 + getUpdates pairing avoids treating unrelated HTTP 409s as
    # exclusive-ownership conflicts.
    if "409" in msg and "getupdates" in msg:
        return True
    return False


def is_global_startup_conflict(
    error_code: str | None = None,
    message: str | None = None,
) -> bool:
    """Return True when a platform failure is a gateway-global single-writer conflict.

    Production token locks (Discord/Telegram/Slack/WhatsApp/Signal) are
    emitted by ``BasePlatformAdapter._acquire_platform_lock`` with
    ``retryable=True`` so a *later* reconnect can recover after a live
    holder exits. At zero-connected startup, however, those failures are
    still exclusive-ownership conflicts and must force
    ``GATEWAY_FATAL_CONFIG_EXIT_CODE`` — classification inspects the
    conflict semantics (code/message), not the retryable flag alone.

    Telegram production 409 getUpdates conflicts must also fail closed even
    when the adapter only recorded a generic ``telegram_connect_error``.

    Platform-local auth/config errors (Buzz membership, missing
    credentials, unpaired bridges) must stay isolated so optional
    adapters cannot take down unrelated gateway duties.
    """
    code = (error_code or "").strip().lower()
    msg = message or ""
    if code:
        if code == "lock_conflict":
            return True
        if code.endswith("_lock"):
            return True
        if "polling_conflict" in code:
            return True
        # Generic Telegram connect envelope still carrying a production 409.
        if code == "telegram_connect_error" and _message_looks_like_polling_conflict(msg):
            return True
        return False

    msg_l = msg.lower()
    if _message_looks_like_polling_conflict(msg_l):
        return True
    if "already in use" not in msg_l:
        return False
    return any(
        marker in msg_l
        for marker in (
            "stop the other gateway",
            "another local hermes",
            "another profile",
            "pid ",
        )
    )


# Set by ``hermes gateway run --external-supervisor``. Unlike systemd's
# INVOCATION_ID and launchd's XPC_SERVICE_NAME, this survives wrappers that
# intentionally replace the child environment (for example ``sudo env -i``).
EXTERNAL_GATEWAY_SUPERVISOR_ENV = "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)

# In-band restart (``/restart``, SIGUSR1, self-restart from a child CLI)
# waits for active turns to finish *before* ``stop()`` begins. Distinct
# from ``restart_drain_timeout``, which is the force-interrupt budget
# once ``stop()`` is running (and must stay short under systemd
# TimeoutStopSec). See #77184.
DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_after_turn_timeout"]
)


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
