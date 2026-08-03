"""Shared gateway restart constants and supervisor detection helpers."""

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


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)
