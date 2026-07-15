"""Shared gateway restart constants and parsing helpers."""

import math

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG from sysexits.h — fatal configuration error (e.g. token
# collision, no messaging platforms).  The s6 finish script translates
# this into exit 125 (permanent failure) so the supervisor stops
# restarting the gateway.  See #51228.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)
DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT = float(
    DEFAULT_CONFIG["gateway"]["signal_interrupt_grace_timeout"]
)
DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT = 5.0


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


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
