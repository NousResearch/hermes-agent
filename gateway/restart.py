"""Shared gateway restart constants and parsing helpers."""

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG from sysexits.h — used when the gateway exits due to a fatal
# configuration error (e.g. a token collision or no messaging platforms
# configured). Distinct from GATEWAY_SERVICE_RESTART_EXIT_CODE (75) so the
# s6 finish script (see S6ServiceManager._render_finish_script) can tell
# "please restart me" apart from "config is broken, don't restart" and
# stop s6-supervise from looping forever on a config that will never fix
# itself. See #51228.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)
