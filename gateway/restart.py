"""Shared gateway restart constants, parsing helpers, and safety gates."""

import os
from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG, get_hermes_home

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)

GATEWAY_RESTART_APPROVAL_REQUIRED_ENV = "HERMES_GATEWAY_RESTART_REQUIRES_APPROVAL"
GATEWAY_RESTART_APPROVED_ENV = "HERMES_GATEWAY_RESTART_APPROVED"
GATEWAY_RESTART_APPROVAL_REQUIRED_MARKER = ".gateway_restart_approval_required"


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def _truthy(raw: object) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "approved"}


def gateway_restart_approval_required(hermes_home: Path | None = None) -> bool:
    """Return True when local policy requires an explicit restart approval."""
    if _truthy(os.getenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV)):
        return True
    try:
        home = hermes_home or get_hermes_home()
        return (Path(home) / GATEWAY_RESTART_APPROVAL_REQUIRED_MARKER).exists()
    except Exception:
        return False


def gateway_restart_approved(*, approved: bool = False) -> bool:
    """Return True when this restart call carries an explicit approval override."""
    return bool(approved) or _truthy(os.getenv(GATEWAY_RESTART_APPROVED_ENV))


def gateway_restart_approval_message(source: str = "gateway restart") -> str:
    return (
        f"Refusing {source}: explicit approval is required by local policy. "
        "Ask the user for approval first, then rerun with --approved or set "
        f"{GATEWAY_RESTART_APPROVED_ENV}=1 for the single approved command. "
        "Emergency recovery can still bypass this by setting the same override."
    )


def require_gateway_restart_approval(
    *,
    source: str = "gateway restart",
    approved: bool = False,
    hermes_home: Path | None = None,
) -> None:
    """Raise PermissionError when restart policy requires approval and none was supplied."""
    if gateway_restart_approval_required(hermes_home) and not gateway_restart_approved(approved=approved):
        raise PermissionError(gateway_restart_approval_message(source))
