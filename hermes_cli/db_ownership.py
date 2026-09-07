"""Policy decisions for direct Hermes session DB opens.

This module is intentionally side-effect free. Callers decide how to route a
non-ALLOW result; the policy itself never opens, closes, or modifies a DB.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Optional


class DbOpenDecision(Enum):
    """Action a caller must take before opening a session DB directly."""

    ALLOW = auto()
    SAFE_READ = auto()
    ROUTE_VIA_GATEWAY = auto()
    REFUSE = auto()


def decide_direct_db_open(
    *,
    role: str,
    operation: str,
    db_path: Path,
    gateway_pid: Optional[int],
    hermes_home: Optional[Path] = None,
) -> DbOpenDecision:
    """Return the safe policy for a direct session-DB open.

    ``gateway_pid`` is the caller's already-verified gateway liveness signal;
    this function deliberately does not inspect processes. ``hermes_home`` is
    supplied by test callers to make accidental canonical-home access fail
    closed. The function has no filesystem or process side effects.
    """
    if role not in {"gateway", "cli", "web", "tui", "maintenance", "test"}:
        raise ValueError(f"unknown DB owner role: {role}")
    if operation not in {"read", "write", "structural"}:
        raise ValueError(f"unknown DB operation: {operation}")

    path = Path(db_path).resolve()
    if role == "test":
        if hermes_home is not None:
            home = Path(hermes_home).resolve()
            if path == home / "state.db" or path.is_relative_to(home):
                raise RuntimeError(
                    "test DB access must use an isolated HERMES_HOME, not the canonical home"
                )
        return DbOpenDecision.ALLOW

    if role == "gateway":
        return DbOpenDecision.ALLOW

    if role == "maintenance":
        return DbOpenDecision.ALLOW if gateway_pid is None else DbOpenDecision.REFUSE

    if gateway_pid is not None:
        if operation == "read":
            return DbOpenDecision.SAFE_READ
        return DbOpenDecision.ROUTE_VIA_GATEWAY

    return DbOpenDecision.ALLOW
