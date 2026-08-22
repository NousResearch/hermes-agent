"""Jarvis Phase 1 inherited-environment credential boundary.

This module is deliberately stdlib-only and safe to import before dotenv,
configuration, plugin, provider, or credential-store modules. The decision is
snapshotted once at import time so a later dotenv load or in-process mutation
cannot change the boundary after startup.
"""

from __future__ import annotations

import os


PHASE1_CAPABILITY_MODE_ENV = "HERMES_PHASE1_CAPABILITY_MODE"


class Phase1CapabilityModeError(RuntimeError):
    """Raised when the Phase 1 boundary is invalid or would be crossed."""


def _parse_phase1_capability_mode(raw: str | None) -> bool:
    if raw is None or raw.strip().lower() in {"", "0", "false", "no", "off"}:
        return False
    if raw.strip() == "1":
        return True
    raise Phase1CapabilityModeError(
        f"{PHASE1_CAPABILITY_MODE_ENV} must be 1 when enabled "
        "(or unset/0/false/no/off when disabled)"
    )


_PHASE1_CAPABILITY_MODE = _parse_phase1_capability_mode(
    os.environ.get(PHASE1_CAPABILITY_MODE_ENV)
)


def phase1_capability_mode_enabled() -> bool:
    """Return the immutable startup decision for this process."""
    return _PHASE1_CAPABILITY_MODE


def require_persistent_credential_expansion_allowed(source: str) -> None:
    """Fail closed if *source* would expand inherited capabilities."""
    if _PHASE1_CAPABILITY_MODE:
        raise Phase1CapabilityModeError(
            "Jarvis Phase 1 capability mode forbids persistent credential "
            f"expansion from {source}"
        )
