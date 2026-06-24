"""Telemetry consent posture and the aggregate-plane gate.

Consent is a single field, ``telemetry.consent_state``:

  * "unknown" — no choice recorded; never uploads (the default).
  * "local"   — declined the aggregate plane; local plane only.
  * "aggregate" — opted in to the aggregate plane.

The config file is the source of truth: set ``telemetry.consent_state`` with
``hermes config set`` (or a managed-scope pin). There is no separate boolean mirror —
a single field cannot drift out of sync with itself, so a stray value can't
accidentally imply consent.

``allow_aggregate`` is the hard gate. An administrator pins
``telemetry.allow_aggregate: false`` through the managed-scope layer
(``/etc/hermes/config.yaml``), which takes precedence over the user's config; when it
is false, the aggregate plane is off regardless of ``consent_state``.

This module makes the decisions; it performs no I/O and contains no uploader. A future
uploader must call :func:`may_upload_aggregate` at its boundary.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict

CONSENT_UNKNOWN = "unknown"
CONSENT_LOCAL = "local"
CONSENT_AGGREGATE = "aggregate"
_VALID_STATES = {CONSENT_UNKNOWN, CONSENT_LOCAL, CONSENT_AGGREGATE}


@dataclass(slots=True)
class TelemetryDecision:
    """The resolved telemetry posture for the current process."""
    local_enabled: bool
    aggregate_enabled: bool
    consent_state: str
    install_id: str
    allow_aggregate: bool

    def may_upload_aggregate(self) -> bool:
        """The single gate the uploader must consult before any network send."""
        return self.allow_aggregate and self.consent_state == CONSENT_AGGREGATE


def _telemetry_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config.get("telemetry") if isinstance(config, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def ensure_install_id(config: Dict[str, Any]) -> str:
    """Return a stable install id, minting one if the config slot is empty.

    Does not persist — the caller writes the returned value back to config.yaml. A
    fresh uuid4 is used; clearing ``telemetry.install_id`` (e.g. with
    ``hermes config set telemetry.install_id ""``) causes the next call to mint anew.
    """
    tel = _telemetry_cfg(config)
    existing = tel.get("install_id")
    if isinstance(existing, str) and existing.strip():
        return existing
    return str(uuid.uuid4())


def resolve(config: Dict[str, Any]) -> TelemetryDecision:
    """Resolve the effective telemetry posture from config.

    ``consent_state`` is the single source of truth for the aggregate opt-in.
    ``allow_aggregate`` (admin-pinnable via managed scope) hard-disables the aggregate
    plane regardless of consent.
    """
    tel = _telemetry_cfg(config)

    local_enabled = bool(tel.get("local", True))
    allow_aggregate = bool(tel.get("allow_aggregate", True))
    state = tel.get("consent_state", CONSENT_UNKNOWN)
    if state not in _VALID_STATES:
        state = CONSENT_UNKNOWN

    aggregate_enabled = allow_aggregate and state == CONSENT_AGGREGATE

    return TelemetryDecision(
        local_enabled=local_enabled,
        aggregate_enabled=aggregate_enabled,
        consent_state=state,
        install_id=ensure_install_id(config),
        allow_aggregate=allow_aggregate,
    )


def may_upload_aggregate(config: Dict[str, Any]) -> bool:
    """Convenience gate for the uploader boundary."""
    return resolve(config).may_upload_aggregate()


__all__ = [
    "CONSENT_UNKNOWN",
    "CONSENT_LOCAL",
    "CONSENT_AGGREGATE",
    "TelemetryDecision",
    "resolve",
    "may_upload_aggregate",
    "ensure_install_id",
]
