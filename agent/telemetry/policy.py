"""Telemetry consent posture and the aggregate-metrics gate.

Consent is a single config field, ``telemetry.consent_state``:

  * "unknown" — no choice recorded; never uploads (the default).
  * "local"   — declined aggregate metrics; local telemetry only.
  * "aggregate" — opted in to aggregate metrics.

The config file is the source of truth: set ``telemetry.consent_state`` with
``hermes config set`` (or a managed-scope pin). Callers that gate behavior read
``telemetry.*`` directly from config; this module only provides the consent
constants, the install-id helper, and the upload gate a future uploader must
consult.

``allow_aggregate`` is the hard gate. An administrator pins
``telemetry.allow_aggregate: false`` through the managed-scope layer
(``/etc/hermes/config.yaml``), which takes precedence over the user's config; when
it is false, aggregate metrics are off regardless of ``consent_state``.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

CONSENT_UNKNOWN = "unknown"
CONSENT_LOCAL = "local"
CONSENT_AGGREGATE = "aggregate"
VALID_CONSENT_STATES = {CONSENT_UNKNOWN, CONSENT_LOCAL, CONSENT_AGGREGATE}


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


def may_upload_aggregate(config: Dict[str, Any]) -> bool:
    """Whether aggregate metrics may upload — the gate a future uploader consults.

    True only when the admin hard gate allows it AND the user has opted in via
    ``telemetry.consent_state``.
    """
    tel = _telemetry_cfg(config)
    allow_aggregate = bool(tel.get("allow_aggregate", True))
    state = tel.get("consent_state", CONSENT_UNKNOWN)
    return allow_aggregate and state == CONSENT_AGGREGATE


__all__ = [
    "CONSENT_UNKNOWN",
    "CONSENT_LOCAL",
    "CONSENT_AGGREGATE",
    "VALID_CONSENT_STATES",
    "may_upload_aggregate",
    "ensure_install_id",
]
