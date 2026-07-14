"""Install identity for gateway monitoring.

The install id is a stable, resettable pseudonymous identifier attached to
exported health signals so an operator can tell instances apart in their
collector. It carries no account identity and can be rotated by clearing
``monitoring.install_id`` in config.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict


def _monitoring_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("monitoring", "telemetry"):  # accept legacy telemetry.* keys
        cfg = config.get(key) if isinstance(config, dict) else None
        if isinstance(cfg, dict) and cfg.get("install_id"):
            return cfg
    cfg = config.get("monitoring") if isinstance(config, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def ensure_install_id(config: Dict[str, Any]) -> str:
    """Return a stable install id, minting one if the config slot is empty.

    Does not persist — the caller writes the returned value back to
    config.yaml. Clearing ``monitoring.install_id`` (e.g. with
    ``hermes config set monitoring.install_id ""``) mints anew on next call.
    """
    cfg = _monitoring_cfg(config)
    existing = cfg.get("install_id")
    if isinstance(existing, str) and existing.strip():
        return existing
    return str(uuid.uuid4())


__all__ = [
    "ensure_install_id",
]
