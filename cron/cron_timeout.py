"""Resolve cron inactivity timeout (seconds) from env and config."""

from __future__ import annotations

import os
from typing import Any, Mapping


def resolve_cron_inactivity_timeout_seconds(cfg: Mapping[str, Any] | None) -> float:
    """HERMES_CRON_TIMEOUT overrides config; 0 means unlimited.

    Default 600s matches historical behavior when neither env nor config is set.
    """
    env_val = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    if cfg and isinstance(cfg.get("cron"), dict):
        raw = cfg["cron"].get("inactivity_timeout_seconds")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return 600.0
