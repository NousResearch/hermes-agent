"""Config for the chronoception plugin (read from config.yaml, fail-closed).

    chronoception:
      enabled: false            # master switch — absent/false = plugin inert
      clock: true               # emit a compact time+delta line EVERY turn (temporal
                                #   grounding; trades prefix-cache reuse for it). false =
                                #   only warn on a long idle gap (rare, cache-cheap).
      gap_report_seconds: 1800  # warn once on resume after this much real idle time
      max_chars: 400            # hard truncation of the injected block

Every failure mode (missing block, malformed values, unreadable config) resolves
to ``enabled: false``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "clock": True,
    "gap_report_seconds": 1800,
    "max_chars": 400,
}

_warned_config_failure = False


def get_settings() -> Dict[str, Any]:
    """Return effective settings (defaults + config.yaml). Never raises."""
    global _warned_config_failure
    cfg: Dict[str, Any] = dict(DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly

        raw = load_config_readonly().get("chronoception", {})
        if not isinstance(raw, dict):
            return cfg
        for key in DEFAULTS:
            if key in raw and raw[key] is not None:
                cfg[key] = raw[key]
    except Exception:
        if not _warned_config_failure:
            _warned_config_failure = True
            logger.warning("chronoception: config read failed — disabled", exc_info=True)
        return dict(DEFAULTS)

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["clock"] = bool(cfg["clock"])
    for key, floor in (("gap_report_seconds", 0), ("max_chars", 200)):
        try:
            value = float(cfg[key])
        except (TypeError, ValueError):
            value = float(DEFAULTS[key])
        cfg[key] = max(floor, value)
    cfg["max_chars"] = int(cfg["max_chars"])
    return cfg


def is_enabled() -> bool:
    return bool(get_settings()["enabled"])
