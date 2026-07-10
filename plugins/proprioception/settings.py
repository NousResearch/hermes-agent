"""Config access for the proprioception plugin.

One config block in ``~/.hermes/config.yaml`` controls everything::

    proprioception:
      enabled: true                # master switch — absent/false = plugin inert
      heartbeat: delta             # off | delta | always (see cost note below)
      min_interval_seconds: 60     # per-session floor between non-critical emissions
      cache_ttl_seconds: 30        # process-wide snapshot cache
      stale_grace_seconds: 90      # serve last-good dashboard data this long on fetch failure
      timeout_seconds: 1.5         # per-source fetch budget
      dashboard_url: http://127.0.0.1:8787/api/home
      context_window: 131072       # denominator for context-fill buckets
      max_chars: 700               # heartbeat hard truncation
      gap_report_seconds: 1800     # report a suspension gap once idle wall-time exceeds this
      clock: false                 # opt-in: emit a compact time+delta EVERY turn (temporal
                                   #   grounding; trades prefix-cache reuse for it — off by default)

Cost note on ``heartbeat: always``: every emission makes the current
user message diverge from what history replays next turn, so the
previous turn's assistant/tool tail is re-prefilled once per emission.
``delta`` keeps that rare; ``always`` pays it EVERY turn — on tool-heavy
local sessions that can cut prefix-cache reuse dramatically. ``always``
is a diagnostic mode, not a daily driver.

Everything is fail-closed: missing block, malformed values, or an
unreadable config all resolve to ``enabled: false``. A config read
failure is logged at WARNING once per process (the feature silently
disabling itself is otherwise invisible).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "heartbeat": "delta",
    "min_interval_seconds": 60,
    "cache_ttl_seconds": 30,
    "stale_grace_seconds": 90,
    "timeout_seconds": 1.5,
    "dashboard_url": "http://127.0.0.1:8787/api/home",
    "context_window": 131072,
    "max_chars": 700,
    "gap_report_seconds": 1800,
    "clock": False,
}

_warned_config_failure = False

_VALID_HEARTBEAT_MODES = {"off", "delta", "always"}


def get_settings() -> Dict[str, Any]:
    """Return the effective proprioception settings (defaults + config.yaml).

    Never raises; any failure returns the defaults (which are disabled).
    """
    global _warned_config_failure
    cfg: Dict[str, Any] = dict(DEFAULTS)
    try:
        from hermes_cli.config import load_config_readonly

        raw = load_config_readonly().get("proprioception", {})
        if not isinstance(raw, dict):
            return cfg
        for key in DEFAULTS:
            if key in raw and raw[key] is not None:
                cfg[key] = raw[key]
    except Exception:
        if not _warned_config_failure:
            _warned_config_failure = True
            logger.warning(
                "proprioception: config read failed — feature disabled until "
                "config.yaml is readable again",
                exc_info=True,
            )
        return dict(DEFAULTS)

    # Sanitize: wrong types must not crash the turn prologue.
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["clock"] = bool(cfg["clock"])
    if str(cfg["heartbeat"]).lower() not in _VALID_HEARTBEAT_MODES:
        cfg["heartbeat"] = "delta"
    else:
        cfg["heartbeat"] = str(cfg["heartbeat"]).lower()
    for numeric_key, floor in (
        ("min_interval_seconds", 0),
        ("cache_ttl_seconds", 1),
        ("stale_grace_seconds", 0),
        ("timeout_seconds", 0.2),
        ("context_window", 1024),
        ("max_chars", 100),
        ("gap_report_seconds", 0),
    ):
        try:
            value = float(cfg[numeric_key])
        except (TypeError, ValueError):
            value = float(DEFAULTS[numeric_key])
        cfg[numeric_key] = max(floor, value)
    cfg["context_window"] = int(cfg["context_window"])
    cfg["max_chars"] = int(cfg["max_chars"])
    return cfg


def is_enabled() -> bool:
    """Cheap master-switch probe used by the tool's check_fn and the hook."""
    return bool(get_settings()["enabled"])
