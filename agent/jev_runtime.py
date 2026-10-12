"""Explicit opt-in for optional jev subprocess judgments."""

import logging

logger = logging.getLogger(__name__)


def is_jev_enabled(task: str) -> bool:
    """Reading a scoped config never enables external judgments by default."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly().get("jev", {})
        task_config = config.get(task, {}) if isinstance(config, dict) else {}
        return isinstance(task_config, dict) and task_config.get("enabled") is True
    # health: allow BLE001 -- Config failures must leave external judgments disabled; exception text may expose private config.
    except Exception as exc:
        logger.debug("jev opt-in lookup failed (%s)", type(exc).__name__)
        return False
