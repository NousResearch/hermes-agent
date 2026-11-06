"""Configuration readers for task-scoped auxiliary routing and retries."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("agent.auxiliary_client")
_DEFAULT_TRANSIENT_RETRIES = 2


def get_openrouter_settings(
    task: Optional[str] = None,
    fallback_model: str = "",
) -> tuple[bool, str]:
    """Resolve the global free-only policy plus an explicitly scoped MoA exception."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
        free_only = bool(cfg_get(cfg, "auxiliary", "free_only", default=False))
        if task == "moa_reference":
            task_override = cfg_get(cfg, "auxiliary", "moa_reference", "free_only", default=None)
            if isinstance(task_override, bool):
                free_only = task_override
        configured_model = cfg_get(cfg, "auxiliary", "openrouter_model")
        model = (
            configured_model.strip()
            if isinstance(configured_model, str) and configured_model.strip()
            else fallback_model
        )
        return free_only, model
    except Exception:
        logger.exception("Auxiliary OpenRouter policy unavailable; preserving the free-only guard")
        return True, fallback_model


def get_transient_retry_count(task: Optional[str] = None) -> int:
    """Resolve the task override or global retry count, clamped to [0, 6]."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        value = cfg_get(cfg, "auxiliary", task, "transient_retries", default=None) if task else None
        if value is None:
            value = cfg_get(cfg, "auxiliary", "transient_retries")
        return _DEFAULT_TRANSIENT_RETRIES if value is None else max(0, min(int(value), 6))
    except Exception:
        logger.exception("Auxiliary retry policy unavailable; using the default retry count")
        return _DEFAULT_TRANSIENT_RETRIES
