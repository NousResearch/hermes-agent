"""Pure cron scheduler helpers owned by the fork."""

from __future__ import annotations

import os

from hermes_constants import parse_reasoning_effort, resolve_reasoning_config


def get_script_timeout(script_timeout, default_script_timeout: int, *, load_config, logger) -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if script_timeout != default_script_timeout:
        try:
            timeout = int(float(script_timeout))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", script_timeout)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return default_script_timeout


def resolve_cron_reasoning_config(job: dict, cfg, model: str) -> dict | None:
    """Resolve a cron job's per-job reasoning override, falling back to config."""
    reasoning_config = None
    job_effort = str(job.get("reasoning_effort") or "").strip()
    if job_effort:
        reasoning_config = parse_reasoning_effort(job_effort)
    if reasoning_config is None:
        reasoning_config = resolve_reasoning_config(cfg if isinstance(cfg, dict) else {}, str(model))
    return reasoning_config
