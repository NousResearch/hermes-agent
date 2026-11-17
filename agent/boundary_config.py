"""Startup validation of a profile's boundary config (education design 5).

The guards themselves read config per call and fail closed, so nothing here
is needed for safety — this makes a broken block LOUD at gateway start
instead of at the first refused write: ``notion.write_guard``, every
``autonomy.actions`` level (a level an action has no code path for is
rejected, per the design's "levels only unlock capabilities that exist"),
the veto settings, and the Slack posting lists.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def boundary_config_errors(cfg) -> list[str]:
    if not isinstance(cfg, dict):
        return []
    errors: list[str] = []
    from tools.notion_write_guard import validate
    errors += validate(cfg)

    from agent import autonomy
    block = cfg.get("autonomy") if isinstance(cfg.get("autonomy"), dict) else {}
    actions = block.get("actions") if isinstance(block.get("actions"), dict) else {}
    for action in actions:
        try:
            autonomy.resolve_level(str(action), cfg)
        except autonomy.AutonomyConfigError as e:
            errors.append(str(e))
    if block:
        try:
            autonomy.veto_settings(cfg)
            autonomy.veto_expiry_hours(cfg)
        except autonomy.AutonomyConfigError as e:
            errors.append(str(e))

    from agent.slack_post_policy import post_policy
    policy = post_policy(cfg)
    if policy.error:
        errors.append(policy.error)
    elif policy.allowed is not None and policy.gated - policy.allowed:
        errors.append("slack.post_gated_channels lists channel(s) missing from "
                      f"slack.post_allowed_channels ({', '.join(sorted(policy.gated - policy.allowed))}); "
                      "posts there are refused.")
    return list(dict.fromkeys(errors))


def log_boundary_config_errors(profile_name: str, cfg: Optional[dict] = None) -> list[str]:
    """Log each error for ``profile_name`` (reads the active profile's
    config.yaml when ``cfg`` is None). Never raises."""
    try:
        if cfg is None:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config()
        errors = boundary_config_errors(cfg)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[boundary] profile '%s': could not validate boundary config: %s", profile_name, e)
        return []
    for err in errors:
        logger.error("[boundary] profile '%s': %s", profile_name, err)
    return errors
