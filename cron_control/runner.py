"""Control-plane orchestration for cron shadow/evaluate/action cycles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config

from .actions import execute_verdict_actions
from .evaluator import evaluate_snapshot, persist_verdicts
from .shadow import collect_shadow_snapshot, persist_shadow_snapshot


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def control_plane_settings(config: dict[str, Any] | None = None) -> dict[str, bool]:
    cfg = config if isinstance(config, dict) else load_config() or {}
    cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
    control_cfg = cron_cfg.get("control_plane", {}) if isinstance(cron_cfg, dict) else {}
    enabled = _coerce_bool(os.getenv("HERMES_CRON_CONTROL_PLANE"), default=False)
    enabled = enabled or _coerce_bool(control_cfg.get("enabled"), default=False)
    approve_actions = _coerce_bool(os.getenv("HERMES_CRON_CONTROL_PLANE_APPROVE"), default=False)
    approve_actions = approve_actions or _coerce_bool(control_cfg.get("approve_actions"), default=False)
    persist_shadow = _coerce_bool(control_cfg.get("persist_shadow"), default=False)
    return {
        "enabled": enabled,
        "approve_actions": approve_actions,
        "persist_shadow": persist_shadow,
    }


def run_control_plane_cycle(
    *,
    jobs_path: Path | None = None,
    executions_path: Path | None = None,
    control_plane_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = control_plane_settings(config)
    if not settings["enabled"]:
        return {
            "enabled": False,
            "approve_actions": settings["approve_actions"],
            "persist_shadow": settings["persist_shadow"],
            "snapshot": None,
            "verdicts": [],
            "actions": [],
        }

    snapshot = collect_shadow_snapshot(
        jobs_path=jobs_path,
        executions_path=executions_path,
        control_plane_path=control_plane_path,
    )
    if settings["persist_shadow"]:
        persist_shadow_snapshot(snapshot, control_plane_path)
    verdicts = evaluate_snapshot(snapshot)
    if verdicts:
        persist_verdicts(snapshot, verdicts, control_plane_path)
    actions = execute_verdict_actions(
        verdicts,
        approved=settings["approve_actions"],
        control_plane_path=control_plane_path,
    )
    return {
        "enabled": True,
        "approve_actions": settings["approve_actions"],
        "persist_shadow": settings["persist_shadow"],
        "snapshot": snapshot,
        "verdicts": verdicts,
        "actions": actions,
    }
