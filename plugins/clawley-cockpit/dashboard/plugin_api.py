"""Clawley cockpit dashboard plugin — read-only backend API routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()


def build_status_snapshot() -> dict[str, Any]:
    """Build a side-effect-free local status snapshot for the dashboard tab."""

    quantos_perf = _load_optional_json(os.environ.get("CLAWLEY_QUANTOS_CHANNEL_PERFORMANCE"))
    return {
        "schema": "clawley_cockpit_status.v1",
        "write_performed": False,
        "sections": {
            "quantos": _quantos_section(quantos_perf),
            "gateway": {"status": "available", "source": "dashboard_plugin_runtime"},
            "cron": {"status": "not_mutated", "next_action": "wire_daily_brief_collector_when_ready"},
            "kanban": {"status": "proposal_only", "github_mutation_allowed": False},
            "maintainer_sweep": {"status": "proposal_only", "deterministic_apply_required": True},
        },
        "safety_flags": {
            "read_only": True,
            "write_performed": False,
            "github_mutation_allowed": False,
            "broker_order_submitted": False,
            "live_trading": False,
            "secrets_redacted": True,
        },
    }


def _quantos_section(perf: dict[str, Any] | None) -> dict[str, Any]:
    if perf is None:
        return {"status": "not_configured", "next_action": "set_CLAWLEY_QUANTOS_CHANNEL_PERFORMANCE"}
    raw_totals = perf.get("totals")
    totals: dict[str, Any] = raw_totals if isinstance(raw_totals, dict) else {}
    return {
        "status": "available",
        "source": perf.get("source", "unknown"),
        "setups_total": totals.get("setups_total", 0),
        "resolved_winrate": totals.get("resolved_winrate"),
        "resolved_expectancy_r": totals.get("resolved_expectancy_r"),
        "next_safe_action": perf.get("next_safe_action", "review_channel_performance_before_any_paper_bridge"),
    }


def _load_optional_json(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    try:
        payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@router.get("/status")
async def status() -> dict[str, Any]:
    return build_status_snapshot()
