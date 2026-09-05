"""Raw-YAML config and token/cost analytics dashboard routes.

Extracted from ``hermes_cli.web_server``; app state and helpers are late-bound through
:mod:`hermes_cli.web_deps` (cycle-safe, monkeypatch-friendly).
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_deps import late
from hermes_cli.web_server_profiles import (
    _approval_mode_of, _aux_task_summary, _aux_usage_rows, _broadcast_gateway_session_info, _is_other_profile,
)
from hermes_cli.web_models import RawConfigUpdate

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
save_config = late("save_config", "hermes_cli.config")

# ── Raw YAML config ──────────────────────────────────────────────────────────


@router.get("/api/config/raw")
async def get_config_raw(profile: Optional[str] = None):
    """Raw config.yaml text plus its resolved path.

    ``path`` is resolved inside ``_profile_scope`` so the Config page header
    shows the file the switched profile actually reads/writes — /api/status's
    ``config_path`` is machine-global and always reports the dashboard
    process's own profile, which is wrong under the global profile switcher.
    """
    def _run():
        with _profile_scope(profile):
            path = get_config_path()
        if not path.exists():
            return {"yaml": "", "path": str(path)}
        return {"yaml": path.read_text(encoding="utf-8"), "path": str(path)}

    return await asyncio.to_thread(_run)


@router.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate, profile: Optional[str] = None):
    def _run():
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        with _profile_scope(body.profile or profile):
            # Full-document replacement: the editor owns the whole file; never
            # merge omitted sections back from disk.
            # See #62723.
            approvals_mode_changed = _approval_mode_of(parsed) != _approval_mode_of(read_raw_config())
            save_config(parsed, merge_existing=False)
        # Same indicator refresh as the schema-driven save.
        if approvals_mode_changed and not _is_other_profile(body.profile or profile):
            _broadcast_gateway_session_info()
        return {"ok": True}

    try:
        return await asyncio.to_thread(_run)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


def _rows(db, sql: str, cutoff: float) -> List[Dict[str, Any]]:
    return [dict(r) for r in db._conn.execute(sql, (cutoff,)).fetchall()]


def _get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine
    from agent.usage_window import (
        aggregate_window_usage,
        get_usage_window_coverage,
        get_window_usage_rows,
    )

    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)
        usage_rows = get_window_usage_rows(db._conn, cutoff)
        usage_coverage = get_usage_window_coverage(
            db._conn, cutoff, usage_rows=usage_rows
        )
        daily, by_model, totals = aggregate_window_usage(usage_rows)
        aux_rows = _aux_usage_rows(db, cutoff, usage_rows)
        usage = InsightsEngine(db).get_usage_breakdown(days=days)

        return {
            "daily": daily,
            "by_model": by_model,
            # Aux-task summary across models (vision, compression, ...). Lets
            # the dashboard answer "what is compression costing me" directly.
            "by_task": _aux_task_summary(aux_rows),
            "totals": totals,
            "period_days": days,
            "usage_coverage": usage_coverage,
            "skills": usage["skills"],
            # Per-tool-name call counts (already computed by InsightsEngine);
            # the desktop Capabilities page aggregates these per toolset.
            "tools": usage["tools"],
        }
    finally:
        db.close()


@router.get("/api/analytics/usage")
async def get_usage_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """``days`` is clamped to 1-365 (idea from #74778): huge or non-positive
    values would force expensive full-history SQL and InsightsEngine work, or
    produce empty/inverted time windows. The UI only offers 7/30/90-day
    presets."""
    return await asyncio.to_thread(_get_usage_analytics, days, profile)


_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "api_calls", "tool_calls",
)


def _has_usage(row: Dict[str, Any]) -> bool:
    return any((row.get(key) or 0) != 0 for key in _USAGE_KEYS)


def _fold_session_only_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold model/task rows that carry no billing_provider and no usage into the single
    accounted provider row for that model/task.

    Session rows can be created before the first billable call finishes; if that early row
    records only the model and task while a later row has real accounting, the Models page
    used to show a duplicate "0 tokens / — API calls" card. Only folds when ownership is
    unambiguous (exactly one provider row).
    """
    rows_by_model_task: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        key = (row.get("model") or "", row.get("task") or "")
        rows_by_model_task.setdefault(key, []).append(row)

    rows: List[Dict[str, Any]] = []
    for model_rows in rows_by_model_task.values():
        provider_rows = [r for r in model_rows if r.get("billing_provider")]
        if len(provider_rows) != 1:
            rows.extend(model_rows)
            continue
        target = provider_rows[0]
        for row in model_rows:
            if row is target or row.get("billing_provider") or _has_usage(row):
                continue
            target["sessions"] = (target.get("sessions") or 0) + (row.get("sessions") or 0)
            target["last_used_at"] = max(target.get("last_used_at") or 0, row.get("last_used_at") or 0)
            total_tokens = (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
            sessions = target.get("sessions") or 0
            target["avg_tokens_per_session"] = total_tokens / sessions if sessions else 0
        rows.append(target)
        rows.extend(
            r for r in model_rows
            if r is not target and (r.get("billing_provider") or _has_usage(r))
        )
    return rows


def _model_capabilities(provider: str, model_name: str) -> dict:
    """models.dev capability metadata for the card; {} when unknown or lookup fails."""
    try:
        from agent.models_dev import get_model_capabilities
        mc = get_model_capabilities(provider=provider, model=model_name)
    except Exception:
        return {}
    if mc is None:
        return {}
    return {
        "supports_tools": mc.supports_tools,
        "supports_vision": mc.supports_vision,
        "supports_reasoning": mc.supports_reasoning,
        "context_window": mc.context_window,
        "max_output_tokens": mc.max_output_tokens,
        "model_family": mc.model_family,
    }


_AUX_SUMMED_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens", "estimated_cost", "sessions", "api_calls",
)
_MODEL_CARD_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "sessions", "api_calls", "tool_calls",
    "last_used_at", "avg_tokens_per_session",
)


def _get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Per-model token/cost/session breakdown plus models.dev capability metadata."""
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)

        from agent.usage_window import aggregate_model_usage, aggregate_window_usage, get_usage_window_coverage, get_window_usage_rows
        usage_rows = get_window_usage_rows(db._conn, cutoff)
        usage_coverage = get_usage_window_coverage(db._conn, cutoff, usage_rows=usage_rows)
        tool_rows = _rows(db, """SELECT COALESCE(model, 'unknown') AS model,
            COALESCE(billing_provider, '') AS billing_provider, SUM(COALESCE(tool_call_count, 0)) AS tool_calls
            FROM sessions WHERE started_at > ? GROUP BY model, billing_provider""", cutoff)
        raw_rows = aggregate_model_usage(usage_rows, tool_rows)
        _, _, totals = aggregate_window_usage(usage_rows)

        rows = _fold_session_only_rows(raw_rows)
        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = []
        for row in rows:
            model = {
                "model": row["model"],
                "provider": row.get("billing_provider") or "",
                **{key: row[key] for key in _MODEL_CARD_KEYS},
                "capabilities": _model_capabilities(row.get("billing_provider") or "", row["model"]),
            }
            # Keep the auxiliary task dimension visible when a task shares the
            # same model/provider as the main conversation route.
            if row.get("task"):
                model["aux_task"] = row["task"]
            models.append(model)

        totals["distinct_models"] = len({m["model"] for m in models})

        return {"models": models, "totals": totals, "period_days": days, "usage_coverage": usage_coverage}
    finally:
        db.close()


@router.get("/api/analytics/models")
async def get_models_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """Return model analytics without blocking the serving event loop."""
    return await asyncio.to_thread(_get_models_analytics, days, profile)
