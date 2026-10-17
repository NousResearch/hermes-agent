"""Raw-YAML config and token/cost analytics dashboard routes.

Extracted from ``hermes_cli.web_server``; app state and helpers are late-bound through
:mod:`hermes_cli.web_deps` (cycle-safe, monkeypatch-friendly).
"""

import asyncio
import sqlite3
import time
from typing import Any, Dict, List, Optional

import hermes_yaml as yaml
from fastapi import APIRouter, HTTPException, Query

from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import corrupt_store_as_status
from hermes_cli.web_server_profiles import (
    _approval_mode_of, _aux_task_summary, _aux_usage_rows, _broadcast_gateway_session_info, _is_other_profile, _merge_aux_into_by_model,
)
from hermes_cli.web_models import RawConfigUpdate

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_session_db_path_for_profile = late("_session_db_path_for_profile", "hermes_cli.web_server_sessions")
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
        return {"yaml": path.read_text(encoding="utf-8-sig"), "path": str(path)}

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


def _rows(db, sql: str, cutoff: float) -> list[dict[str, Any]]:
    return [dict(r) for r in db._conn.execute(sql, (cutoff,)).fetchall()]


_PROVIDER_USAGE_FIELDS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_write_tokens": "cache_write_tokens",
    "reasoning_tokens": "reasoning_tokens",
    "estimated_cost": "estimated_cost_usd",
    "actual_cost": "actual_cost_usd",
    "api_calls": "api_call_count",
}


def _provider_residual_expression(key: str, column: str) -> str:
    total = f"COALESCE(s.{column}, 0)"
    attributed = f"COALESCE(t.{key}, 0)"
    residual = f"MAX(0, {total} - {attributed})"
    if key in ("estimated_cost", "actual_cost"):
        # Session and route sums can add floats in a different order. Roundoff
        # alone must not create a phantom provider/session after a route switch.
        # A relative tolerance still preserves genuinely tiny cost-only usage.
        tolerance = f"MAX(ABS({total}), ABS({attributed})) * 1e-12"
        residual = f"CASE WHEN {residual} <= {tolerance} THEN 0 ELSE {residual} END"
    return f"{residual} AS {key}"


def _get_provider_usage(db, cutoff: float) -> List[Dict[str, Any]]:
    """Attribute recorded usage to providers, including add-only auxiliary calls.

    Session counters include primary turns (including ``voice_chat``), but not
    record_auxiliary_usage calls. Only primary detail can cover a session's
    residual; subtracting auxiliary detail here would silently discard usage.
    One SQL statement keeps attribution and distinct-session counts consistent.
    """
    # Read-only dashboards may attach an older schema. Only these known schema
    # omissions are compatible; malformed stores and unrelated SQL errors must
    # still reach the route's existing error handling.
    columns = {row["name"] for row in db._conn.execute("PRAGMA table_info(session_model_usage)")}
    if columns:
        # Voice turns use update_token_counts, unlike auxiliary side calls.
        primary = "COALESCE(u.task, '') IN ('', 'voice_chat')" if "task" in columns else "1"
        values = ", ".join(f"COALESCE(u.{column}, 0) AS {key}" for key, column in _PROVIDER_USAGE_FIELDS.items())
        detail_sql = f"""
            SELECT u.session_id,
                   COALESCE(NULLIF(u.billing_provider, ''), 'unknown') AS provider,
                   {primary} AS is_primary, {values}
            FROM session_model_usage u JOIN selected_sessions s ON s.id = u.session_id
        """
    else:
        # Empty detail source: every session's usage becomes a legacy residual.
        values = ", ".join(f"0 AS {key}" for key in _PROVIDER_USAGE_FIELDS)
        detail_sql = f"SELECT id AS session_id, '' AS provider, 1 AS is_primary, {values} FROM selected_sessions WHERE 0"

    sums = ", ".join(f"SUM({key}) AS {key}" for key in _PROVIDER_USAGE_FIELDS)
    residuals = ", ".join(
        _provider_residual_expression(key, column)
        for key, column in _PROVIDER_USAGE_FIELDS.items()
    )
    keys = ", ".join(_PROVIDER_USAGE_FIELDS)
    has_usage = " OR ".join(f"{key} > 0" for key in _PROVIDER_USAGE_FIELDS)
    return _rows(db, f"""
        WITH selected_sessions AS (
            SELECT * FROM sessions WHERE started_at > ?
        ), route_usage AS (
            {detail_sql}
        ), primary_totals AS (
            SELECT session_id, {sums}
            FROM route_usage WHERE is_primary
            GROUP BY session_id
        ), residual_usage AS (
            SELECT s.id AS session_id,
                   COALESCE(NULLIF(s.billing_provider, ''), 'unknown') AS provider,
                   {residuals}
            FROM selected_sessions s LEFT JOIN primary_totals t ON t.session_id = s.id
        ), combined_usage AS (
            SELECT session_id, provider, {keys} FROM route_usage
            UNION ALL
            SELECT session_id, provider, {keys} FROM residual_usage WHERE {has_usage}
        )
        SELECT provider, {sums}, COUNT(DISTINCT session_id) AS sessions
        FROM combined_usage GROUP BY provider
        ORDER BY CASE WHEN SUM(actual_cost) > 0 THEN SUM(actual_cost)
                      ELSE SUM(estimated_cost) END DESC,
                 SUM(input_tokens) + SUM(output_tokens) DESC, provider
    """, cutoff)


def _get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)
        # Local calendar day, per-row (DST-correct), the same day /insights uses (agent/insights.py).
        daily = _rows(db, """
            SELECT date(started_at, 'unixepoch', 'localtime') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, cutoff)

        by_model = _rows(db, """
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, cutoff)

        # Fold in auxiliary usage (vision, compression, ...) from session_model_usage.
        # Aux calls never touch the sessions counters, so this is add-only — no double count.
        # Without it the models list shows only the main agent model even when aux models are actively
        # burning tokens (issue #23270).
        aux_rows = _aux_usage_rows(db, cutoff)
        by_model = _merge_aux_into_by_model(by_model, aux_rows)

        totals = _rows(db, """
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, cutoff)[0]
        usage = InsightsEngine(db).get_usage_breakdown(days=days)

        return {
            "daily": daily,
            "by_model": by_model,
            "by_provider": _get_provider_usage(db, cutoff),
            "by_task": _aux_task_summary(aux_rows),  # "what is compression costing me"
            "totals": totals,
            "period_days": days,
            "skills": usage["skills"],
            "tools": usage["tools"],  # per-tool-name counts; desktop aggregates per toolset
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
    with corrupt_store_as_status(_session_db_path_for_profile(profile)):
        return await asyncio.to_thread(_get_usage_analytics, days, profile)


_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "api_calls", "tool_calls",
)


def _has_usage(row: dict[str, Any]) -> bool:
    return any((row.get(key) or 0) != 0 for key in _USAGE_KEYS)


def _fold_session_only_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold model rows that carry no billing_provider and no usage into the single
    accounted provider row for that model.

    Session rows can be created before the first billable call finishes; if that early row
    records only the model name while a later row has real accounting, the Models page used
    to show a duplicate "0 tokens / — API calls" card. Only folds when ownership is
    unambiguous (exactly one provider row).
    """
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_model.setdefault(row.get("model") or "", []).append(row)

    rows: list[dict[str, Any]] = []
    for model_rows in rows_by_model.values():
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
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens", "estimated_cost", "api_calls",
)


def _merge_aux_into_rows(raw_rows: list[dict[str, Any]], aux_rows: list[dict[str, Any]]) -> None:
    """Add auxiliary usage onto the matching (model, billing_provider) row in place.

    Aux calls happen inside sessions the sessions-derived row already counted, so
    ``sessions`` is never added onto an existing row; only an aux-only pair (no
    sessions-derived row) gets a new row that carries its own session count.
    """
    index: dict[tuple, dict[str, Any]] = {
        (row.get("model") or "", row.get("billing_provider") or ""): row
        for row in raw_rows
    }
    for aux in aux_rows:
        key = (aux.get("model") or "unknown", aux.get("billing_provider") or "")
        target = index.get(key)
        if target is None:
            target = {
                "model": key[0],
                "billing_provider": key[1],
                **{k: 0 for k in _AUX_SUMMED_KEYS},
                "actual_cost": 0,
                "sessions": aux.get("sessions") or 0,
                "tool_calls": 0,
                "last_used_at": None,
                "avg_tokens_per_session": 0,
            }
            index[key] = target
            raw_rows.append(target)
        for k in _AUX_SUMMED_KEYS:
            target[k] = (target.get(k) or 0) + (aux.get(k) or 0)
        if aux.get("last_used_at") is not None:
            target["last_used_at"] = max(target.get("last_used_at") or 0, aux["last_used_at"])
        sessions = target.get("sessions") or 0
        if sessions:
            target["avg_tokens_per_session"] = (
                (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
            ) / sessions


_MODEL_CARD_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "sessions", "api_calls", "tool_calls",
    "last_used_at", "avg_tokens_per_session",
)


def _attach_tool_calls(db, cutoff: float, raw_rows: list[dict[str, Any]]) -> None:
    """Fill the ``tool_calls`` card metric for per-call rows, in place.

    Tool calls are session-level data (``sessions.tool_call_count``), not per API
    call, so they cannot come from ``session_model_usage``. Attribute each
    session-window (model, billing_provider) tool-call total to the card of the
    pair its sessions row records — the session's last active route — falling
    back to any card of that model when the pair has no per-call row (route
    switched after the last call). Never zeroes the metric (#71778).
    """
    pair_rows = _rows(db, """
        SELECT model, billing_provider, SUM(COALESCE(tool_call_count, 0)) as tool_calls
        FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        GROUP BY model, billing_provider
    """, cutoff)
    by_pair = {
        (r.get("model") or "", r.get("billing_provider") or ""): r.get("tool_calls") or 0
        for r in pair_rows
    }
    index: dict[tuple, dict[str, Any]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        row["tool_calls"] = 0
        index.setdefault((row["model"], row.get("billing_provider") or ""), row)
        by_model.setdefault(row["model"], []).append(row)
    for (model, provider), calls in by_pair.items():
        target = index.get((model, provider)) or (by_model.get(model) or [None])[0]
        if target is not None:
            target["tool_calls"] += calls


def _get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Per-model token/cost/session breakdown plus models.dev capability metadata."""
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)

        # Main usage from session_model_usage: every API call's delta lands there
        # with the model/provider active at call time, so a mid-session /model
        # switch (or a silent fallback rewrite) splits across the pairs that
        # actually ran. The sessions table keeps only the final pair, and
        # grouping it attributed the whole session to it (#71778). Insights
        # (_compute_model_breakdown) reads the same table.
        try:
            cur = db._conn.execute("""
                SELECT u.model,
                       u.billing_provider,
                       SUM(u.input_tokens) as input_tokens,
                       SUM(u.output_tokens) as output_tokens,
                       SUM(u.cache_read_tokens) as cache_read_tokens,
                       SUM(u.reasoning_tokens) as reasoning_tokens,
                       COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                       COALESCE(SUM(u.actual_cost_usd), 0) as actual_cost,
                       COUNT(DISTINCT u.session_id) as sessions,
                       SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                       MAX(u.last_seen) as last_used_at,
                       AVG(u.input_tokens + u.output_tokens) as avg_tokens_per_session
                FROM session_model_usage u
                JOIN sessions s ON s.id = u.session_id
                WHERE s.started_at > ? AND u.model IS NOT NULL AND u.model != ''
                      AND u.task = ''
                GROUP BY u.model, u.billing_provider
                ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
            """, (cutoff,))
            raw_rows = [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            raw_rows = []  # pre-v17 DB without session_model_usage
        if not raw_rows:
            # No per-call rows in the window (pre-table DB): fall back to the
            # sessions aggregate, which for those sessions is the only source.
            raw_rows = _rows(db, """
                SELECT model,
                       billing_provider,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       SUM(cache_read_tokens) as cache_read_tokens,
                       SUM(reasoning_tokens) as reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                       COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                       COUNT(*) as sessions,
                       SUM(COALESCE(api_call_count, 0)) as api_calls,
                       SUM(tool_call_count) as tool_calls,
                       MAX(started_at) as last_used_at,
                       AVG(input_tokens + output_tokens) as avg_tokens_per_session
                FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
                GROUP BY model, billing_provider
                ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
            """, cutoff)
        else:
            _attach_tool_calls(db, cutoff, raw_rows)

        # Aux usage (vision/compression/title/approval/...) is folded into the
        # (model, provider) row the sessions query already produced. #23270 made
        # aux-only models visible; _aux_usage_rows groups by (model, task,
        # provider), so appending each row emitted one card per aux task beside
        # the main card, and their session counts summed past
        # totals.total_sessions (#89631). Only a pair with no sessions-derived
        # row becomes a new row, carrying its own session count.
        _merge_aux_into_rows(raw_rows, _aux_usage_rows(db, cutoff))

        rows = _fold_session_only_rows(raw_rows)
        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = [
            {
                "model": row["model"],
                "provider": row.get("billing_provider") or "",
                **{key: row[key] for key in _MODEL_CARD_KEYS},
                "capabilities": _model_capabilities(row.get("billing_provider") or "", row["model"]),
            }
            for row in rows
        ]

        totals = _rows(db, """
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, cutoff)[0]
        # Counted over the same merged row set the cards come from, so a model
        # reached only through auxiliary usage is in the header as well as on
        # the page (#89631).
        totals["distinct_models"] = len({row["model"] for row in rows})

        return {"models": models, "totals": totals, "period_days": days}
    finally:
        db.close()


@router.get("/api/analytics/models")
async def get_models_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """Return model analytics without blocking the serving event loop."""
    with corrupt_store_as_status(_session_db_path_for_profile(profile)):
        return await asyncio.to_thread(_get_models_analytics, days, profile)
