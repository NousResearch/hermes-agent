"""Raw-YAML config and token/cost analytics dashboard routes.

Extracted from ``hermes_cli.web_server``; app state and helpers are late-bound through
:mod:`hermes_cli.web_deps` (cycle-safe, monkeypatch-friendly).
"""

import asyncio
import os
import re
import time
from glob import glob
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_deps import late
from hermes_cli.web_server_profiles import (
    _approval_mode_of, _aux_task_summary, _aux_usage_rows, _broadcast_gateway_session_info, _is_other_profile, _merge_aux_into_by_model,
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


# ── Log-based token accounting ──────────────────────────────────────────────
# The database ``sessions.api_call_count`` / ``session_model_usage`` rows lag
# behind real usage: the desktop gateway path writes absolute totals that
# overwrite per-call deltas, so any model the user switched into mid-session
# loses its token attribution. The plain ``agent.log`` file, by contrast,
# records one ``API call #N: model=… provider=… in=… out=… total=… latency=…s``
# line per call — independent of any model switch. This parser uses that file
# as the source of truth for the Usage tab's ``by_model`` breakdown.
# Trade-offs vs. the DB query:
#   • + accurate per-call attribution for every API call (CLI, gateway, cron, …)
#   • + survives DB resets / corrupted tables / schema migrations
#   • − depends on log rotation retention (ConcurrentRotatingFileHandler keeps
#     ``agent.log`` + a few rotated backups in ``<hermes_home>/logs/``)
#   • − ~3 MB log = ~10 ms parse, cached per-process for repeat reads

_API_CALL_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"API call #\d+: "
    r"model=(?P<model>\S+)\s+"
    r"provider=(?P<provider>\S+)\s+"
    r"in=(?P<in>\d+)\s+"
    r"out=(?P<out>\d+)\s+"
    r"total=(?P<total>\d+)\s+"
    r"latency=(?P<lat>[\d.]+)s"
)

# Per-process cache: keyed by file (path, size, mtime) → ALL parsed rows
# (pre-cutoff). The caller applies its own cutoff filter on the cached rows,
# so different ``days`` values reuse one parse pass per file.
_LOG_PARSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _parse_log_files(log_dir: str, cutoff: float) -> List[Dict[str, Any]]:
    """Parse every ``agent.log*`` file in ``log_dir`` and return rows newer than
    ``cutoff`` (Unix epoch seconds). Includes rotated backups (``agent.log.1``,
    ``agent.log.2``, …) so historical calls remain visible until rotation
    garbage-collects them.

    Each returned row: ``{"ts": "YYYY-MM-DD HH:MM:SS", "epoch": float,
    "model": str, "provider": str, "in": int, "out": int, "total": int,
    "latency": float}``. Rows are sorted oldest-first so aggregation order is
    deterministic.

    Cache: one entry per file, keyed by (path, size, mtime). Stores **all**
    rows (pre-cutoff) so a later call with a different ``cutoff`` reuses the
    parse without re-reading the file. The ``cutoff`` filter is applied to
    the cached rows on every call.
    """
    if not os.path.isdir(log_dir):
        return []

    # agent.log first, then rotated backups in numerical order.
    paths: List[str] = sorted(
        glob(os.path.join(log_dir, "agent.log")),
        key=lambda p: (0 if os.path.basename(p) == "agent.log" else 1, p),
    )
    paths.extend(
        sorted(
            glob(os.path.join(log_dir, "agent.log.*")),
            key=lambda p: (1, p),
        )
    )

    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        # Cheap invalidation: skip parse when (path, size, mtime) matches cache.
        cache_key = (path, st.st_size, int(st.st_mtime))
        cached = _LOG_PARSE_CACHE.get(path)
        if cached and cached.get("key") == cache_key:
            file_rows = cached["rows"]
        else:
            file_rows: List[Dict[str, Any]] = []
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        m = _API_CALL_RE.search(line)
                        if not m:
                            continue
                        try:
                            epoch = time.mktime(
                                time.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                            )
                        except ValueError:
                            continue
                        file_rows.append({
                            "ts": m.group("ts"),
                            "epoch": epoch,
                            "model": m.group("model"),
                            "provider": m.group("provider"),
                            "in": int(m.group("in")),
                            "out": int(m.group("out")),
                            "total": int(m.group("total")),
                            "latency": float(m.group("lat")),
                        })
            except OSError:
                continue
            _LOG_PARSE_CACHE[path] = {"key": cache_key, "rows": file_rows}
        # Apply cutoff filter on cached rows (not inside the parse loop —
        # the cache stores pre-cutoff rows so a different cutoff reuses them).
        rows.extend(r for r in file_rows if r["epoch"] >= cutoff)

    rows.sort(key=lambda r: r["epoch"])
    return rows


def _aggregate_by_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group rows by ``model`` and sum tokens. Keeps the schema the frontend
    expects (matches what the DB query used to return)."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model = r["model"]
        b = buckets.get(model)
        if b is None:
            b = {
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
                "sessions": 0,
                "api_calls": 0,
            }
            buckets[model] = b
        b["input_tokens"] += r["in"]
        b["output_tokens"] += r["out"]
        b["api_calls"] += 1
    out = list(buckets.values())
    out.sort(
        key=lambda r: (r["input_tokens"] + r["output_tokens"]),
        reverse=True,
    )
    return out


def _aggregate_log_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum all log rows into a totals dict matching the old DB schema."""
    total_in = sum(r["in"] for r in rows)
    total_out = sum(r["out"] for r in rows)
    return {
        "total_input": total_in,
        "total_output": total_out,
        "total_cache_read": 0,  # log line has no cache_read field
        "total_reasoning": 0,   # log line has no reasoning field
        "total_estimated_cost": 0.0,
        "total_actual_cost": 0,
        "total_sessions": 0,    # filled later from DB
        "total_api_calls": len(rows),
    }


def _aggregate_log_daily(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group log rows by day, matching the old DB daily schema."""
    by_day: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        day = r["ts"][:10]  # YYYY-MM-DD
        d = by_day.get(day)
        if d is None:
            d = {
                "day": day,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost": 0,
                "actual_cost": 0,
                "sessions": 0,
                "api_calls": 0,
            }
            by_day[day] = d
        d["input_tokens"] += r["in"]
        d["output_tokens"] += r["out"]
        d["api_calls"] += 1
    return sorted(by_day.values(), key=lambda d: d["day"])


def _get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)

        # ── Per-model breakdown: read the agent.log directly.
        #    The DB only stores session-level aggregates (and gateway absolute
        #    writes overwrite per-call deltas), so any /model switch mid-session
        #    would silently misattribute historical tokens. The log file is
        #    append-only and records one line per API call — the source of
        #    truth for "which model actually consumed these tokens".
        try:
            from hermes_constants import get_hermes_home
            log_dir = str(get_hermes_home() / "logs")
        except Exception:
            log_dir = os.path.expandvars(r"%LOCALAPPDATA%\hermes\logs")
        log_rows = _parse_log_files(log_dir, cutoff)
        by_model = _aggregate_by_model(log_rows)

        # ── Totals and daily breakdown: also from the log, so the summary
        #    cards and chart use the same source of truth as ``by_model``.
        #    The ``sessions`` table totals are inaccurate for the same reason
        #    ``by_model`` was: gateway absolute writes overwrite per-call
        #    deltas, undercounting tokens for switched-away models. The log
        #    records every call faithfully.
        log_totals = _aggregate_log_totals(log_rows)
        daily = _aggregate_log_daily(log_rows)

        # Session count still comes from the DB — the log has no concept of
        # "sessions", only individual API calls.
        db_session_count = _rows(db, """
            SELECT COUNT(*) as total_sessions
            FROM sessions WHERE started_at > ?
        """, cutoff)[0]["total_sessions"]

        # Auxiliary usage (vision, compression, title_generation, background
        # review) is recorded via ``record_aux_usage`` →
        # ``session_db.record_auxiliary_usage`` — a separate code path that
        # writes only to ``session_model_usage`` (task != '') and never emits
        # an ``API call #N`` log line. Two consequences:
        #   1. ``by_model`` from the log is missing aux-only models (e.g.
        #      ``Qwen/Qwen3-8B`` used for title generation, ``Qwen3.8-Flash``
        #      used for vision) — they'd silently disappear from the Usage tab.
        #   2. Aux calls that reuse the main-loop model (e.g. ``background_review``
        #      calling DeepSeek-V4) DO double-count if we naively add their
        #      tokens — the log already has the same model's main-loop calls.
        # Fix: merge aux rows only for models NOT already present in the log
        # (i.e. aux-only models), preserving the ``aux_tasks`` breakdown
        # ``_merge_aux_into_by_model`` attaches. Also fold aux tokens into
        # totals/daily so the summary cards are consistent with by_model.
        aux_rows = _aux_usage_rows(db, cutoff)
        log_models = {r["model"] for r in by_model}
        aux_only_rows = [r for r in aux_rows if r.get("model") not in log_models]
        by_model = _merge_aux_into_by_model(by_model, aux_only_rows)

        # Fold aux-only tokens into totals + daily for consistency.
        aux_in = sum(r.get("input_tokens") or 0 for r in aux_only_rows)
        aux_out = sum(r.get("output_tokens") or 0 for r in aux_only_rows)
        aux_calls = sum(r.get("api_calls") or 0 for r in aux_only_rows)
        log_totals["total_input"] += aux_in
        log_totals["total_output"] += aux_out
        log_totals["total_api_calls"] += aux_calls
        log_totals["total_sessions"] = db_session_count
        for r in aux_only_rows:
            ts = r.get("last_used_at")
            if ts:
                day = time.strftime("%Y-%m-%d", time.localtime(ts))
                d = next((x for x in daily if x["day"] == day), None)
                if d is None:
                    d = {"day": day, "input_tokens": 0, "output_tokens": 0,
                         "cache_read_tokens": 0, "reasoning_tokens": 0,
                         "estimated_cost": 0, "actual_cost": 0,
                         "sessions": 0, "api_calls": 0}
                    daily.append(d)
                d["input_tokens"] += r.get("input_tokens") or 0
                d["output_tokens"] += r.get("output_tokens") or 0
                d["api_calls"] += r.get("api_calls") or 0
        daily.sort(key=lambda d: d["day"])

        totals = log_totals
        usage = InsightsEngine(db).get_usage_breakdown(days=days)

        return {
            "daily": daily,
            "by_model": by_model,
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
    return await asyncio.to_thread(_get_usage_analytics, days, profile)


_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens",
    "estimated_cost", "actual_cost", "api_calls", "tool_calls",
)


def _has_usage(row: Dict[str, Any]) -> bool:
    return any((row.get(key) or 0) != 0 for key in _USAGE_KEYS)


def _fold_session_only_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold model rows that carry no billing_provider and no usage into the single
    accounted provider row for that model.

    Session rows can be created before the first billable call finishes; if that early row
    records only the model name while a later row has real accounting, the Models page used
    to show a duplicate "0 tokens / — API calls" card. Only folds when ownership is
    unambiguous (exactly one provider row).
    """
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_model.setdefault(row.get("model") or "", []).append(row)

    rows: List[Dict[str, Any]] = []
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

        # Aux-only models (dedicated vision/compression) as (model, provider) rows,
        # keyed like the GROUP BY above, so they appear on the Models page.
        # See #23270.
        for aux in _aux_usage_rows(db, cutoff):
            raw_rows.append({
                "model": aux.get("model") or "unknown",
                "billing_provider": aux.get("billing_provider") or "",
                **{key: aux.get(key) or 0 for key in _AUX_SUMMED_KEYS},
                "actual_cost": 0,
                "tool_calls": 0,
                "last_used_at": aux.get("last_used_at"),
                "avg_tokens_per_session": 0,
                "aux_task": aux.get("task") or "",
            })

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
            SELECT COUNT(DISTINCT model) as distinct_models,
                   SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, cutoff)[0]

        return {"models": models, "totals": totals, "period_days": days}
    finally:
        db.close()


@router.get("/api/analytics/models")
async def get_models_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """Return model analytics without blocking the serving event loop."""
    return await asyncio.to_thread(_get_models_analytics, days, profile)
