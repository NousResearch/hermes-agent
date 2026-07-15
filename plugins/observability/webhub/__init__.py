"""webhub — optional OpenRouter observability sink for Hermes.

Captures OpenRouter request telemetry into a small local SQLite store, writes a
rolling briefing snippet for morning briefings, and exposes helper functions
for dashboard and Prometheus surfaces.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from hermes_cli.config import load_config, load_env
from utils import base_url_host_matches

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()


@dataclass(frozen=True)
class _WindowSummary:
    total_requests: int
    ok_requests: int
    error_requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float
    last_request_at: str


def _storage_root() -> Path:
    return get_hermes_home() / "observability" / "webhub"


def _db_path() -> Path:
    return _storage_root() / "metrics.db"


def _briefing_path() -> Path:
    return _storage_root() / "latest_briefing.md"


def _ensure_storage_dir() -> None:
    _storage_root().mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_storage_dir()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS openrouter_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            observed_at_ts INTEGER NOT NULL,
            session_id TEXT,
            turn_id TEXT,
            api_request_id TEXT,
            provider TEXT,
            base_url TEXT,
            model TEXT,
            api_mode TEXT,
            status TEXT NOT NULL,
            latency_ms REAL NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            finish_reason TEXT,
            error_type TEXT,
            error_message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_webhub_requests_ts
        ON openrouter_requests(observed_at_ts)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_webhub_requests_model_ts
        ON openrouter_requests(model, observed_at_ts)
        """
    )
    return conn


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _runtime_env() -> dict[str, Any]:
    try:
        return load_env()
    except Exception:
        return {}


def _env_value(name: str) -> str:
    env = _runtime_env()
    return str(os.environ.get(name) or env.get(name) or "").strip()


def is_openrouter_request(provider: str = "", base_url: str = "") -> bool:
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider == "openrouter":
        return True
    return base_url_host_matches(str(base_url or ""), "openrouter.ai")


def _canonical_usage(usage: Optional[dict[str, Any]] = None):
    from agent.usage_pricing import CanonicalUsage

    data = usage or {}
    return CanonicalUsage(
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        cache_read_tokens=int(data.get("cache_read_tokens") or 0),
        cache_write_tokens=int(data.get("cache_write_tokens") or 0),
        reasoning_tokens=int(data.get("reasoning_tokens") or 0),
        request_count=int(data.get("request_count") or 1),
    )


def _estimate_cost_usd(model: str, usage: Optional[dict[str, Any]], provider: str, base_url: str) -> float:
    if not usage:
        return 0.0
    try:
        from agent.usage_pricing import estimate_usage_cost

        cost = estimate_usage_cost(
            model or "",
            _canonical_usage(usage),
            provider=provider or "",
            base_url=base_url or "",
            api_key="",
        )
        return float(cost.amount_usd or 0.0)
    except Exception:
        logger.debug("webhub: cost estimation failed", exc_info=True)
        return 0.0


def _record_request(
    *,
    status: str,
    session_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    provider: str = "",
    base_url: str = "",
    model: str = "",
    api_mode: str = "",
    latency_ms: float = 0.0,
    usage: Optional[dict[str, Any]] = None,
    finish_reason: str = "",
    error_type: str = "",
    error_message: str = "",
    observed_at: Optional[datetime] = None,
) -> None:
    if not is_openrouter_request(provider=provider, base_url=base_url):
        return
    observed = observed_at or _utc_now()
    usage = usage or {}
    cost_usd = _estimate_cost_usd(model, usage, provider, base_url)
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO openrouter_requests (
                    observed_at,
                    observed_at_ts,
                    session_id,
                    turn_id,
                    api_request_id,
                    provider,
                    base_url,
                    model,
                    api_mode,
                    status,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    finish_reason,
                    error_type,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _isoformat(observed),
                    int(observed.timestamp()),
                    session_id,
                    turn_id,
                    api_request_id,
                    provider,
                    base_url,
                    model,
                    api_mode,
                    status,
                    float(latency_ms or 0.0),
                    int(usage.get("input_tokens") or 0),
                    int(usage.get("output_tokens") or 0),
                    int(usage.get("cache_read_tokens") or 0),
                    int(usage.get("cache_write_tokens") or 0),
                    int(usage.get("reasoning_tokens") or 0),
                    int(usage.get("total_tokens") or 0),
                    cost_usd,
                    finish_reason,
                    error_type,
                    error_message,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    _refresh_briefing_file()


def _window_seconds(window_hours: int) -> int:
    return max(1, int(window_hours)) * 3600


def _empty_summary() -> _WindowSummary:
    return _WindowSummary(
        total_requests=0,
        ok_requests=0,
        error_requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        avg_latency_ms=0.0,
        last_request_at="",
    )


def _query_window_summary(conn: sqlite3.Connection, *, window_hours: int) -> _WindowSummary:
    cutoff = int(time.time()) - _window_seconds(window_hours)
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_requests,
            SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_requests,
            SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS error_requests,
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
            COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
            COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
            COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
            COALESCE(MAX(observed_at), '') AS last_request_at
        FROM openrouter_requests
        WHERE observed_at_ts >= ?
        """,
        (cutoff,),
    ).fetchone()
    if row is None:
        return _empty_summary()
    return _WindowSummary(
        total_requests=int(row["total_requests"] or 0),
        ok_requests=int(row["ok_requests"] or 0),
        error_requests=int(row["error_requests"] or 0),
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
        cache_read_tokens=int(row["cache_read_tokens"] or 0),
        cache_write_tokens=int(row["cache_write_tokens"] or 0),
        reasoning_tokens=int(row["reasoning_tokens"] or 0),
        total_tokens=int(row["total_tokens"] or 0),
        estimated_cost_usd=float(row["estimated_cost_usd"] or 0.0),
        avg_latency_ms=float(row["avg_latency_ms"] or 0.0),
        last_request_at=str(row["last_request_at"] or ""),
    )


def get_dashboard_summary(window_hours: int = 24) -> dict[str, Any]:
    with _LOCK:
        conn = _connect()
        try:
            summary = _query_window_summary(conn, window_hours=window_hours)
            cutoff = int(time.time()) - _window_seconds(window_hours)
            top_models = [
                {
                    "model": str(row["model"] or "unknown"),
                    "requests": int(row["requests"] or 0),
                    "estimated_cost_usd": float(row["estimated_cost_usd"] or 0.0),
                    "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
                    "total_tokens": int(row["total_tokens"] or 0),
                }
                for row in conn.execute(
                    """
                    SELECT
                        model,
                        COUNT(*) AS requests,
                        COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens
                    FROM openrouter_requests
                    WHERE observed_at_ts >= ?
                    GROUP BY model
                    ORDER BY requests DESC, estimated_cost_usd DESC, model ASC
                    LIMIT 5
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            hourly = [
                {
                    "bucket_start": _isoformat(datetime.fromtimestamp(int(row["bucket_ts"]), tz=timezone.utc)),
                    "requests": int(row["requests"] or 0),
                    "estimated_cost_usd": float(row["estimated_cost_usd"] or 0.0),
                    "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
                }
                for row in conn.execute(
                    """
                    SELECT
                        ((observed_at_ts / 3600) * 3600) AS bucket_ts,
                        COUNT(*) AS requests,
                        COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                    FROM openrouter_requests
                    WHERE observed_at_ts >= ?
                    GROUP BY bucket_ts
                    ORDER BY bucket_ts ASC
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            recent_errors = [
                {
                    "observed_at": str(row["observed_at"] or ""),
                    "model": str(row["model"] or ""),
                    "error_type": str(row["error_type"] or ""),
                    "error_message": str(row["error_message"] or ""),
                }
                for row in conn.execute(
                    """
                    SELECT observed_at, model, error_type, error_message
                    FROM openrouter_requests
                    WHERE observed_at_ts >= ? AND status != 'ok'
                    ORDER BY observed_at_ts DESC
                    LIMIT 5
                    """,
                    (cutoff,),
                ).fetchall()
            ]
        finally:
            conn.close()

    throughput_per_hour = (
        float(summary.total_requests) / float(max(1, window_hours))
        if summary.total_requests
        else 0.0
    )
    success_rate = (
        float(summary.ok_requests) / float(summary.total_requests)
        if summary.total_requests
        else 0.0
    )
    result = {
        "window_hours": int(window_hours),
        "requests": {
            "total": summary.total_requests,
            "ok": summary.ok_requests,
            "error": summary.error_requests,
            "success_rate": success_rate,
            "throughput_per_hour": throughput_per_hour,
        },
        "tokens": {
            "input": summary.input_tokens,
            "output": summary.output_tokens,
            "cache_read": summary.cache_read_tokens,
            "cache_write": summary.cache_write_tokens,
            "reasoning": summary.reasoning_tokens,
            "total": summary.total_tokens,
        },
        "cost": {
            "estimated_usd": summary.estimated_cost_usd,
        },
        "latency": {
            "avg_ms": summary.avg_latency_ms,
        },
        "last_request_at": summary.last_request_at,
        "top_models": top_models,
        "hourly": hourly,
        "recent_errors": recent_errors,
    }
    result["briefing_markdown"] = _render_briefing_from_summary(result)
    return result


def _render_briefing_from_summary(summary: dict[str, Any]) -> str:
    window_hours = int(summary.get("window_hours") or 24)
    requests = summary["requests"]
    cost = summary["cost"]
    latency = summary["latency"]
    top_models = summary["top_models"]
    lead = top_models[0]["model"] if top_models else "none yet"
    return (
        f"# Webhub observability\n\n"
        f"- Window: last {window_hours}h\n"
        f"- OpenRouter requests: {requests['total']} total, {requests['ok']} ok, {requests['error']} errors\n"
        f"- Success rate: {requests['success_rate'] * 100:.1f}%\n"
        f"- Throughput: {requests['throughput_per_hour']:.2f} requests/hour\n"
        f"- Estimated cost: ${cost['estimated_usd']:.4f}\n"
        f"- Average latency: {latency['avg_ms']:.1f} ms\n"
        f"- Leading model: {lead}\n"
    )


def render_briefing_markdown(window_hours: int = 24) -> str:
    return _render_briefing_from_summary(get_dashboard_summary(window_hours=window_hours))


def _refresh_briefing_file() -> None:
    try:
        _ensure_storage_dir()
        _briefing_path().write_text(render_briefing_markdown(window_hours=24), encoding="utf-8")
    except Exception:
        logger.debug("webhub: could not refresh briefing file", exc_info=True)


def briefing_file_path() -> str:
    return str(_briefing_path())


def get_runtime_status() -> dict[str, Any]:
    config = load_config() or {}
    enabled = set((((config.get("plugins") or {}).get("enabled")) or []))
    return {
        "plugin_enabled": "observability/webhub" in enabled,
        "openrouter_configured": bool(_env_value("OPENROUTER_API_KEY")),
        "slack_configured": bool(_env_value("SLACK_BOT_TOKEN")),
        "briefing_file": briefing_file_path(),
    }


def render_prometheus_metrics() -> str:
    summary = get_dashboard_summary(window_hours=24)
    requests = summary["requests"]
    tokens = summary["tokens"]
    cost = summary["cost"]
    latency = summary["latency"]
    lines = [
        "# HELP hermes_webhub_requests_total OpenRouter requests observed by the webhub sink.",
        "# TYPE hermes_webhub_requests_total counter",
        f"hermes_webhub_requests_total {requests['total']}",
        "# HELP hermes_webhub_requests_status_total OpenRouter requests by status.",
        "# TYPE hermes_webhub_requests_status_total counter",
        f'hermes_webhub_requests_status_total{{status="ok"}} {requests["ok"]}',
        f'hermes_webhub_requests_status_total{{status="error"}} {requests["error"]}',
        "# HELP hermes_webhub_estimated_cost_usd Estimated OpenRouter spend over the rolling 24h window.",
        "# TYPE hermes_webhub_estimated_cost_usd gauge",
        f"hermes_webhub_estimated_cost_usd {cost['estimated_usd']}",
        "# HELP hermes_webhub_avg_latency_ms Average OpenRouter latency over the rolling 24h window.",
        "# TYPE hermes_webhub_avg_latency_ms gauge",
        f"hermes_webhub_avg_latency_ms {latency['avg_ms']}",
        "# HELP hermes_webhub_tokens_total Token volume over the rolling 24h window by token class.",
        "# TYPE hermes_webhub_tokens_total gauge",
        f'hermes_webhub_tokens_total{{kind="input"}} {tokens["input"]}',
        f'hermes_webhub_tokens_total{{kind="output"}} {tokens["output"]}',
        f'hermes_webhub_tokens_total{{kind="cache_read"}} {tokens["cache_read"]}',
        f'hermes_webhub_tokens_total{{kind="cache_write"}} {tokens["cache_write"]}',
        f'hermes_webhub_tokens_total{{kind="reasoning"}} {tokens["reasoning"]}',
        f'hermes_webhub_tokens_total{{kind="total"}} {tokens["total"]}',
        "# HELP hermes_webhub_throughput_per_hour Average requests per hour over the rolling 24h window.",
        "# TYPE hermes_webhub_throughput_per_hour gauge",
        f"hermes_webhub_throughput_per_hour {requests['throughput_per_hour']}",
    ]
    for item in summary["top_models"]:
        model = str(item["model"]).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'hermes_webhub_model_requests_total{{model="{model}"}} {int(item["requests"])}'
        )
    return "\n".join(lines) + "\n"


def _status_command(_args: str) -> str:
    runtime = get_runtime_status()
    summary = get_dashboard_summary(window_hours=24)
    requests = summary["requests"]
    cost = summary["cost"]
    latency = summary["latency"]
    lines = [
        "Webhub observability (OpenRouter, last 24h):",
        f"- Plugin enabled: {'yes' if runtime['plugin_enabled'] else 'no'}",
        f"- OpenRouter configured: {'yes' if runtime['openrouter_configured'] else 'no'}",
        f"- Slack configured: {'yes' if runtime['slack_configured'] else 'no'}",
        f"- Requests: {requests['total']} total / {requests['ok']} ok / {requests['error']} errors",
        f"- Throughput: {requests['throughput_per_hour']:.2f} req/h",
        f"- Estimated cost: ${cost['estimated_usd']:.4f}",
        f"- Average latency: {latency['avg_ms']:.1f} ms",
        f"- Briefing file: {runtime['briefing_file']}",
    ]
    return "\n".join(lines)


def on_post_api_request(**kwargs: Any) -> None:
    try:
        _record_request(
            status="ok",
            session_id=str(kwargs.get("session_id") or ""),
            turn_id=str(kwargs.get("turn_id") or ""),
            api_request_id=str(kwargs.get("api_request_id") or ""),
            provider=str(kwargs.get("provider") or ""),
            base_url=str(kwargs.get("base_url") or ""),
            model=str(kwargs.get("model") or kwargs.get("response_model") or ""),
            api_mode=str(kwargs.get("api_mode") or ""),
            latency_ms=float(kwargs.get("api_duration") or 0.0) * 1000.0,
            usage=kwargs.get("usage") if isinstance(kwargs.get("usage"), dict) else None,
            finish_reason=str(kwargs.get("finish_reason") or ""),
        )
    except Exception:
        logger.debug("webhub: post_api_request hook failed", exc_info=True)


def on_api_request_error(**kwargs: Any) -> None:
    try:
        error = kwargs.get("error") if isinstance(kwargs.get("error"), dict) else {}
        _record_request(
            status="error",
            session_id=str(kwargs.get("session_id") or ""),
            turn_id=str(kwargs.get("turn_id") or ""),
            api_request_id=str(kwargs.get("api_request_id") or ""),
            provider=str(kwargs.get("provider") or ""),
            base_url=str(kwargs.get("base_url") or ""),
            model=str(kwargs.get("model") or ""),
            api_mode=str(kwargs.get("api_mode") or ""),
            latency_ms=float(kwargs.get("api_duration") or 0.0) * 1000.0,
            error_type=str(error.get("type") or ""),
            error_message=str(error.get("message") or ""),
        )
    except Exception:
        logger.debug("webhub: api_request_error hook failed", exc_info=True)


def register(ctx) -> None:
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_command(
        "webhub-status",
        _status_command,
        description="Show the local OpenRouter observability summary captured by the webhub sink",
    )
