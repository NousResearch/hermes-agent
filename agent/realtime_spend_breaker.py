"""Cross-process WAL SQLite AI spend ledger & multi-tier fast leak circuit breaker.

Tracks real-time net EUR AI spend across all Hermes processes (Desktop, Gateway,
Cron, Subagents, background_review) and providers (Vertex AI after -50% promo credit,
OpenRouter, etc.) with <1s shared visibility via WAL-mode SQLite.

Enforces quality-neutral multi-tier leak stopping:
- Tier 0: Exempts critical 24/7 operational responders (chat-247-*, evelina-*,
  alert-triage, pos-sentinel, secret-403-fastresponder, incident-*) from daily caps.
- Tier 1: Halts runaway non-interactive sessions (> €4.00 net in 15m or > €10.00 in 1h
  or >= 4 cache floor-resets with > €2.50 in 15m) and stalled/non-productive Desktop
  loops (>= 35 consecutive calls since last human message with making_productive_progress=False
  and > €6.00 net in 15m).
- Tier 2: Suppresses background_review and non-Tier-0 bulk crons when 15m net > €8.00
  or today net > €150.00 (75% of the €200/day May 2027 AI spend target).
- Tier 3: Hard-halts all non-Tier-0 non-interactive calls (cron, subagent,
  background_review) when today net > €195.00 (€200/day = €6,000/mo May 2027 target),
  while never blocking productive interactive Desktop turns.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")

# Vertex AI net EUR rates per 1M tokens on uop-atlas (after -50% promotional credit)
# USD list -> -50% promo -> EUR at 0.8981 EUR/USD (verified against BigQuery billing_export 202611)
# Global SKUs: 3BAE-62B7-C9B9 (in), 3B3E-7322-E883 (cache), 1E84-F91D-8B95 (out)
VERTEX_ARGON_UNCACHED_IN_EUR_PER_M = 1.796200
VERTEX_ARGON_CACHED_IN_EUR_PER_M = 0.089810
VERTEX_ARGON_OUT_EUR_PER_M = 8.981000

# Regional europe-west2 SKUs (A891-E7A5-FDA5, F61C-57E6-DBE0, EA0E-13A1-6B07) carry 1.10x multiplier
VERTEX_EU_WEST2_REGIONAL_MULTIPLIER = 1.10
_EU_WEST2_MACRO_CLUSTERS: frozenset[str] = frozenset({"0L8P", "9_oP", "6-wP", "6_gP"})

# Global Flash SKUs: 2A35-AC68-14D6 (in), 8658-6939-8BEC (cache), 90FD-DC5F-DC55 (out)
VERTEX_FLASH_UNCACHED_IN_EUR_PER_M = 0.673573
VERTEX_FLASH_CACHED_IN_EUR_PER_M = 0.067357
VERTEX_FLASH_OUT_EUR_PER_M = 3.367844

USD_TO_EUR = 0.92

# Budget & breaker thresholds (EUR net) aligned with May 2027 Sparplan (€6,000/mo = €200/day)
MAY_2027_DAILY_TARGET_EUR = 200.0
MAY_2027_MONTHLY_AI_TARGET_EUR = 6000.0
MAY_2027_MONTHLY_INFRA_TARGET_EUR = 10500.0

TIER1_NON_INTERACTIVE_15M_CAP_EUR = 4.00
TIER1_NON_INTERACTIVE_1H_CAP_EUR = 10.00
TIER1_NON_INTERACTIVE_CACHE_RESET_COUNT = 4
TIER1_NON_INTERACTIVE_CACHE_RESET_MIN_EUR = 2.50

TIER1_STALLED_DESKTOP_MIN_CALLS = 35
TIER1_STALLED_DESKTOP_15M_CAP_EUR = 6.00

TIER0_CRITICAL_CRON_15M_SAFETY_CAP_EUR = 12.00

TIER2_ESTATE_15M_BURST_EUR = 8.00
TIER2_DAILY_SOFT_CAP_EUR = 150.00
TIER3_DAILY_HARD_CAP_EUR = 195.00

_TIER0_CRITICAL_PREFIXES: tuple[str, ...] = (
    "chat-247",
    "chat247-",
    "nw-chat-",
    "newintra-",
    "workos-",
    "medius-",
    "lager-",
    "2izii-",
    "post-deploy-verify-",
    "evelina-",
    "emir-morning-routine",
    "alert-triage",
    "alerts-digest",
    "pos-sentinel",
    "secret-403",
    "incident-",
    "estate-doctor",
    "svea-",
    "cart-",
    "store-support-",
)

_LAST_STATUS_WRITE_TS: float = 0.0


def _db_path() -> Path:
    env_p = os.environ.get("HERMES_AI_SPEND_DB_PATH")
    if env_p:
        return Path(env_p)
    return Path.home() / ".hermes" / "state" / "ai-spend-realtime.db"


def _status_path() -> Path:
    env_p = os.environ.get("HERMES_AI_SPEND_STATUS_PATH")
    if env_p:
        return Path(env_p)
    return Path.home() / ".hermes" / "state" / "ai-cost-realtime-status.json"


def _emergency_brake_flag_path() -> Path:
    env_p = os.environ.get("HERMES_AI_EMERGENCY_BRAKE_FLAG_PATH")
    if env_p:
        return Path(env_p)
    return Path.home() / ".hermes" / "state" / "ai-emergency-brake-override.flag"


def is_manual_emergency_brake_active() -> bool:
    try:
        return _emergency_brake_flag_path().exists()
    except Exception:
        return False


def _day_cet(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=_STOCKHOLM_TZ).strftime("%Y-%m-%d")


def _day_start_epoch_cet(ts: float) -> float:
    dt = datetime.datetime.fromtimestamp(ts, tz=_STOCKHOLM_TZ)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=2.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=2000")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_call_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            day_cet TEXT NOT NULL,
            session_id TEXT NOT NULL,
            source TEXT NOT NULL,
            job_name TEXT DEFAULT '',
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            uncached_in INTEGER NOT NULL,
            cached_in INTEGER NOT NULL,
            out_tokens INTEGER NOT NULL,
            net_eur REAL NOT NULL,
            cluster_id TEXT DEFAULT '',
            cache_reset INTEGER DEFAULT 0,
            request_id TEXT UNIQUE
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts ON llm_call_ledger(ts)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_day ON llm_call_ledger(day_cet)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_sess_ts ON llm_call_ledger(session_id, ts)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS quarantined_sessions (
            session_id TEXT PRIMARY KEY,
            quarantined_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            reason TEXT NOT NULL,
            net_eur_window REAL NOT NULL
        )
        """
    )
    return con


def extract_vertex_cluster_id(request_id: str | None) -> str:
    if not request_id or not isinstance(request_id, str):
        return ""
    rid = request_id.strip()
    if len(rid) >= 16:
        return rid[10:16]
    return rid[:16]


def compute_call_net_eur(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    uncached_input_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float | None = None,
    cluster_id: str | None = None,
) -> float:
    prov = (provider or "").lower()
    mod = (model or "").lower()
    uncached = max(0, int(uncached_input_tokens or 0))
    cached = max(0, int(cache_read_tokens or 0))
    out = max(0, int(output_tokens or 0))
    cid = str(cluster_id or "")
    macro = cid[2:6] if len(cid) >= 6 else ""
    reg_mult = VERTEX_EU_WEST2_REGIONAL_MULTIPLIER if macro in _EU_WEST2_MACRO_CLUSTERS else 1.0

    if "argon" in mod and (prov in ("vertex", "", "google") or estimated_cost_usd is None):
        return round(
            reg_mult
            * (
                uncached * VERTEX_ARGON_UNCACHED_IN_EUR_PER_M
                + cached * VERTEX_ARGON_CACHED_IN_EUR_PER_M
                + out * VERTEX_ARGON_OUT_EUR_PER_M
            )
            / 1_000_000.0,
            6,
        )
    if "flash" in mod and ("gemini" in mod or prov in ("vertex", "google")):
        return round(
            (
                uncached * VERTEX_FLASH_UNCACHED_IN_EUR_PER_M
                + cached * VERTEX_FLASH_CACHED_IN_EUR_PER_M
                + out * VERTEX_FLASH_OUT_EUR_PER_M
            )
            / 1_000_000.0,
            6,
        )
    if estimated_cost_usd is not None and estimated_cost_usd > 0:
        return round(float(estimated_cost_usd) * USD_TO_EUR, 6)
    return 0.0


def is_tier0_critical_cron(job_name: str | None, session_id: str | None = None) -> bool:
    ctx_name = ""
    if not job_name:
        try:
            from cron.execution_identity import current_cron_execution
            ctx = current_cron_execution()
            if isinstance(ctx, dict):
                ctx_name = str(ctx.get("job_name") or "")
            elif ctx is not None:
                ctx_name = str(getattr(ctx, "job_name", "") or "")
        except Exception:
            pass
    for candidate in (job_name or "", ctx_name, session_id or ""):
        low = candidate.strip().lower()
        if low and any(low.startswith(p) or p in low for p in _TIER0_CRITICAL_PREFIXES):
            return True
    return False


def record_realtime_llm_call(
    *,
    session_id: str,
    source: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    uncached_input_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    request_id: str | None = None,
    job_name: str = "",
    estimated_cost_usd: float | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    now = float(ts if ts is not None else time.time())
    day_cet = _day_cet(now)
    uncached = max(0, int(uncached_input_tokens or 0))
    cached = max(0, int(cache_read_tokens or 0))
    out = max(0, int(output_tokens or 0))
    prompt = int(prompt_tokens or (uncached + cached))
    cluster_id = extract_vertex_cluster_id(request_id)
    is_argon_live = (
        str(source) != "log_tail"
        and not str(session_id).startswith("log_tail")
        and prompt > 55_000
        and "argon" in (model or "").lower()
    )
    cache_reset = (
        1
        if (is_argon_live and prompt > 110_000 and cached < 42_000 and uncached > 70_000)
        else 0
    )
    net_eur = compute_call_net_eur(
        provider=provider,
        model=model,
        prompt_tokens=prompt,
        uncached_input_tokens=uncached,
        cache_read_tokens=cached,
        output_tokens=out,
        estimated_cost_usd=estimated_cost_usd,
        cluster_id=cluster_id,
    )
    req_key = request_id or f"{session_id}:{now:.6f}:{prompt}:{out}"

    row_info = {
        "ts": now,
        "day_cet": day_cet,
        "session_id": session_id or "unknown",
        "source": source or "unknown",
        "job_name": job_name or "",
        "provider": provider or "unknown",
        "model": model or "unknown",
        "prompt_tokens": prompt,
        "uncached_in": uncached,
        "cached_in": cached,
        "out_tokens": out,
        "net_eur": net_eur,
        "cluster_id": cluster_id,
        "cache_reset": cache_reset,
        "request_id": req_key,
    }
    try:
        con = _connect()
        try:
            if is_argon_live and not cache_reset:
                prev_row = con.execute(
                    """
                    SELECT prompt_tokens, cached_in
                    FROM llm_call_ledger
                    WHERE session_id = ? AND ts >= ? AND request_id != ?
                    ORDER BY ts DESC, id DESC LIMIT 1
                    """,
                    (row_info["session_id"], now - 1800.0, req_key),
                ).fetchone()
                if prev_row is not None:
                    prev_prompt, prev_cached = int(prev_row[0] or 0), int(prev_row[1] or 0)
                    if (
                        prev_cached >= 42_000
                        and cached < 42_000
                        and prompt >= prev_prompt * 0.85
                    ) or (
                        prompt >= prev_prompt
                        and prev_cached >= 35_000
                        and (prev_cached - cached) >= 16_384
                        and cached < prompt * 0.50
                    ):
                        cache_reset = 1
                        row_info["cache_reset"] = 1
            with con:
                con.execute(
                    """
                    INSERT INTO llm_call_ledger (
                        ts, day_cet, session_id, source, job_name, provider, model,
                        prompt_tokens, uncached_in, cached_in, out_tokens, net_eur,
                        cluster_id, cache_reset, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id) DO UPDATE SET
                        session_id = CASE WHEN llm_call_ledger.source = 'log_tail' AND excluded.source != 'log_tail' THEN excluded.session_id ELSE llm_call_ledger.session_id END,
                        source = CASE WHEN llm_call_ledger.source = 'log_tail' AND excluded.source != 'log_tail' THEN excluded.source ELSE llm_call_ledger.source END,
                        job_name = CASE WHEN llm_call_ledger.job_name = '' AND excluded.job_name != '' THEN excluded.job_name ELSE llm_call_ledger.job_name END,
                        out_tokens = MAX(llm_call_ledger.out_tokens, excluded.out_tokens),
                        net_eur = MAX(llm_call_ledger.net_eur, excluded.net_eur)
                    """,
                    (
                        now,
                        day_cet,
                        row_info["session_id"],
                        row_info["source"],
                        row_info["job_name"],
                        row_info["provider"],
                        row_info["model"],
                        prompt,
                        uncached,
                        cached,
                        out,
                        net_eur,
                        cluster_id,
                        cache_reset,
                        req_key,
                    ),
                )
        finally:
            con.close()
        global _LAST_STATUS_WRITE_TS
        if now - _LAST_STATUS_WRITE_TS >= 10.0:
            _LAST_STATUS_WRITE_TS = now
            write_realtime_status_snapshot(now=now)
    except Exception as exc:
        logger.debug("record_realtime_llm_call failed open: %s", exc)
    return row_info


def get_realtime_spend_summary(now: float | None = None) -> dict[str, Any]:
    ts_now = float(now if now is not None else time.time())
    day_cet = _day_cet(ts_now)
    t_15m = ts_now - 900.0
    t_1h = ts_now - 3600.0
    try:
        con = _connect()
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(net_eur), 0.0),
                       COALESCE(SUM(uncached_in), 0), COALESCE(SUM(cached_in), 0),
                       COALESCE(SUM(cache_reset), 0)
                FROM llm_call_ledger WHERE ts >= ? AND source != 'log_tail'
                """,
                (t_15m,),
            )
            c15, eur15, u15, ca15, cr15_total = cur.fetchone()

            cur.execute(
                """
                SELECT COALESCE(MAX(s_cr), 0)
                FROM (
                    SELECT COALESCE(SUM(cache_reset), 0) AS s_cr
                    FROM llm_call_ledger
                    WHERE ts >= ? AND source != 'log_tail'
                    GROUP BY session_id
                )
                """,
                (t_15m,),
            )
            cr15 = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(net_eur), 0.0),
                       COALESCE(SUM(uncached_in), 0), COALESCE(SUM(cached_in), 0),
                       COALESCE(SUM(cache_reset), 0)
                FROM llm_call_ledger WHERE ts >= ? AND source != 'log_tail'
                """,
                (t_1h,),
            )
            c1h, eur1h, u1h, ca1h, cr1h = cur.fetchone()

            cur.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(net_eur), 0.0),
                       COALESCE(SUM(CASE WHEN source NOT IN ('desktop', 'mobile', 'tui', 'tui_gateway', 'gui', 'web', 'cli', 'interactive', 'log_tail') THEN net_eur ELSE 0.0 END), 0.0),
                       COALESCE(SUM(uncached_in), 0), COALESCE(SUM(cached_in), 0),
                       COALESCE(SUM(cache_reset), 0)
                FROM llm_call_ledger WHERE day_cet = ? AND source != 'log_tail'
                """,
                (day_cet,),
            )
            cday, eurday, eurday_non_int, uday, caday, crday = cur.fetchone()

            cur.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(net_eur), 0.0),
                       COALESCE(SUM(uncached_in), 0), COALESCE(SUM(cached_in), 0)
                FROM llm_call_ledger WHERE day_cet = ?
                """,
                (day_cet,),
            )
            cday_all, eurday_all, uday_all, caday_all = cur.fetchone()

            cur.execute(
                "SELECT COUNT(*) FROM quarantined_sessions WHERE expires_at > ?",
                (ts_now,),
            )
            q_count = cur.fetchone()[0]
        finally:
            con.close()
    except Exception as exc:
        logger.debug("get_realtime_spend_summary failed open: %s", exc)
        return {
            "ts": ts_now,
            "day_cet": day_cet,
            "calls_15m": 0,
            "net_eur_15m": 0.0,
            "cache_hit_pct_15m": 100.0,
            "cache_resets_15m": 0,
            "calls_1h": 0,
            "net_eur_1h": 0.0,
            "cache_hit_pct_1h": 100.0,
            "cache_resets_1h": 0,
            "calls_today": 0,
            "net_eur_today": 0.0,
            "calls_today_all": 0,
            "net_eur_today_all": 0.0,
            "cache_hit_pct_today_all": 100.0,
            "net_eur_today_non_interactive": 0.0,
            "cache_hit_pct_today": 100.0,
            "cache_resets_today": 0,
            "active_quarantined_sessions": 0,
            "may_2027_daily_target_eur": MAY_2027_DAILY_TARGET_EUR,
        }

    hit15 = round(100.0 * ca15 / max(1, u15 + ca15), 2) if (u15 + ca15) > 0 else 100.0
    hit1h = round(100.0 * ca1h / max(1, u1h + ca1h), 2) if (u1h + ca1h) > 0 else 100.0
    hitday = round(100.0 * caday / max(1, uday + caday), 2) if (uday + caday) > 0 else 100.0
    hitday_all = round(100.0 * caday_all / max(1, uday_all + caday_all), 2) if (uday_all + caday_all) > 0 else 100.0
    manual_brake = is_manual_emergency_brake_active()
    eurday_f = float(eurday or 0.0)
    eur15_f = float(eur15 or 0.0)
    q_int = int(q_count or 0)
    if manual_brake:
        breaker_state = "MANUAL_EMERGENCY_BRAKE"
    elif eurday_f >= TIER3_DAILY_HARD_CAP_EUR:
        breaker_state = "TIER3_DAILY_HARD_CAP"
    elif eurday_f >= TIER2_DAILY_SOFT_CAP_EUR or eur15_f >= TIER2_ESTATE_15M_BURST_EUR:
        breaker_state = "TIER2_DAILY_SOFT_CAP"
    elif q_int > 0:
        breaker_state = "TIER1_SESSION_QUARANTINED"
    else:
        breaker_state = "ARMED_OK"

    return {
        "ts": ts_now,
        "day_cet": day_cet,
        "breaker_state": breaker_state,
        "manual_emergency_brake_active": manual_brake,
        "calls_15m": int(c15 or 0),
        "net_eur_15m": round(eur15_f, 4),
        "cache_hit_pct_15m": hit15,
        "cache_resets_15m": int(cr15 or 0),
        "cache_resets_15m_total": int(cr15_total or 0),
        "calls_1h": int(c1h or 0),
        "net_eur_1h": round(float(eur1h or 0.0), 4),
        "cache_hit_pct_1h": hit1h,
        "cache_resets_1h": int(cr1h or 0),
        "calls_today": int(cday or 0),
        "net_eur_today": round(eurday_f, 4),
        "calls_today_all": int(cday_all or 0),
        "net_eur_today_all": round(float(eurday_all or 0.0), 4),
        "cache_hit_pct_today_all": hitday_all,
        "net_eur_today_non_interactive": round(float(eurday_non_int or 0.0), 4),
        "cache_hit_pct_today": hitday,
        "cache_resets_today": int(crday or 0),
        "active_quarantined_sessions": q_int,
        "may_2027_daily_target_eur": MAY_2027_DAILY_TARGET_EUR,
    }


def write_realtime_status_snapshot(now: float | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = get_realtime_spend_summary(now=now)
    if extra:
        summary.update(extra)
    spath = _status_path()
    try:
        spath.parent.mkdir(parents=True, exist_ok=True)
        tmp = spath.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
        tmp.replace(spath)
    except Exception as exc:
        logger.debug("write_realtime_status_snapshot failed open: %s", exc)
    return summary


def quarantine_session(session_id: str, reason: str, net_eur_window: float, ttl_s: float = 3600.0, now: float | None = None) -> None:
    ts_now = float(now if now is not None else time.time())
    try:
        con = _connect()
        try:
            with con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO quarantined_sessions (
                        session_id, quarantined_at, expires_at, reason, net_eur_window
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, ts_now, ts_now + ttl_s, reason, float(net_eur_window)),
                )
        finally:
            con.close()
    except Exception as exc:
        logger.debug("quarantine_session failed open: %s", exc)


def evaluate_realtime_spend_breaker(
    *,
    session_id: str,
    source: str,
    model: str = "",
    provider: str = "",
    job_name: str = "",
    making_productive_progress: bool = True,
    consecutive_calls_since_user: int = 0,
    now: float | None = None,
) -> dict[str, Any]:
    ts_now = float(now if now is not None else time.time())
    day_cet = _day_cet(ts_now)
    src = (source or "desktop").strip().lower()
    is_interactive = src in ("desktop", "mobile", "tui", "tui_gateway", "gui", "web", "cli", "interactive")
    is_tier0 = is_tier0_critical_cron(job_name, session_id)

    try:
        con = _connect()
        try:
            cur = con.cursor()
            # 1) Check explicit quarantine table
            cur.execute(
                "SELECT reason, net_eur_window FROM quarantined_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, ts_now),
            )
            q_row = cur.fetchone()
            if q_row is not None:
                if not is_interactive or not making_productive_progress:
                    return {
                        "allowed": False,
                        "reason": f"quarantined:{q_row[0]}",
                        "session_net_eur_15m": float(q_row[1]),
                    }

            # 2) Per-session 15m and 1h spend + cache resets
            cur.execute(
                """
                SELECT COALESCE(SUM(net_eur), 0.0), COALESCE(SUM(cache_reset), 0)
                FROM llm_call_ledger WHERE session_id = ? AND ts >= ?
                """,
                (session_id, ts_now - 900.0),
            )
            s_eur_15m, s_resets_15m = cur.fetchone()
            s_eur_15m = float(s_eur_15m or 0.0)
            s_resets_15m = int(s_resets_15m or 0)

            cur.execute(
                "SELECT COALESCE(SUM(net_eur), 0.0) FROM llm_call_ledger WHERE session_id = ? AND ts >= ?",
                (session_id, ts_now - 3600.0),
            )
            s_eur_1h = float(cur.fetchone()[0] or 0.0)

            # 3) Estate 15m and Today spend
            cur.execute(
                "SELECT COALESCE(SUM(net_eur), 0.0) FROM llm_call_ledger WHERE ts >= ? AND source != 'log_tail'",
                (ts_now - 900.0,),
            )
            estate_15m = float(cur.fetchone()[0] or 0.0)

            cur.execute(
                "SELECT COALESCE(SUM(net_eur), 0.0) FROM llm_call_ledger WHERE day_cet = ? AND source != 'log_tail'",
                (day_cet,),
            )
            estate_today = float(cur.fetchone()[0] or 0.0)
        finally:
            con.close()
    except Exception as exc:
        logger.debug("evaluate_realtime_spend_breaker failed open: %s", exc)
        return {"allowed": True, "reason": "fail_open_db_error"}

    # Tier 1A: Non-interactive per-session runaway breaker
    if not is_interactive:
        cap_15m = TIER0_CRITICAL_CRON_15M_SAFETY_CAP_EUR if is_tier0 else TIER1_NON_INTERACTIVE_15M_CAP_EUR
        if s_eur_15m >= cap_15m:
            reason = f"tier1_session_15m_cap (€{s_eur_15m:.2f} >= €{cap_15m:.2f})"
            quarantine_session(session_id, reason, s_eur_15m, ttl_s=1800.0, now=ts_now)
            return {"allowed": False, "reason": reason, "session_net_eur_15m": s_eur_15m}
        if not is_tier0 and s_eur_1h >= TIER1_NON_INTERACTIVE_1H_CAP_EUR:
            reason = f"tier1_session_1h_cap (€{s_eur_1h:.2f} >= €{TIER1_NON_INTERACTIVE_1H_CAP_EUR:.2f})"
            quarantine_session(session_id, reason, s_eur_1h, ttl_s=3600.0, now=ts_now)
            return {"allowed": False, "reason": reason, "session_net_eur_1h": s_eur_1h}
        if (
            not is_tier0
            and s_resets_15m >= TIER1_NON_INTERACTIVE_CACHE_RESET_COUNT
            and s_eur_15m >= TIER1_NON_INTERACTIVE_CACHE_RESET_MIN_EUR
        ):
            reason = f"tier1_cache_reset_storm (resets={s_resets_15m}, €{s_eur_15m:.2f} in 15m)"
            quarantine_session(session_id, reason, s_eur_15m, ttl_s=1800.0, now=ts_now)
            return {"allowed": False, "reason": reason, "session_net_eur_15m": s_eur_15m}

    # Tier 1B: Progress-aware Desktop stalled loop breaker
    if is_interactive:
        if (
            not making_productive_progress
            and consecutive_calls_since_user >= TIER1_STALLED_DESKTOP_MIN_CALLS
            and s_eur_15m >= TIER1_STALLED_DESKTOP_15M_CAP_EUR
        ):
            reason = (
                f"tier1_stalled_desktop_loop (calls={consecutive_calls_since_user}, "
                f"€{s_eur_15m:.2f} in 15m with no productive progress)"
            )
            quarantine_session(session_id, reason, s_eur_15m, ttl_s=900.0, now=ts_now)
            return {"allowed": False, "reason": reason, "session_net_eur_15m": s_eur_15m}
        # Interactive Desktop sessions with productive progress are NEVER blocked by Tier 2 or Tier 3
        return {
            "allowed": True,
            "reason": "interactive_allowed",
            "session_net_eur_15m": s_eur_15m,
            "estate_net_eur_today": estate_today,
        }

    # Tier-0 critical operational crons are exempt from Tier 2 & Tier 3 estate caps
    if is_tier0:
        return {
            "allowed": True,
            "reason": "tier0_critical_cron_exempt",
            "session_net_eur_15m": s_eur_15m,
            "estate_net_eur_today": estate_today,
        }

    # Manual Home Assistant Emergency Brake Override blocks all non-Tier-0 non-interactive sessions
    if is_manual_emergency_brake_active():
        return {
            "allowed": False,
            "reason": "manual_ha_emergency_brake (Home Assistant Emergency Brake switch is ON)",
            "estate_net_eur_today": estate_today,
        }

    # Tier 3: Daily Hard Cap (€195/day net) blocks all non-Tier-0 non-interactive sessions
    if estate_today >= TIER3_DAILY_HARD_CAP_EUR:
        return {
            "allowed": False,
            "reason": f"tier3_daily_hard_cap (estate today €{estate_today:.2f} >= €{TIER3_DAILY_HARD_CAP_EUR:.2f})",
            "estate_net_eur_today": estate_today,
        }

    # Tier 2: Daily Soft Cap (€150/day net) or 15m Burst (€8/15m) suppresses background_review and non-Tier-0 crons
    if src in ("background_review", "cron") and (
        estate_today >= TIER2_DAILY_SOFT_CAP_EUR or estate_15m >= TIER2_ESTATE_15M_BURST_EUR
    ):
        return {
            "allowed": False,
            "reason": (
                f"tier2_daily_soft_cap (estate today €{estate_today:.2f} / 15m €{estate_15m:.2f})"
            ),
            "estate_net_eur_today": estate_today,
        }

    return {
        "allowed": True,
        "reason": "ok",
        "session_net_eur_15m": s_eur_15m,
        "estate_net_eur_today": estate_today,
    }
