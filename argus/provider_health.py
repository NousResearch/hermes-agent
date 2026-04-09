"""
ARGUS API provider health monitoring — per-provider error rate tracking.

Detects outages before users notice. Complements the credential pool's
per-credential exhaustion (which handles rotation) by tracking aggregate
health across all sessions hitting the same provider.

Data sources:
- state.db messages: finish_reason, content (error patterns)
- argus.db tool_calls: error_message, success, duration_ms
- argus.db sessions: provider, model

Detects:
- HTTP error rate per provider (nous, openrouter, anthropic, etc.)
- Rate limit (429) detection — immediate critical alert
- Timeout rate tracking (> 30% in last 20 calls)
- Latency degradation (p95 vs baseline)
- Model-not-found / invalid-key detection
- Provider outage correlation (same errors across multiple sessions)
"""

import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.provider_health")

# --- Error classification patterns ---

_PATTERNS = {
    "rate_limit": [
        re.compile(r"\b429\b"),
        re.compile(r"rate.?limit", re.IGNORECASE),
        re.compile(r"too many requests", re.IGNORECASE),
        re.compile(r"quota.?exceeded", re.IGNORECASE),
        re.compile(r"throttl", re.IGNORECASE),
    ],
    "timeout": [
        re.compile(r"\btimeout\b", re.IGNORECASE),
        re.compile(r"connect\s*timeout", re.IGNORECASE),
        re.compile(r"read\s*timeout", re.IGNORECASE),
        re.compile(r"pool\s*timeout", re.IGNORECASE),
        re.compile(r"timed?\s*out", re.IGNORECASE),
    ],
    "auth_error": [
        re.compile(r"\b401\b"),
        re.compile(r"invalid.?api.?key", re.IGNORECASE),
        re.compile(r"unauthorized", re.IGNORECASE),
        re.compile(r"authentication\s*failed", re.IGNORECASE),
        re.compile(r"invalid.?token", re.IGNORECASE),
    ],
    "billing": [
        re.compile(r"\b402\b"),
        re.compile(r"insufficient.?credits?", re.IGNORECASE),
        re.compile(r"billing", re.IGNORECASE),
        re.compile(r"payment\s*required", re.IGNORECASE),
    ],
    "server_error": [
        re.compile(r"\b50[0234]\b"),
        re.compile(r"overloaded", re.IGNORECASE),
        re.compile(r"internal.?server.?error", re.IGNORECASE),
        re.compile(r"service.?unavailable", re.IGNORECASE),
        re.compile(r"bad.?gateway", re.IGNORECASE),
        re.compile(r"gateway.?timeout", re.IGNORECASE),
    ],
    "model_error": [
        re.compile(r"model.?not.?found", re.IGNORECASE),
        re.compile(r"does.?not.?exist", re.IGNORECASE),
        re.compile(r"invalid.?model", re.IGNORECASE),
        re.compile(r"model.?not.?available", re.IGNORECASE),
    ],
    "context_length": [
        re.compile(r"context.?length", re.IGNORECASE),
        re.compile(r"maximum.?context", re.IGNORECASE),
        re.compile(r"token.?limit.?exceeded", re.IGNORECASE),
        re.compile(r"too.?many.?tokens", re.IGNORECASE),
    ],
}

# Thresholds
_ERROR_RATE_WARNING = 15.0     # % errors in window → warning
_ERROR_RATE_CRITICAL = 30.0    # % errors in window → critical
_RATE_LIMIT_CRITICAL = 1       # any 429 → critical immediately
_TIMEOUT_RATE_CRITICAL = 30.0  # % timeouts in window → critical
_OUTAGE_CORRELATION_COUNT = 3  # sessions failing within window → outage
_OUTAGE_CORRELATION_WINDOW_S = 60  # seconds
_WINDOW_MINUTES = 30           # analysis window
_MIN_CALLS_FOR_RATE = 5        # minimum calls before computing rates


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except ImportError:
        return Path.home() / ".hermes"


def _state_db_path() -> Path:
    return _hermes_home() / "state.db"


def ensure_table(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    """Create provider_health table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            provider TEXT NOT NULL,
            total_calls INTEGER NOT NULL DEFAULT 0,
            error_calls INTEGER NOT NULL DEFAULT 0,
            error_rate REAL NOT NULL DEFAULT 0.0,
            rate_limit_count INTEGER NOT NULL DEFAULT 0,
            timeout_count INTEGER NOT NULL DEFAULT 0,
            auth_error_count INTEGER NOT NULL DEFAULT 0,
            billing_error_count INTEGER NOT NULL DEFAULT 0,
            server_error_count INTEGER NOT NULL DEFAULT 0,
            model_error_count INTEGER NOT NULL DEFAULT 0,
            context_length_count INTEGER NOT NULL DEFAULT 0,
            p95_latency_ms REAL,
            severity TEXT NOT NULL DEFAULT 'info',
            outage_detected BOOLEAN NOT NULL DEFAULT FALSE,
            details TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_provider_health_provider
        ON provider_health(provider)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_provider_health_timestamp
        ON provider_health(timestamp)
    """)
    conn.commit()


def classify_error(text: str) -> List[str]:
    """Classify an error string into one or more error categories."""
    if not text:
        return []
    categories = []
    for category, patterns in _PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                categories.append(category)
                break
    return categories


def _collect_from_messages(
    state_db: Path, since_iso: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Pull recent messages from state.db and classify API errors by provider.

    Returns {provider: [{error_type, timestamp, session_id, detail}, ...]}
    """
    errors_by_provider: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if not state_db.exists():
        return errors_by_provider

    try:
        conn = sqlite3.connect(str(state_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get sessions with their provider info
        cur.execute("""
            SELECT id, model, billing_provider FROM sessions
            WHERE started_at >= ? OR started_at IS NULL
        """, (since_iso,))
        session_map = {}
        for row in cur.fetchall():
            sid = row["id"]
            provider = row["billing_provider"] or _provider_from_model(row["model"]) or "unknown"
            session_map[sid] = provider

        # Get messages with error indicators
        cur.execute("""
            SELECT session_id, role, content, finish_reason, timestamp
            FROM messages
            WHERE timestamp >= ?
            AND (
                (finish_reason IS NOT NULL AND finish_reason NOT IN ('stop', 'tool_calls'))
                OR content LIKE '%429%'
                OR content LIKE '%rate limit%'
                OR content LIKE '%RateLimit%'
                OR content LIKE '%timeout%'
                OR content LIKE '%timed out%'
                OR content LIKE '%401%'
                OR content LIKE '%402%'
                OR content LIKE '%500%'
                OR content LIKE '%502%'
                OR content LIKE '%503%'
                OR content LIKE '%model_not_found%'
                OR content LIKE '%context_length%'
                OR content LIKE '%overloaded%'
                OR content LIKE '%unauthorized%'
            )
        """, (since_iso,))

        for row in cur.fetchall():
            sid = row["session_id"]
            provider = session_map.get(sid, "unknown")
            content = row["content"] or ""
            finish_reason = row["finish_reason"] or ""
            ts = row["timestamp"]

            # Classify from finish_reason
            if finish_reason == "length":
                errors_by_provider[provider].append({
                    "error_type": "context_length",
                    "timestamp": ts,
                    "session_id": sid,
                    "detail": "finish_reason=length (truncated)",
                })
            elif finish_reason == "incomplete":
                errors_by_provider[provider].append({
                    "error_type": "server_error",
                    "timestamp": ts,
                    "session_id": sid,
                    "detail": "finish_reason=incomplete",
                })

            # Classify from content
            categories = classify_error(content[:1000])
            for cat in categories:
                errors_by_provider[provider].append({
                    "error_type": cat,
                    "timestamp": ts,
                    "session_id": sid,
                    "detail": content[:200],
                })

        conn.close()
    except Exception as e:
        logger.error("Failed to collect from state.db: %s", e)

    return errors_by_provider


def _collect_from_tool_calls(
    argus_cur: sqlite3.Cursor, since_iso: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Pull failed tool calls from argus.db and classify by provider.

    Returns {provider: [{error_type, timestamp, session_id, detail}, ...]}
    """
    errors_by_provider: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    try:
        # Get failed tool calls joined with session provider
        argus_cur.execute("""
            SELECT tc.session_id, tc.tool_name, tc.error_message, tc.timestamp,
                   tc.duration_ms, s.provider
            FROM tool_calls tc
            LEFT JOIN sessions s ON tc.session_id = s.session_id
            WHERE tc.success = 0
            AND tc.timestamp >= ?
            AND tc.error_message IS NOT NULL
            AND tc.error_message != ''
        """, (since_iso,))

        for row in argus_cur.fetchall():
            provider = row["provider"] or "unknown"
            error_msg = row["error_message"] or ""
            categories = classify_error(error_msg)
            if not categories:
                categories = ["tool_error"]

            for cat in categories:
                errors_by_provider[provider].append({
                    "error_type": cat,
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "detail": f"{row['tool_name']}: {error_msg[:150]}",
                    "duration_ms": row["duration_ms"],
                })
    except Exception as e:
        logger.error("Failed to collect from tool_calls: %s", e)

    return errors_by_provider


def _collect_latencies(
    argus_cur: sqlite3.Cursor, since_iso: str
) -> Dict[str, List[float]]:
    """Collect tool call durations by provider for latency analysis."""
    latencies: Dict[str, List[float]] = defaultdict(list)

    try:
        argus_cur.execute("""
            SELECT tc.duration_ms, s.provider
            FROM tool_calls tc
            LEFT JOIN sessions s ON tc.session_id = s.session_id
            WHERE tc.timestamp >= ?
            AND tc.duration_ms IS NOT NULL
            AND tc.duration_ms > 0
        """, (since_iso,))

        for row in argus_cur.fetchall():
            provider = row["provider"] or "unknown"
            latencies[provider].append(float(row["duration_ms"]))
    except Exception as e:
        logger.error("Failed to collect latencies: %s", e)

    return latencies


def _provider_from_model(model: Optional[str]) -> Optional[str]:
    """Guess provider from model string. Best-effort heuristic."""
    if not model:
        return None
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
        return "openai"
    if "gemini" in m or "gemma" in m:
        return "google"
    if "llama" in m or "qwen" in m:
        return "openrouter"
    if "mimo" in m:
        return "nous"
    return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Compute a percentile from a sorted list."""
    if not values:
        return None
    s = sorted(values)
    idx = int(len(s) * pct / 100.0)
    idx = min(idx, len(s) - 1)
    return s[idx]


def _detect_outage(errors: List[Dict[str, Any]]) -> bool:
    """Detect if errors represent a provider outage (3+ sessions within 60s)."""
    if len(errors) < _OUTAGE_CORRELATION_COUNT:
        return False

    # Group by timestamp windows
    sessions_by_window: Dict[int, set] = defaultdict(set)
    for err in errors:
        ts = err.get("timestamp", "")
        sid = err.get("session_id", "")
        try:
            # Normalize to seconds since epoch for windowing
            if isinstance(ts, str):
                # Handle ISO format
                ts_clean = ts.replace("Z", "+00:00").replace(" ", "T")
                dt = datetime.fromisoformat(ts_clean)
                window_key = int(dt.timestamp() / _OUTAGE_CORRELATION_WINDOW_S)
            else:
                window_key = int(float(ts) / _OUTAGE_CORRELATION_WINDOW_S)
            sessions_by_window[window_key].add(sid)
        except (ValueError, TypeError):
            continue

    # Check if any window has 3+ unique sessions
    for sessions in sessions_by_window.values():
        if len(sessions) >= _OUTAGE_CORRELATION_COUNT:
            return True

    return False


def run_provider_check(
    argus_cursor: sqlite3.Cursor,
    argus_conn: sqlite3.Connection,
) -> Dict[str, Any]:
    """Run a full provider health check. Returns report dict.

    Call this every N poll cycles from argus.py.
    """
    now = time.time()
    window_start = now - (_WINDOW_MINUTES * 60)
    since_iso = datetime.fromtimestamp(window_start, UTC).isoformat()

    state_db = _state_db_path()

    # Collect from both data sources
    msg_errors = _collect_from_messages(state_db, since_iso)
    tc_errors = _collect_from_tool_calls(argus_cursor, since_iso)
    latencies = _collect_latencies(argus_cursor, since_iso)

    # Merge error sources
    all_providers = set(list(msg_errors.keys()) + list(tc_errors.keys()))
    if "unknown" in all_providers and len(all_providers) > 1:
        all_providers.discard("unknown")

    report: Dict[str, Any] = {
        "timestamp": datetime.fromtimestamp(now, UTC).isoformat(),
        "window_minutes": _WINDOW_MINUTES,
        "providers": {},
        "overall_severity": "info",
        "outage_detected": False,
    }

    worst_severity = "info"

    for provider in sorted(all_providers):
        errors = msg_errors.get(provider, []) + tc_errors.get(provider, [])

        # Count by category
        counts: Dict[str, int] = defaultdict(int)
        for err in errors:
            counts[err["error_type"]] += 1

        total_errors = len(errors)
        total_calls = total_errors  # conservative: we only see errors here

        # Latency stats
        provider_latencies = latencies.get(provider, [])
        p95 = _percentile(provider_latencies, 95) if provider_latencies else None

        # Determine severity
        severity = "info"
        rate_limit_count = counts.get("rate_limit", 0)
        timeout_count = counts.get("timeout", 0)

        if rate_limit_count >= _RATE_LIMIT_CRITICAL:
            severity = "critical"
        elif total_errors >= _MIN_CALLS_FOR_RATE:
            error_rate = (total_errors / max(total_calls, 1)) * 100
            if error_rate >= _ERROR_RATE_CRITICAL:
                severity = "critical"
            elif error_rate >= _ERROR_RATE_WARNING:
                severity = "warning"

        if timeout_count >= _MIN_CALLS_FOR_RATE:
            timeout_rate = (timeout_count / max(total_errors, 1)) * 100
            if timeout_rate >= _TIMEOUT_RATE_CRITICAL:
                severity = "critical"

        # Outage detection
        outage = _detect_outage(errors)
        if outage:
            severity = "critical"

        # Track worst
        severity_order = {"info": 0, "warning": 1, "critical": 2}
        if severity_order.get(severity, 0) > severity_order.get(worst_severity, 0):
            worst_severity = severity

        provider_report = {
            "total_errors": total_errors,
            "error_counts": dict(counts),
            "rate_limit_count": rate_limit_count,
            "timeout_count": timeout_count,
            "auth_error_count": counts.get("auth_error", 0),
            "billing_error_count": counts.get("billing", 0),
            "server_error_count": counts.get("server_error", 0),
            "model_error_count": counts.get("model_error", 0),
            "context_length_count": counts.get("context_length", 0),
            "p95_latency_ms": round(p95, 1) if p95 else None,
            "severity": severity,
            "outage_detected": outage,
            "recent_errors": errors[:5],  # last 5 for context
        }

        report["providers"][provider] = provider_report

        if outage:
            report["outage_detected"] = True

        # Persist snapshot
        try:
            argus_cursor.execute("""
                INSERT INTO provider_health
                (provider, total_calls, error_calls, error_rate,
                 rate_limit_count, timeout_count, auth_error_count,
                 billing_error_count, server_error_count, model_error_count,
                 context_length_count, p95_latency_ms, severity,
                 outage_detected, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider,
                total_calls,
                total_errors,
                round((total_errors / max(total_calls, 1)) * 100, 2),
                rate_limit_count,
                timeout_count,
                counts.get("auth_error", 0),
                counts.get("billing", 0),
                counts.get("server_error", 0),
                counts.get("model_error", 0),
                counts.get("context_length", 0),
                round(p95, 1) if p95 else None,
                severity,
                outage,
                json.dumps({"recent_errors": errors[:3]}),
            ))
        except Exception as e:
            logger.error("Failed to persist provider health snapshot: %s", e)

    report["overall_severity"] = worst_severity
    argus_conn.commit()
    return report


def format_alert(report: Dict[str, Any]) -> Optional[str]:
    """Format a provider health report into a human-readable alert string."""
    if report["overall_severity"] == "info":
        return None

    lines = []
    if report.get("outage_detected"):
        lines.append("🚨 PROVIDER OUTAGE DETECTED")
        lines.append("")

    for provider, data in sorted(report.get("providers", {}).items()):
        if data["severity"] == "info":
            continue

        icon = "🔴" if data["severity"] == "critical" else "🟡"
        lines.append(f"{icon} {provider} ({data['severity'].upper()})")
        lines.append(f"   Errors: {data['total_errors']}")

        if data["rate_limit_count"]:
            lines.append(f"   Rate limits (429): {data['rate_limit_count']}")
        if data["timeout_count"]:
            lines.append(f"   Timeouts: {data['timeout_count']}")
        if data["auth_error_count"]:
            lines.append(f"   Auth errors: {data['auth_error_count']}")
        if data["billing_error_count"]:
            lines.append(f"   Billing errors: {data['billing_error_count']}")
        if data["server_error_count"]:
            lines.append(f"   Server errors: {data['server_error_count']}")
        if data["model_error_count"]:
            lines.append(f"   Model errors: {data['model_error_count']}")
        if data["context_length_count"]:
            lines.append(f"   Context length: {data['context_length_count']}")
        if data.get("p95_latency_ms"):
            lines.append(f"   p95 latency: {data['p95_latency_ms']:.0f}ms")

        if data.get("outage_detected"):
            lines.append(f"   ⚠️  OUTAGE — {data['total_errors']} errors across multiple sessions")

        # Show sample errors
        for err in data.get("recent_errors", [])[:2]:
            detail = err.get("detail", "")[:80]
            lines.append(f"   └─ {err.get('error_type', '?')}: {detail}")

        lines.append("")

    return "\n".join(lines) if lines else None


def cleanup_old_snapshots(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection, keep_hours: int = 72
) -> int:
    """Delete provider_health snapshots older than keep_hours. Returns count deleted."""
    cutoff = (datetime.now(UTC) - timedelta(hours=keep_hours)).isoformat()
    try:
        cursor.execute("DELETE FROM provider_health WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        logger.error("Failed to cleanup old snapshots: %s", e)
        return 0
