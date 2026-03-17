"""Audit event logger — structured, append-only record of all agent actions.

Captures tool calls, API calls, errors, security events, and session lifecycle
in a separate SQLite database (~/.hermes/audit.db). Designed for:
- Post-incident investigation ("what happened?")
- Problem detection (repeated errors, cost spikes, security anomalies)
- Usage reporting (per-user, per-model, per-tool breakdowns)

The audit log is write-only from the agent's perspective — it never reads
its own audit trail during operation, so it has zero impact on agent behavior.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Event severity levels
SEVERITY_DEBUG = "debug"
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"

# Event type constants
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_ERROR = "tool_error"
EVENT_API_CALL = "api_call"
EVENT_API_ERROR = "api_error"
EVENT_API_RETRY = "api_retry"
EVENT_SESSION_START = "session_start"
EVENT_SESSION_END = "session_end"
EVENT_AUTH_REFRESH = "auth_refresh"
EVENT_AUTH_FAILURE = "auth_failure"
EVENT_APPROVAL_REQUEST = "approval_request"
EVENT_APPROVAL_RESULT = "approval_result"
EVENT_COMPRESSION = "compression"
EVENT_FALLBACK = "fallback_activated"
EVENT_COST = "cost_estimate"

_HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
_AUDIT_DB_PATH = _HERMES_HOME / "audit.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    session_id TEXT,
    user_id TEXT,
    platform TEXT,
    model TEXT,
    provider TEXT,

    -- Tool call fields
    tool_name TEXT,
    tool_args TEXT,
    tool_result_preview TEXT,
    duration_ms REAL,
    success INTEGER,

    -- API call fields
    status_code INTEGER,
    request_id TEXT,
    retry_count INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,

    -- Error fields
    error_type TEXT,
    error_message TEXT,

    -- Flexible context (JSON blob for event-specific data)
    context TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool_name);
CREATE INDEX IF NOT EXISTS idx_events_error ON events(event_type, error_type);
"""


class AuditLogger:
    """Thread-safe, append-only audit event logger backed by SQLite.

    Usage::

        audit = AuditLogger()
        audit.log_tool_call("terminal", {"command": "ls"}, result="file.txt", duration_ms=120)
        audit.log_api_error(status_code=500, error="Internal server error")
        events = audit.query(event_type="tool_error", last_hours=24)
        audit.close()
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _AUDIT_DB_PATH
        self._lock = threading.Lock()
        self._conn = None
        self._closed = False
        self._init_db()

    def _init_db(self):
        import sqlite3
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=5,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        except Exception as e:
            logger.debug("Failed to initialize audit database: %s", e)
            self._conn = None

    def _insert(self, **fields):
        if not self._conn or self._closed:
            return
        fields.setdefault("timestamp", time.time())
        cols = list(fields.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [fields[c] for c in cols]
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO events ({col_names}) VALUES ({placeholders})",
                    values,
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("Audit log write failed: %s", e)

    # ── Logging methods ──────────────────────────────────────────────

    def log_tool_call(
        self,
        tool_name: str,
        args: Any = None,
        result: str = None,
        duration_ms: float = None,
        success: bool = True,
        session_id: str = None,
        user_id: str = None,
        platform: str = None,
        context: Dict = None,
    ):
        args_str = json.dumps(args, ensure_ascii=False, default=str)[:2000] if args else None
        result_preview = str(result)[:500] if result else None
        self._insert(
            event_type=EVENT_TOOL_CALL,
            severity=SEVERITY_INFO,
            tool_name=tool_name,
            tool_args=args_str,
            tool_result_preview=result_preview,
            duration_ms=duration_ms,
            success=1 if success else 0,
            session_id=session_id,
            user_id=user_id,
            platform=platform,
            context=json.dumps(context, default=str) if context else None,
        )

    def log_tool_error(
        self,
        tool_name: str,
        args: Any = None,
        error: str = None,
        error_type: str = None,
        duration_ms: float = None,
        session_id: str = None,
        user_id: str = None,
        platform: str = None,
    ):
        args_str = json.dumps(args, ensure_ascii=False, default=str)[:2000] if args else None
        self._insert(
            event_type=EVENT_TOOL_ERROR,
            severity=SEVERITY_ERROR,
            tool_name=tool_name,
            tool_args=args_str,
            duration_ms=duration_ms,
            success=0,
            error_type=error_type or "Exception",
            error_message=str(error)[:1000] if error else None,
            session_id=session_id,
            user_id=user_id,
            platform=platform,
        )

    def log_api_call(
        self,
        model: str = None,
        provider: str = None,
        status_code: int = None,
        request_id: str = None,
        duration_ms: float = None,
        input_tokens: int = None,
        output_tokens: int = None,
        retry_count: int = 0,
        session_id: str = None,
        context: Dict = None,
    ):
        self._insert(
            event_type=EVENT_API_CALL,
            severity=SEVERITY_INFO,
            model=model,
            provider=provider,
            status_code=status_code,
            request_id=request_id,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            session_id=session_id,
            context=json.dumps(context, default=str) if context else None,
        )

    def log_api_error(
        self,
        model: str = None,
        provider: str = None,
        status_code: int = None,
        request_id: str = None,
        error: str = None,
        error_type: str = None,
        retry_count: int = 0,
        duration_ms: float = None,
        session_id: str = None,
    ):
        self._insert(
            event_type=EVENT_API_ERROR,
            severity=SEVERITY_ERROR if status_code and status_code >= 500 else SEVERITY_WARNING,
            model=model,
            provider=provider,
            status_code=status_code,
            request_id=request_id,
            error_type=error_type,
            error_message=str(error)[:1000] if error else None,
            retry_count=retry_count,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    def log_session_event(
        self,
        event_type: str,
        session_id: str = None,
        user_id: str = None,
        platform: str = None,
        model: str = None,
        provider: str = None,
        context: Dict = None,
    ):
        self._insert(
            event_type=event_type,
            severity=SEVERITY_INFO,
            session_id=session_id,
            user_id=user_id,
            platform=platform,
            model=model,
            provider=provider,
            context=json.dumps(context, default=str) if context else None,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str = SEVERITY_WARNING,
        session_id: str = None,
        user_id: str = None,
        platform: str = None,
        context: Dict = None,
    ):
        self._insert(
            event_type=event_type,
            severity=severity,
            session_id=session_id,
            user_id=user_id,
            platform=platform,
            context=json.dumps(context, default=str) if context else None,
        )

    # ── Query methods ────────────────────────────────────────────────

    def query(
        self,
        event_type: str = None,
        severity: str = None,
        session_id: str = None,
        tool_name: str = None,
        last_hours: float = None,
        limit: int = 100,
        errors_only: bool = False,
    ) -> List[Dict]:
        if not self._conn:
            return []
        conditions = []
        params = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if last_hours:
            conditions.append("timestamp > ?")
            params.append(time.time() - (last_hours * 3600))
        if errors_only:
            conditions.append("success = 0 OR severity IN ('error', 'critical')")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug("Audit query failed: %s", e)
            return []

    def summary(self, last_hours: float = 24) -> Dict:
        """Generate a summary of events in the given time window."""
        if not self._conn:
            return {}
        since = time.time() - (last_hours * 3600)
        try:
            with self._lock:
                cur = self._conn.execute
                total = cur("SELECT COUNT(*) FROM events WHERE timestamp > ?", (since,)).fetchone()[0]
                errors = cur("SELECT COUNT(*) FROM events WHERE timestamp > ? AND severity IN ('error', 'critical')", (since,)).fetchone()[0]
                tool_calls = cur("SELECT COUNT(*) FROM events WHERE timestamp > ? AND event_type = 'tool_call'", (since,)).fetchone()[0]
                api_calls = cur("SELECT COUNT(*) FROM events WHERE timestamp > ? AND event_type = 'api_call'", (since,)).fetchone()[0]
                api_errors = cur("SELECT COUNT(*) FROM events WHERE timestamp > ? AND event_type = 'api_error'", (since,)).fetchone()[0]

                # Top errors
                top_errors = cur(
                    "SELECT error_type, error_message, COUNT(*) as cnt "
                    "FROM events WHERE timestamp > ? AND severity IN ('error', 'critical') "
                    "GROUP BY error_type, error_message ORDER BY cnt DESC LIMIT 5",
                    (since,)
                ).fetchall()

                # Top tools by usage
                top_tools = cur(
                    "SELECT tool_name, COUNT(*) as cnt, AVG(duration_ms) as avg_ms "
                    "FROM events WHERE timestamp > ? AND event_type = 'tool_call' AND tool_name IS NOT NULL "
                    "GROUP BY tool_name ORDER BY cnt DESC LIMIT 10",
                    (since,)
                ).fetchall()

            return {
                "period_hours": last_hours,
                "total_events": total,
                "errors": errors,
                "tool_calls": tool_calls,
                "api_calls": api_calls,
                "api_errors": api_errors,
                "error_rate": f"{(errors / total * 100):.1f}%" if total else "0%",
                "top_errors": [
                    {"type": e[0], "message": e[1][:100] if e[1] else "", "count": e[2]}
                    for e in top_errors
                ],
                "top_tools": [
                    {"name": t[0], "calls": t[1], "avg_ms": round(t[2], 1) if t[2] else None}
                    for t in top_tools
                ],
            }
        except Exception as e:
            logger.debug("Audit summary failed: %s", e)
            return {}

    def detect_problems(self, last_hours: float = 24) -> List[Dict]:
        """Analyze recent events and detect patterns that indicate problems.

        Returns a list of problem dicts with: type, severity, message, evidence.
        """
        if not self._conn:
            return []
        problems = []
        since = time.time() - (last_hours * 3600)

        try:
            with self._lock:
                cur = self._conn.execute

                # 1. Repeated API errors (same error_type 3+ times)
                repeated_errors = cur(
                    "SELECT error_type, status_code, COUNT(*) as cnt "
                    "FROM events WHERE timestamp > ? AND event_type = 'api_error' "
                    "AND error_type IS NOT NULL "
                    "GROUP BY error_type, status_code HAVING cnt >= 3 "
                    "ORDER BY cnt DESC",
                    (since,)
                ).fetchall()
                for err_type, status, count in repeated_errors:
                    problems.append({
                        "type": "repeated_api_error",
                        "severity": "error",
                        "message": f"{err_type} (HTTP {status}) occurred {count} times",
                        "evidence": {"error_type": err_type, "status_code": status, "count": count},
                    })

                # 2. High API error rate (>25% of API calls failed)
                api_total = cur(
                    "SELECT COUNT(*) FROM events WHERE timestamp > ? "
                    "AND event_type IN ('api_call', 'api_error')",
                    (since,)
                ).fetchone()[0]
                api_errors = cur(
                    "SELECT COUNT(*) FROM events WHERE timestamp > ? AND event_type = 'api_error'",
                    (since,)
                ).fetchone()[0]
                if api_total >= 4 and api_errors / api_total > 0.25:
                    problems.append({
                        "type": "high_api_error_rate",
                        "severity": "warning",
                        "message": f"API error rate {api_errors}/{api_total} ({api_errors/api_total*100:.0f}%)",
                        "evidence": {"total": api_total, "errors": api_errors},
                    })

                # 3. Slow tool calls (avg > 10s for any tool)
                slow_tools = cur(
                    "SELECT tool_name, AVG(duration_ms) as avg_ms, COUNT(*) as cnt "
                    "FROM events WHERE timestamp > ? AND event_type = 'tool_call' "
                    "AND duration_ms IS NOT NULL AND tool_name IS NOT NULL "
                    "GROUP BY tool_name HAVING avg_ms > 10000 AND cnt >= 2",
                    (since,)
                ).fetchall()
                for tool, avg_ms, count in slow_tools:
                    problems.append({
                        "type": "slow_tool",
                        "severity": "warning",
                        "message": f"{tool} averaging {avg_ms/1000:.1f}s over {count} calls",
                        "evidence": {"tool": tool, "avg_ms": round(avg_ms, 1), "count": count},
                    })

                # 4. Tool failures (same tool failing 3+ times)
                failing_tools = cur(
                    "SELECT tool_name, COUNT(*) as cnt "
                    "FROM events WHERE timestamp > ? AND event_type = 'tool_error' "
                    "AND tool_name IS NOT NULL "
                    "GROUP BY tool_name HAVING cnt >= 3",
                    (since,)
                ).fetchall()
                for tool, count in failing_tools:
                    problems.append({
                        "type": "repeated_tool_failure",
                        "severity": "error",
                        "message": f"{tool} failed {count} times",
                        "evidence": {"tool": tool, "count": count},
                    })

                # 5. Security: denied commands
                denied = cur(
                    "SELECT COUNT(*) FROM events WHERE timestamp > ? "
                    "AND event_type = 'approval_result' AND severity = 'warning'",
                    (since,)
                ).fetchone()[0]
                if denied >= 3:
                    problems.append({
                        "type": "frequent_denials",
                        "severity": "warning",
                        "message": f"{denied} commands denied by approval system",
                        "evidence": {"denied_count": denied},
                    })

                # 6. Rate limiting detected (429 errors)
                rate_limits = cur(
                    "SELECT COUNT(*) FROM events WHERE timestamp > ? "
                    "AND event_type = 'api_error' AND status_code = 429",
                    (since,)
                ).fetchone()[0]
                if rate_limits >= 2:
                    problems.append({
                        "type": "rate_limited",
                        "severity": "warning",
                        "message": f"Rate limited {rate_limits} times — consider reducing request frequency",
                        "evidence": {"count": rate_limits},
                    })

                # 7. Auth failures (401 errors)
                auth_fails = cur(
                    "SELECT COUNT(*) FROM events WHERE timestamp > ? "
                    "AND event_type IN ('api_error', 'auth_failure') AND status_code = 401",
                    (since,)
                ).fetchone()[0]
                if auth_fails >= 1:
                    problems.append({
                        "type": "auth_failure",
                        "severity": "error",
                        "message": f"{auth_fails} authentication failure(s) — check API key or OAuth credentials",
                        "evidence": {"count": auth_fails},
                    })

        except Exception as e:
            logger.debug("Problem detection failed: %s", e)

        # Sort by severity (critical > error > warning > info)
        severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        problems.sort(key=lambda p: severity_order.get(p["severity"], 9))
        return problems

    def close(self):
        self._closed = True
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


# ── Global singleton ─────────────────────────────────────────────────

_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger
