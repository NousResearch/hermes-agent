#!/usr/bin/env python3
"""
AGATHOS — Agent Guardian & Health Oversight System
The Vigilant Guardian. Monitors all agent sessions, detects entropy,
takes corrective actions. Background daemon on Mac Mini via launchd.
"""

import os
import sys
import json
import sqlite3
import time
import subprocess
import signal
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .wal_monitor import ToolCallMonitor
from . import entropy as _entropy
from . import directives as _directives
from . import actions as _actions
from . import notifications as _notifications
from . import audit as _audit
from . import resources as _resources
from . import cleanup as _cleanup
from . import drift as _drift
from . import provider_health as _provider_health
from . import cost_monitor as _cost_monitor
from . import circuit_breaker as _circuit_breaker
from .subprocess_utils import safe_subprocess

# === PATH RESOLUTION ===
# Add hermes-agent to path for module imports
_HERMES_AGENT = os.path.expanduser("~/.hermes/hermes-agent")
if os.path.isdir(_HERMES_AGENT) and _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)

try:
    from hermes_constants import get_hermes_home

    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

_AGATHOS_HOME = Path(os.path.expanduser("~/hermes"))


def _hermes_path(*parts: str) -> Path:
    """Build a path under HERMES_HOME (~/.hermes)."""
    return _HERMES_HOME.joinpath(*parts)


def _agathos_path(*parts: str) -> Path:
    """Build a path under ARGUS_HOME (~/hermes)."""
    return _AGATHOS_HOME.joinpath(*parts)


# === INTERNALS ===
# Direct module imports when available, subprocess fallback otherwise.
_HERMES_INTERNALS_AVAILABLE = False
try:
    from cron.jobs import (
        list_jobs,
    )
    from hermes_state import SessionDB, DEFAULT_DB_PATH
    from hermes_cli.config import load_config as _hermes_load_config

    _HERMES_INTERNALS_AVAILABLE = True
except (ImportError, TypeError) as _e:
    import logging as _logging

    _logging.getLogger("agathos").warning(
        "Internals unavailable (%s), using subprocess fallback", _e
    )
    from .hermes_fallback import (
        list_jobs,
        SessionDB,
        DEFAULT_DB_PATH,
        _hermes_load_config,
    )


# Corrective prompt templates by entropy type
CORRECTIVE_PROMPTS = {
    "repeat_tool_calls": (
        "ENTROPY CORRECTION: ARGUS detected repeated tool calls without progress. "
        "You are calling the same tool with the same arguments multiple times. "
        "Stop and reassess. Read the file/content you need ONCE, then act on it. "
        "Do not re-read files you already have in context. Complete the task."
    ),
    "repeat_commands": (
        "ENTROPY CORRECTION: ARGUS detected repeated terminal commands. "
        "You are running the same command multiple times. "
        "Check the output you already received before re-running. "
        "If the command failed, fix the issue, don't retry blindly."
    ),
    "stuck_loop": (
        "ENTROPY CORRECTION: ARGUS detected a stuck loop pattern. "
        "Your last several tool calls form a repeating cycle. "
        "STOP. Read your conversation history. Identify what you're trying to accomplish. "
        "Take a different approach. Do not repeat the same sequence."
    ),
    "no_file_changes": (
        "ENTROPY CORRECTION: ARGUS detected write operations that didn't change files. "
        "You are calling write_file/patch but the file content is not changing. "
        "Read the file first, verify what you're writing is actually different. "
        "If using patch, check that old_string matches exactly."
    ),
    "error_cascade": (
        "ENTROPY CORRECTION: ARGUS detected a cascade of tool failures. "
        "Multiple consecutive tool calls have returned errors. "
        "STOP. Read the error messages carefully. The environment or arguments may be wrong. "
        "Check file paths, command syntax, and prerequisites before retrying. "
        "If a tool keeps failing, try a different approach or use a different tool."
    ),
    "budget_pressure": (
        "BUDGET CORRECTION: You are burning through your iteration budget fast "
        "without productive output. "
        "Step back. Summarize what you have accomplished so far and what remains. "
        "Pick the simplest remaining task and complete it in one pass. "
        "Avoid exploratory tool calls — read once, then act."
    ),
    "quality_gate": (
        "QUALITY CORRECTION: Your output quality is below the 0.92 threshold. "
        "Provide mechanistic explanations, not surface descriptions. "
        "Include structured output with headers and metrics. "
        "Feed the pipeline: write facts, generate trajectories, enrich KB."
    ),
    "pipeline_violation": (
        "PIPELINE CORRECTION: You are not hitting all 4 pipeline targets. "
        "Every substantive interaction must produce: "
        "(1) target output, (2) holographic_memory.db facts, "
        "(3) trajectories (Q&A chains), (4) KB enrichment. "
        "Self-assess before finishing."
    ),
}

# === CONFIGURATION ===
# All modules are optional and can be toggled via config.yaml
# Most default to True (enabled), ML features default to False
_DEFAULT_AGATHOS_CONFIG = {
    # Core paths
    "db_path": str(_agathos_path("data", "watcher", "agathos.db")),
    "log_dir": str(_agathos_path("logs", "agathos")),
    
    # Timing
    "poll_interval": 30,
    "session_timeout_minutes": 60,
    
    # Thresholds
    "entropy_threshold": 3,
    "quality_threshold": 0.92,
    "max_restart_count": 3,
    
    # Module toggles (all optional, most default ON)
    "entropy_detection_enabled": True,      # detect_repeat_tool_calls, stuck_loops, etc.
    "actions_enabled": True,                # restart, kill, inject prompts
    "notifications_enabled": True,        # Telegram, Discord, Slack, etc.
    "metrics_enabled": True,              # Prometheus metrics export
    "wal_monitor_enabled": True,          # Real-time WAL monitoring
    "provider_health_enabled": True,      # Provider health tracking
    "prime_directives_enabled": True,     # Prime directive checking
    "cleanup_enabled": True,              # Orphaned session cleanup
    "drift_detection_enabled": True,        # Quality drift detection
    "resource_checks_enabled": True,      # Resource exhaustion monitoring
    "audit_trail_enabled": True,          # Audit logging
    "cost_monitoring_enabled": True,      # Budget and cost alerts
    
    # ML data export (optional feature, default OFF)
    "ml_data_enabled": False,             # Export trajectories to ~/.hermes/agathos/ml_data/
    "ml_memory_enabled": False,           # Record to holographic memory
    
    # Cost monitoring configuration (disabled by default, user enables if desired)
    "cost_monitoring": {
        "enabled": False,                   # Disabled by default - user enables in config.yaml
        "daily_budget": 20.00,              # USD - default $20 if enabled
        "alert_at_percent": 80,             # Alert at 80% of budget
        "expensive_session_threshold": 2.00,  # Alert on single session >$2
        "per_provider_limits": {},          # Auto-populated from discover_providers()
    },
    
    # Circuit breaker for provider failover (disabled by default)
    "circuit_breaker": {
        "enabled": False,                  # Disabled by default - user enables in config.yaml
        "failure_threshold": 3,            # Consecutive failures to open circuit
        "error_rate_threshold": 0.50,    # 50% error rate over window
        "min_requests_for_rate": 5,      # Min requests before rate check
        "recovery_timeout": 300,           # Seconds before retry (half-open)
        "half_open_requests": 2,         # Successful requests to close
        "notify_on_open": True,            # Alert when circuit opens
        "notify_on_close": True,           # Alert when circuit closes
    },
}


def _load_agathos_config() -> Dict:
    """Load AGATHOS config from hermes config.yaml 'agathos' key, falling back to defaults."""
    try:
        hermes_config = _hermes_load_config()
        agathos_overrides = hermes_config.get("agathos", {})
        return {**_DEFAULT_AGATHOS_CONFIG, **agathos_overrides}
    except Exception:
        return dict(_DEFAULT_AGATHOS_CONFIG)


# Runtime config — loaded once at import
CONFIG = _load_agathos_config()


# === DAEMON MANAGEMENT ===
from .daemon_mgmt import (
    _get_agathos_pid_path,
    write_agathos_pid_file,
    remove_agathos_pid_file,
    get_agathos_running_pid,
    is_agathos_running,
    get_agathos_launchd_label,
    get_agathos_launchd_plist_path,
    generate_agathos_launchd_plist,
    agathos_launchd_install,
    agathos_launchd_uninstall,
    agathos_launchd_status,
)

# === LOGGING ===
os.makedirs(CONFIG["log_dir"], exist_ok=True)

_LOG_MAX_BYTES = CONFIG.get("log_max_size_mb", 5) * 1024 * 1024  # 5MB default
_LOG_BACKUP_COUNT = CONFIG.get("log_backup_count", 3)

logger = logging.getLogger("agathos")
logger.setLevel(logging.INFO)


_rotating_handler = logging.handlers.RotatingFileHandler(
    str(Path(CONFIG["log_dir"]) / "agathos.log"),
    maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT,
)
_rotating_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(_rotating_handler)


_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(_stream_handler)


class Agathos:
    def __init__(self):
        self.db_path = CONFIG["db_path"]
        self.conn = None
        self.cursor = None
        self.running = False

        # WAL monitor
        self.wal_monitor = ToolCallMonitor(
            poll_interval=CONFIG.get("poll_interval", 30) / 10,
            repeat_threshold=CONFIG.get("entropy_threshold", 3),
        )

        # Ensure database directory exists
        os.makedirs(Path(self.db_path).parent, exist_ok=True)

        # Initialize database
        self._init_database()

        # Load schema
        self._load_schema()

        # Load directive checks from directives.yaml
        self._directives = _directives.load_directives()
        logger.info(
            "Loaded %d directive checks", len(self._directives.get("checks", []))
        )

        # Initialize audit trail table
        _audit.ensure_table(self.cursor, self.conn)
        _provider_health.ensure_table(self.cursor, self.conn)

        # Initialize drift detection table and detector
        _drift.ensure_table(self.cursor, self.conn)
        self._drift_detector = _drift.DriftDetector()
        self._drift_detector.check()  # initialize baseline

        # Cycle counter for periodic checks
        self._cycle_count = 0

    def _init_database(self):
        """Initialize database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent readers + single writer
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.cursor = self.conn.cursor()
        logger.info("Database connected: %s", self.db_path)

    def _load_schema(self):
        """Load database schema if tables don't exist."""
        schema_path = Path(__file__).parent / "watcher_schema.sql"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema = f.read()
            self.conn.executescript(schema)
            self.conn.commit()
            logger.info("Database schema loaded")

        # Migrate existing tables — add columns that may not exist
        self._migrate_schema()

    def _migrate_schema(self):
        """Add columns to existing tables that were added after initial schema."""
        migrations = [
            ("tool_calls", "success", "BOOLEAN"),
            ("tool_calls", "error_message", "TEXT"),
        ]

        for table, column, col_type in migrations:
            try:
                self.cursor.execute("PRAGMA table_info(%s)" % table)
                existing = {row[1] for row in self.cursor.fetchall()}
                if column not in existing:
                    self.cursor.execute(
                        "ALTER TABLE %s ADD COLUMN %s %s" % (table, column, col_type)
                    )
                    logger.info("Migrated: added %s.%s %s", table, column, col_type)
            except sqlite3.Error:
                pass

        self.conn.commit()

    def _discover_by_source(self, source_type: str) -> List[Dict]:
        """Discover sessions from a specific source (cron, delegate, manual).

        Uses Hermes internals: list_jobs() for cron, SessionDB for delegate/manual.
        """
        sessions = []

        try:
            if source_type == "cron":
                jobs = list_jobs(include_disabled=False)
                for job in jobs:
                    sessions.append({
                        "session_id": f"cron_{job['id']}",
                        "session_type": "cron",
                        "job_id": job["id"],
                        "task_description": job.get("name", "Unknown"),
                        "model": job.get("model"),
                        "provider": job.get("provider"),
                        "metadata": json.dumps(job),
                    })

            elif source_type in ("delegate", "manual"):
                db = SessionDB(DEFAULT_DB_PATH)
                try:
                    all_sessions = db.list_sessions_rich(limit=50)
                finally:
                    db.close()

                for s in all_sessions:
                    src = s.get("source", "")
                    if source_type == "delegate" and src in ("delegate", "delegate_task"):
                        sessions.append({
                            "session_id": f"delegate_{s['id']}",
                            "session_type": "delegate_task",
                            "task_description": s.get("title", f"Delegate {s['id'][:12]}"),
                            "metadata": json.dumps({
                                "session_id": s["id"],
                                "source": src,
                                "started_at": s.get("started_at"),
                            }),
                        })
                    elif source_type == "manual" and src in ("cli", "telegram", "manual", "gateway"):
                        sessions.append({
                            "session_id": f"manual_{s['id']}",
                            "session_type": "manual",
                            "task_description": s.get("title", f"Session {s['id'][:12]}"),
                            "metadata": json.dumps({
                                "session_id": s["id"],
                                "source": src,
                                "started_at": s.get("started_at"),
                            }),
                        })

        except Exception as e:
            logger.error("Error discovering %s sessions: %s", source_type, e, exc_info=True)

        return sessions

    def discover_sessions(self) -> List[Dict]:
        """Discover all active agent sessions from cron, delegate, and manual sources."""
        return (
            self._discover_by_source("cron") +
            self._discover_by_source("delegate") +
            self._discover_by_source("manual")
        )

    def register_session(self, session: Dict):
        """Register a session in the database."""
        try:
            self.cursor.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (session_id, session_type, job_id, task_description, model, provider, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session["session_id"],
                    session["session_type"],
                    session.get("job_id"),
                    session.get("task_description"),
                    session.get("model"),
                    session.get("provider"),
                    session.get("metadata"),
                ),
            )
            self.conn.commit()
            logger.debug("Registered session: %s", session["session_id"])

        except Exception as e:
            logger.error(
                "Error registering session %s: %s",
                session["session_id"],
                e,
                exc_info=True,
            )

    def collect_metrics(self, session_id: str) -> None:
        """Collect metrics for a session from logs and holographic_memory.db."""
        try:
            # 1. Populate tool_calls from session data (fills entropy detection table)
            self._populate_tool_calls_from_session(session_id)

            # 2. Count existing tool calls for this session
            self.cursor.execute(
                "SELECT COUNT(*) as cnt FROM tool_calls WHERE session_id = ?",
                (session_id,),
            )
            tool_count = self.cursor.fetchone()["cnt"]

            # 3. Count terminal commands for this session
            self.cursor.execute(
                "SELECT COUNT(*) as cnt FROM terminal_commands WHERE session_id = ?",
                (session_id,),
            )
            cmd_count = self.cursor.fetchone()["cnt"]

            # 4. Get quality metrics from holographic_memory.db
            quality_score = self._fetch_quality_from_holographic(session_id)

            # 5. Update session with collected metrics
            self.cursor.execute(
                """
                UPDATE sessions
                SET tool_call_count = ?, quality_gate_score = ?, last_activity_at = ?
                WHERE session_id = ?
            """,
                (
                    tool_count + cmd_count,
                    quality_score,
                    datetime.now().isoformat(),
                    session_id,
                ),
            )

            # 6. Record quality metric if available
            if quality_score is not None:
                self.cursor.execute(
                    """
                    INSERT INTO quality_metrics (session_id, metric_type, metric_value, details)
                    VALUES (?, 'overall_quality', ?, ?)
                """,
                    (
                        session_id,
                        quality_score,
                        json.dumps(
                            {"tool_calls": tool_count, "terminal_commands": cmd_count}
                        ),
                    ),
                )

            self.conn.commit()

        except Exception as e:
            logger.error(
                "Error collecting metrics for %s: %s", session_id, e, exc_info=True
            )

    def _populate_tool_calls_from_session(self, session_id: str) -> None:
        """Read tool calls from state.db and insert into agathos.db.tool_calls.

        Also detects tool errors using the same heuristic as
        agent.display._detect_tool_failure and populates success/error_message.
        """
        if not _HERMES_INTERNALS_AVAILABLE:
            return
        real_session_id = _actions.strip_session_prefix(session_id)

        try:
            db = SessionDB(DEFAULT_DB_PATH)
            try:
                messages = db.get_messages(real_session_id)
            finally:
                db.close()
        except Exception:
            return

        if not messages:
            return

        # Find the latest timestamp we've already ingested for this session
        self.cursor.execute(
            "SELECT MAX(timestamp) as max_ts FROM tool_calls WHERE session_id = ?",
            (session_id,),
        )
        row = self.cursor.fetchone()
        last_ingested = row["max_ts"] if row and row["max_ts"] else "0"

        try:
            last_ingested_float = float(last_ingested)
        except (ValueError, TypeError):
            last_ingested_float = 0.0

        # Match tool results to calls
        tool_results: Dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id:
                    tool_results[tc_id] = str(msg.get("content", ""))

        # Extract tool calls from assistant messages, detect errors
        inserts = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            ts = msg.get("timestamp", "")
            try:
                ts_float = float(ts) if ts else 0.0
            except (ValueError, TypeError):
                ts_float = 0.0

            if ts_float <= last_ingested_float:
                continue

            tool_calls_raw = msg.get("tool_calls")
            if not tool_calls_raw:
                continue

            try:
                if isinstance(tool_calls_raw, str):
                    tool_calls_raw = json.loads(tool_calls_raw)
                for tc in tool_calls_raw:
                    name = tc.get("function", {}).get("name") or tc.get("name", "")
                    args = tc.get("function", {}).get("arguments") or tc.get(
                        "arguments", ""
                    )
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    if not name:
                        continue

                    # Match tool result and detect errors
                    tc_id = tc.get("id", "")
                    result_content = tool_results.get(tc_id, "")
                    is_error, error_msg = _entropy.detect_tool_error(
                        name, result_content
                    )
                    file_changed = _entropy.detect_file_changed(
                        name, result_content, is_error
                    )

                    inserts.append(
                        (
                            session_id,
                            name,
                            str(args),
                            str(ts),
                            not is_error,  # success = not error
                            error_msg,
                            file_changed,
                        )
                    )
            except (json.JSONDecodeError, TypeError):
                continue

        # Batch insert with success/error_message/file_changed columns
        if inserts:
            self.cursor.executemany(
                """
                INSERT OR IGNORE INTO tool_calls
                (session_id, tool_name, tool_args, timestamp, success, error_message, file_changed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                inserts,
            )
            error_count = sum(1 for i in inserts if not i[4])
            logger.debug(
                "Ingested %d tool calls for session %s (%d errors)",
                len(inserts),
                session_id,
                error_count,
            )

    def _fetch_quality_from_holographic(self, session_id: str) -> Optional[float]:
        """Fetch quality score from holographic_memory.db for a session."""
        holo_db = str(_hermes_path("holographic_memory.db"))
        if not os.path.exists(holo_db):
            return None

        try:
            holo_conn = sqlite3.connect(holo_db)
            holo_conn.row_factory = sqlite3.Row
            cursor = holo_conn.cursor()

            # Get average quality score from recent facts
            cursor.execute("""
                SELECT AVG(quality_score) as avg_quality
                FROM facts
                WHERE timestamp > datetime('now', '-24 hours')
                AND quality_score IS NOT NULL
            """)
            row = cursor.fetchone()
            holo_conn.close()

            if row and row["avg_quality"]:
                return round(row["avg_quality"], 4)

        except Exception as e:
            logger.error("Error reading holographic_memory.db: %s", e, exc_info=True)

        return None

    def detect_entropy(self, session_id: str) -> List[Dict]:
        """Detect entropy patterns in a session."""
        threshold = CONFIG.get("entropy_threshold", 3)
        cursor = self.cursor
        detections = []
        detections.extend(
            _entropy.detect_repeat_tool_calls(cursor, session_id, threshold)
        )
        detections.extend(
            _entropy.detect_repeat_commands(cursor, session_id, threshold)
        )
        detections.extend(_entropy.detect_stuck_loops(cursor, session_id))
        detections.extend(_entropy.detect_no_file_changes(cursor, session_id))
        detections.extend(_entropy.detect_error_cascade(cursor, session_id))

        # Budget pressure needs session type lookup for max_budget
        cursor.execute(
            "SELECT session_type FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        is_delegate = row and row["session_type"] == "delegate_task"
        max_budget = CONFIG.get("max_iterations", 50 if is_delegate else 90)
        detections.extend(
            _entropy.detect_budget_pressure(
                cursor, session_id, max_budget, DEFAULT_DB_PATH
            )
        )
        return detections

    def check_prime_directive(self, session_id: str) -> List[Dict]:
        """Check if session is following prime directive."""
        # Open holographic DB if available
        holo_db = str(_hermes_path("holographic_memory.db"))
        holo_conn = None
        if os.path.exists(holo_db):
            try:
                holo_conn = sqlite3.connect(holo_db)
                holo_conn.row_factory = sqlite3.Row
            except sqlite3.Error:
                pass

        try:
            results = _directives.execute_checks(
                session_id, self.cursor, holo_conn, self._directives
            )
        finally:
            if holo_conn:
                holo_conn.close()

        return results

    def make_decision(
        self,
        session_id: str,
        entropy_detections: List[Dict],
        directive_checks: List[Dict],
    ) -> Optional[Dict]:
        """Make decision about what action to take."""

        # Get session info
        self.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        session = dict(row)

        # Check for critical entropy
        critical_entropy = [
            d for d in entropy_detections if d["severity"] == "critical"
        ]

        if critical_entropy:
            # Check if it's repeat tool calls (kill condition)
            repeat_tool_calls = [
                d for d in critical_entropy if d["entropy_type"] == "repeat_tool_calls"
            ]

            if repeat_tool_calls:
                # High entropy - kill condition
                if session["restart_count"] >= CONFIG["max_restart_count"]:
                    return {
                        "action": "kill",
                        "reason": f"High entropy: repeat tool calls detected ({len(repeat_tool_calls)} times), max restarts reached",
                    }
                else:
                    return {
                        "action": "restart",
                        "reason": f"High entropy: repeat tool calls detected ({len(repeat_tool_calls)} times)",
                    }

            # Other critical entropy - restart
            return {
                "action": "restart",
                "reason": f"Critical entropy detected: {critical_entropy[0]['entropy_type']}",
            }

        # Check for directive violations
        failed_checks = [c for c in directive_checks if not c["passed"]]

        if failed_checks:
            return {
                "action": "restart",
                "reason": f"Prime directive violation: {failed_checks[0]['check_type']}",
            }

        # Check for low quality
        self.cursor.execute(
            """
            SELECT AVG(metric_value) as avg_quality
            FROM quality_metrics
            WHERE session_id = ? AND timestamp > datetime('now', '-1 hour')
        """,
            (session_id,),
        )

        result = self.cursor.fetchone()
        if (
            result
            and result["avg_quality"]
            and result["avg_quality"] < CONFIG["quality_threshold"]
        ):
            return {
                "action": "restart",
                "reason": f"Low quality: {result['avg_quality']:.3f} < {CONFIG['quality_threshold']}",
            }

        # No action needed
        return None

    def _run_periodic_checks(self):
        """Run resource, drift, and cleanup checks every N cycles.

        Table-driven: (cycle_offset, config_key, module, check_func, format_func, audit_func, notify_type)
        """
        cycle = self._cycle_count
        mod = cycle % 10

        # Standard periodic checks
        checks = [
            # (offset, config_key, module, check_fn, format_fn, audit_fn, notify_type)
            (0, "resource_checks_enabled", _resources, "run_resource_check", "format_alert", "record_resource_alert", "resource_alert"),
            (3, "provider_health_enabled", _provider_health, "run_provider_check", "format_alert", "record_provider_alert", "provider_health"),
            (5, "cleanup_enabled", _cleanup, "run_cleanup", None, "record_cleanup_event", None),
            (7, "cost_monitoring_enabled", _cost_monitor, "check_costs", "format_cost_alert", "record_cost_alert", "cost_alert"),
        ]

        for offset, config_key, module, check_fn, format_fn, audit_fn, notify_type in checks:
            if mod != offset or not CONFIG.get(config_key, True):
                continue

            try:
                # Run check (some need cursor/conn, some just CONFIG)
                if check_fn == "check_costs":
                    report = getattr(module, check_fn)(CONFIG)
                    has_alert = report.get("has_alert")
                else:
                    report = getattr(module, check_fn)(self.cursor, self.conn)
                    has_alert = report.get("overall_severity") in ("warning", "critical")

                if not has_alert:
                    continue

                # Format alert if formatter exists
                alert = None
                if format_fn:
                    alert = getattr(module, format_fn)(report)
                elif check_fn == "run_cleanup":
                    total = sum(len(v) for v in report.values())
                    if total > 0:
                        logger.info("Cleanup: %d orphaned sessions found", total)
                        alert = f"Cleanup: {total} orphaned sessions found"
                    else:
                        continue

                if alert:
                    logger.warning("%s alert:\n%s", config_key.replace("_enabled", "").replace("_", " "), alert)

                # Audit
                if CONFIG.get("audit_trail_enabled", True) and audit_fn:
                    audit_method = getattr(_audit, audit_fn)
                    if check_fn == "run_resource_check":
                        audit_method(self.cursor, self.conn, resource_type="system", severity=report["overall_severity"], details=report)
                    elif check_fn == "run_provider_check":
                        audit_method(self.cursor, self.conn, providers=list(report.get("providers", {}).keys()), severity=report["overall_severity"], details=report)
                    else:
                        audit_method(self.cursor, self.conn, details=report)

                # Notify
                if notify_type and CONFIG.get("notifications_enabled", True):
                    _notifications.send_notification(self.cursor, self.conn, "system", notify_type, alert)

            except Exception as e:
                logger.error("%s check failed: %s", config_key.replace("_enabled", ""), e)

        # Config drift check (every cycle) - special handling, not periodic
        if CONFIG.get("drift_detection_enabled", True):
            try:
                changes = self._drift_detector.check()
                if changes:
                    self._drift_detector.record_changes(self.cursor, self.conn, changes)
                    if CONFIG.get("audit_trail_enabled", True):
                        for c in changes:
                            _audit.record_drift_event(self.cursor, self.conn, file_label=c["file"], change_type=c["change_type"], old_hash=c.get("old_hash"), new_hash=c.get("new_hash"))
                    logger.info("Drift: %s %s", c["file"], c["change_type"])
            except Exception as e:
                logger.error("Drift check failed: %s", e)

        # Circuit breaker check (mod 8, has nested config)
        if mod == 8:
            cb_config = CONFIG.get("circuit_breaker", {})
            if cb_config.get("enabled", False):
                try:
                    events = _circuit_breaker.check_circuits(self.cursor, self.conn, CONFIG)
                    for event in events:
                        if event.get("notify"):
                            msg = _circuit_breaker.format_circuit_event(event)
                            logger.warning("Circuit breaker: %s", msg)
                            if CONFIG.get("audit_trail_enabled", True):
                                _audit.record_circuit_event(self.cursor, self.conn, provider=event["provider"], transition=event["transition"], reason=event["reason"])
                            if CONFIG.get("notifications_enabled", True):
                                _notifications.send_notification(self.cursor, self.conn, "system", "circuit_breaker", msg)
                except Exception as e:
                    logger.error("Circuit breaker check failed: %s", e)

    def execute_action(self, session_id: str, decision: Dict):
        """Execute the decided action."""
        action = decision["action"]
        reason = decision["reason"]

        logger.info("Executing %s on %s: %s", action, session_id, reason)

        # Record action in database
        self.cursor.execute(
            """
            INSERT INTO watcher_actions (session_id, action_type, action_reason, details)
            VALUES (?, ?, ?, ?)
        """,
            (session_id, action, reason, json.dumps(decision)),
        )

        action_id = self.cursor.lastrowid

        try:
            if action == "restart":
                _actions.restart_session(self.cursor, self.conn, session_id, reason, CORRECTIVE_PROMPTS)
                if CONFIG.get("notifications_enabled", True):
                    _notifications.send_notification(self.cursor, self.conn, session_id, "restart", f"Restarted: {reason}")

            elif action == "kill":
                _actions.kill_session(self.cursor, self.conn, session_id, reason)
                if CONFIG.get("notifications_enabled", True):
                    _notifications.send_notification(self.cursor, self.conn, session_id, "kill", f"Killed: {reason}")

            elif action == "inject_prompt":
                _actions.inject_prompt(self.cursor, self.conn, session_id, decision.get("prompt", ""))

            # Mark action as successful
            self.cursor.execute(
                """
                UPDATE watcher_actions SET success = TRUE WHERE id = ?
            """,
                (action_id,),
            )

            # Audit trail - if enabled
            if CONFIG.get("audit_trail_enabled", True):
                _audit.record_decision(
                    self.cursor,
                    self.conn,
                    session_id=session_id,
                    action_type=action,
                    severity="critical" if action == "kill" else "warning",
                    decision_reason=reason,
                    action_result="success",
                    metadata={"action_id": action_id, **decision},
                )

            # Export ML training data (optional feature)
            if CONFIG.get("ml_data_enabled", False):
                try:
                    from . import ml_data

                    entropy_type = decision.get("entropy_type", "unknown")
                    severity = "critical" if action == "kill" else "warning"

                    ml_data.export_entropy_event(
                        entropy_type=entropy_type,
                        severity=severity,
                        session_context={
                            "session_id": session_id,
                            "task_description": reason,
                        },
                        detection_details=decision,
                        recovery_action=action,
                        outcome="success",
                        enable_trajectory=True,
                        enable_memory=CONFIG.get("ml_memory_enabled", True),
                    )
                except Exception as e:
                    logger.debug("ML data export failed (non-critical): %s", e)

        except Exception as e:
            logger.error(
                "Error executing %s on %s: %s", action, session_id, e, exc_info=True
            )

            # Mark action as failed
            self.cursor.execute(
                """
                UPDATE watcher_actions SET success = FALSE, details = ? WHERE id = ?
            """,
                (json.dumps({"error": str(e)}), action_id),
            )

            # Audit trail — failure - if enabled
            if CONFIG.get("audit_trail_enabled", True):
                _audit.record_decision(
                    self.cursor,
                    self.conn,
                    session_id=session_id,
                    action_type=action,
                    severity="critical",
                    decision_reason=reason,
                    action_result="failure",
                    metadata={"action_id": action_id, "error": str(e)},
                )

        self.conn.commit()

    def run(self):
        """Main watcher loop."""
        logger.info("Agent Watcher starting...")
        self.running = True

        # Start WAL monitor for real-time tool call detection
        # Start WAL monitor for real-time tool call detection - if enabled
        if CONFIG.get("wal_monitor_enabled", True):
            self.wal_monitor.start()

        write_agathos_pid_file()
        logger.info("ARGUS PID file written: %s", _get_agathos_pid_path())

        while self.running:
            try:
                # Process WAL monitor events (tool call entropy) - if enabled
                if CONFIG.get("wal_monitor_enabled", True):
                    self._process_wal_events()

                # Discover sessions
                sessions = self.discover_sessions()

                for session in sessions:
                    try:
                        sid = session["session_id"]
                        self.register_session(session)
                        
                        # Collect metrics - if enabled
                        if CONFIG.get("metrics_enabled", True):
                            self.collect_metrics(sid)
                        
                        # Detect entropy - if enabled
                        entropy_detections = []
                        if CONFIG.get("entropy_detection_enabled", True):
                            entropy_detections = self.detect_entropy(sid)
                        
                        # Check prime directives - if enabled
                        directive_checks = []
                        if CONFIG.get("prime_directives_enabled", True):
                            directive_checks = self.check_prime_directive(sid)
                        
                        # Make and execute decision - if actions enabled
                        if CONFIG.get("actions_enabled", True):
                            decision = self.make_decision(
                                sid, entropy_detections, directive_checks
                            )
                            if decision:
                                self.execute_action(sid, decision)
                    except Exception as e:
                        logger.error(
                            "Error processing session %s: %s",
                            session.get("session_id"),
                            e,
                        )

                # Sleep before next poll
                self._cycle_count += 1
                self._run_periodic_checks()
                time.sleep(CONFIG["poll_interval"])

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.running = False

            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(CONFIG["poll_interval"])

    def stop(self):
        """Stop the watcher."""
        logger.info("Stopping Agent Watcher...")
        self.running = False
        # Stop WAL monitor - only if it was enabled and started
        if CONFIG.get("wal_monitor_enabled", True):
            self.wal_monitor.stop()
        remove_agathos_pid_file()
        if self.conn:
            self.conn.close()

    def _process_wal_events(self):
        """Process events from the WAL monitor (real-time tool call entropy)."""
        events = self.wal_monitor.get_events(limit=50)

        for event in events:
            if event.event_type == "repeat_detected":
                # Record entropy detection
                self.cursor.execute(
                    """
                    INSERT INTO entropy_detections
                    (session_id, entropy_type, severity, details)
                    VALUES (?, 'repeat_tool_calls', 'warning', ?)
                """,
                    (
                        f"wal_{event.session_id}",
                        json.dumps(
                            {
                                "tool_name": event.tool_name,
                                "source": "wal_monitor",
                                **event.details,
                            }
                        ),
                    ),
                )

                # Insert into tool_calls for compatibility with existing detection
                self.cursor.execute(
                    """
                    INSERT INTO tool_calls
                    (session_id, tool_name, tool_args, timestamp, file_changed)
                    VALUES (?, ?, ?, ?, FALSE)
                """,
                    (
                        f"wal_{event.session_id}",
                        event.tool_name,
                        event.tool_args or "{}",
                        str(event.timestamp),
                    ),
                )

                logger.warning(
                    "WAL: repeat tool '%s' in session %s",
                    event.tool_name,
                    event.session_id[:15],
                )

            elif event.event_type == "stuck_loop_detected":
                self.cursor.execute(
                    """
                    INSERT INTO entropy_detections
                    (session_id, entropy_type, severity, details)
                    VALUES (?, 'stuck_loop', 'critical', ?)
                """,
                    (
                        f"wal_{event.session_id}",
                        json.dumps(
                            {
                                "source": "wal_monitor",
                                **event.details,
                            }
                        ),
                    ),
                )

                logger.warning(
                    "WAL: stuck loop in session %s: %s",
                    event.session_id[:15],
                    event.details.get("pattern"),
                )

        if events:
            self.conn.commit()


def main():
    """Main entry point."""
    daemon = Agathos()

    # Handle signals
    def _signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %s", signum)
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run Agathos
    try:
        daemon.run()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        daemon.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
