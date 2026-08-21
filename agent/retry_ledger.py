"""R0-only, local SQLite retry-ledger pilot; deliberately has no production caller."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

FINGERPRINT_VERSION = "efp/v1"
AUTHORITY_MODES = frozenset(("R0", "R1"))
STRATEGY_MODES = frozenset((
    "same_strategy",
    "fresh_context",
    "different_strategy",
    "human_escalated",
))
CHECK_TYPES = frozenset(("test", "lint", "type", "policy", "build", "custom"))
RESULT_STATES = frozenset(("pass", "fail", "blocked", "skipped", "timeout"))
DECISIONS = frozenset(("stop", "retry", "escalate"))
ERROR_CLASSES = frozenset((
    "none",
    "validation_schema",
    "policy_safety",
    "auth_permission",
    "scope_or_authority",
    "rate_limit",
    "transient_network",
    "timeout",
    "duplicate_fingerprint",
))
STOP_REASONS = frozenset((
    "none",
    "terminal_pass",
    "validation_schema",
    "policy_safety",
    "auth_permission",
    "scope_or_authority",
    "rate_limit_retry_limit",
    "transient_network_retry_limit",
    "timeout_retry_limit",
    "duplicate_second",
    "duplicate_third_hard_stop",
    "iteration_cap",
    "wall_clock_cap",
    "input_tokens_cap",
    "output_tokens_cap",
))
REQUIRED_FIELDS = (
    "run_id",
    "session_id",
    "repo",
    "branch",
    "head_sha",
    "task_id",
    "objective_id",
    "authority_mode",
    "loop_iteration",
    "attempt_number",
    "strategy_mode",
    "check_name",
    "check_type",
    "result_state",
    "error_class",
    "error_fingerprint",
    "decision",
    "decision_reason",
    "tool_calls_count",
    "tokens_used_input",
    "tokens_used_output",
    "estimated_cost_usd",
    "duration_ms",
    "changed_paths",
    "stop_reason",
    "created_at",
)

_UUID = re.compile(r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", re.I)
_ISO_TIME = re.compile(
    r"\b\d{4}-\d\d-\d\d(?:[ T]\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:?\d\d)?)?\b",
    re.I,
)
_LINE = re.compile(r"(?::|\bline\s+)\d+\b", re.I)
_DURATION = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|msec(?:ond)?s?|millisecond(?:s)?|sec(?:ond)?s?|mins?|minutes?|hours?|hrs?)\b",
    re.I,
)
_REQUEST_ID = re.compile(
    r"\b(?:request|req|trace|correlation)[_-]?id\s*[=:]\s*[^\s,;]+", re.I
)
_UNIX_ABS_PATH = re.compile(r"(?<![\w.])/(?:[^\s'\"<>|]+/)*[^\s'\"<>|:]+")
_WINDOWS_ABS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\s'\"<>|]+\\)*[^\s'\"<>|]+")
_SPACE = re.compile(r"\s+")


class RetryLedgerValidationError(ValueError):
    """A supplied event is outside the deliberately closed R0 pilot contract."""


@dataclass(frozen=True)
class BudgetConfig:
    max_iterations: int = 3
    wall_clock_minutes: int = 20
    max_input_tokens: int = 75_000
    max_output_tokens: int = 12_000


@dataclass(frozen=True)
class RetryDecision:
    decision: str
    decision_reason: str
    stop_reason: str


def _safe_shape(value: Any) -> Any:
    """Return a deterministic non-secret diagnostic shape suitable only for hashing."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value).lower()
    if isinstance(value, Mapping):
        return {
            str(key).lower(): _safe_shape(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_shape(item) for item in value]
    text = str(value).lower().strip()
    for pattern, replacement in (
        (_REQUEST_ID, "request_id=<id>"),
        (_UUID, "<uuid>"),
        (_ISO_TIME, "<timestamp>"),
        (_LINE, ":<line>"),
        (_DURATION, "<duration>"),
        (_WINDOWS_ABS_PATH, "<path>"),
        (_UNIX_ABS_PATH, "<path>"),
    ):
        text = pattern.sub(replacement, text)
    return _SPACE.sub(" ", text)


def normalize_fp_v1(error: Mapping[str, Any]) -> str:
    """Hash a normalized safe diagnostic shape, never persist or return raw diagnostics."""
    if not isinstance(error, Mapping):
        raise RetryLedgerValidationError("error shape must be a mapping")
    safe = {
        str(key).lower(): _safe_shape(value)
        for key, value in sorted(error.items(), key=lambda pair: str(pair[0]))
        if value is not None
    }
    blob = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{FINGERPRINT_VERSION}:{digest}"


def decide_r0_retry(
    ledger_state: Mapping[str, Any], budget: BudgetConfig, error_class: str
) -> RetryDecision:
    """Pure policy evaluated before another attempt; it does not inspect a database."""
    if error_class not in ERROR_CLASSES - {"none"}:
        raise RetryLedgerValidationError("error_class is not a closed retry enum")
    if int(ledger_state.get("loop_iteration", 0)) >= budget.max_iterations:
        return RetryDecision("stop", "budget_cap", "iteration_cap")
    if int(ledger_state.get("elapsed_ms", 0)) >= budget.wall_clock_minutes * 60_000:
        return RetryDecision("stop", "budget_cap", "wall_clock_cap")
    if int(ledger_state.get("input_tokens", 0)) >= budget.max_input_tokens:
        return RetryDecision("stop", "budget_cap", "input_tokens_cap")
    if int(ledger_state.get("output_tokens", 0)) >= budget.max_output_tokens:
        return RetryDecision("stop", "budget_cap", "output_tokens_cap")
    if error_class in {
        "validation_schema",
        "policy_safety",
        "auth_permission",
        "scope_or_authority",
    }:
        return RetryDecision("stop", error_class, error_class)
    seen = int(ledger_state.get("fingerprint_count", 0))
    if seen >= 3:
        return RetryDecision(
            "stop", "duplicate_fingerprint", "duplicate_third_hard_stop"
        )
    if seen >= 2:
        return RetryDecision("stop", "duplicate_fingerprint", "duplicate_second")
    attempts = int(ledger_state.get("attempts_for_check", 0))
    limit = {"rate_limit": 2, "transient_network": 2, "timeout": 1}.get(error_class)
    if limit is None:
        raise RetryLedgerValidationError(
            "duplicate_fingerprint requires fingerprint_count"
        )
    if attempts >= limit:
        return RetryDecision(
            "stop", f"{error_class}_retry_limit", f"{error_class}_retry_limit"
        )
    return RetryDecision("retry", error_class, "none")


def validate_retry_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one write and turn only the allowed paths list into canonical JSON."""
    unknown = set(event) - set(REQUIRED_FIELDS)
    if unknown:
        raise RetryLedgerValidationError("raw or unknown event fields are forbidden")
    missing = [name for name in REQUIRED_FIELDS if name not in event]
    if missing:
        raise RetryLedgerValidationError(
            "missing retry-ledger fields: " + ", ".join(missing)
        )
    row = dict(event)
    enums = {
        "authority_mode": AUTHORITY_MODES,
        "strategy_mode": STRATEGY_MODES,
        "check_type": CHECK_TYPES,
        "result_state": RESULT_STATES,
        "error_class": ERROR_CLASSES,
        "decision": DECISIONS,
        "stop_reason": STOP_REASONS,
    }
    for name, allowed in enums.items():
        if row[name] not in allowed:
            raise RetryLedgerValidationError(f"{name} is not a closed enum")
    if row["authority_mode"] != "R0":
        raise RetryLedgerValidationError("R0-only writer rejects non-R0 authority")
    if not isinstance(row["error_fingerprint"], str) or not re.fullmatch(
        r"efp/v1:[0-9a-f]{64}", row["error_fingerprint"]
    ):
        raise RetryLedgerValidationError(
            "error_fingerprint must be an efp/v1 SHA-256 digest"
        )
    for name in ("loop_iteration", "attempt_number", "tool_calls_count", "duration_ms"):
        if (
            not isinstance(row[name], int)
            or isinstance(row[name], bool)
            or row[name] < (1 if name == "attempt_number" else 0)
        ):
            raise RetryLedgerValidationError(
                f"{name} must be a valid non-negative integer"
            )
    for name in ("tokens_used_input", "tokens_used_output"):
        if row[name] is not None and (
            not isinstance(row[name], int)
            or isinstance(row[name], bool)
            or row[name] < 0
        ):
            raise RetryLedgerValidationError(
                f"{name} must be null or a non-negative integer"
            )
    if row["estimated_cost_usd"] is not None and (
        not isinstance(row["estimated_cost_usd"], (int, float))
        or isinstance(row["estimated_cost_usd"], bool)
        or row["estimated_cost_usd"] < 0
    ):
        raise RetryLedgerValidationError(
            "estimated_cost_usd must be null or non-negative"
        )
    if not isinstance(row["changed_paths"], list) or any(
        not isinstance(path, str) or not path for path in row["changed_paths"]
    ):
        raise RetryLedgerValidationError(
            "changed_paths must be a list of non-empty paths"
        )
    if row["changed_paths"]:
        raise RetryLedgerValidationError("R0 changed_paths must be empty")
    row["changed_paths"] = json.dumps(
        sorted(row["changed_paths"]), separators=(",", ":")
    )
    if row["decision"] == "retry" and row["stop_reason"] != "none":
        raise RetryLedgerValidationError("retry decision cannot have a stop reason")
    if row["decision"] == "escalate" and row["stop_reason"] != "none":
        raise RetryLedgerValidationError("escalate decision cannot have a stop reason")
    if row["decision"] == "stop" and row["stop_reason"] == "none":
        raise RetryLedgerValidationError("stop decision requires a stop reason")
    return row


class RetryLedgerWriter:
    def __init__(self, session_db_or_conn: Any) -> None:
        self._conn = (
            session_db_or_conn
            if isinstance(session_db_or_conn, sqlite3.Connection)
            else getattr(session_db_or_conn, "_conn", None)
        )
        if not isinstance(self._conn, sqlite3.Connection):
            raise TypeError(
                "RetryLedgerWriter requires an open SQLite connection or SessionDB"
            )

    def append(self, event: Mapping[str, Any]) -> int:
        row = validate_retry_event(event)
        prior = self._conn.execute(
            "SELECT MAX(loop_iteration) FROM retry_ledger_events WHERE run_id = ?",
            (row["run_id"],),
        ).fetchone()[0]
        if prior is not None and row["loop_iteration"] < prior:
            raise RetryLedgerValidationError(
                "loop_iteration must be monotonic within a run"
            )
        expected = self._conn.execute(
            "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM retry_ledger_events WHERE run_id = ? AND check_name = ?",
            (row["run_id"], row["check_name"]),
        ).fetchone()[0]
        if row["attempt_number"] != expected:
            raise RetryLedgerValidationError(
                "attempt_number must be contiguous by run/check"
            )
        cursor = self._conn.execute(
            f"INSERT INTO retry_ledger_events ({', '.join(REQUIRED_FIELDS)}) VALUES ({', '.join('?' for _ in REQUIRED_FIELDS)})",
            tuple(row[name] for name in REQUIRED_FIELDS),
        )
        self._conn.commit()
        return int(cursor.lastrowid)


def _event(
    run_id: str,
    check_name: str,
    attempt: int,
    iteration: int,
    error_class: str,
    fingerprint: str,
    decision: RetryDecision,
    *,
    result: str = "fail",
    input_tokens: int | None = 100,
    output_tokens: int | None = 25,
    duration_ms: int = 1000,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "session_id": "synthetic-session",
        "repo": "synthetic/local",
        "branch": "pilot",
        "head_sha": "0" * 40,
        "task_id": "synthetic-task",
        "objective_id": "synthetic-objective",
        "authority_mode": "R0",
        "loop_iteration": iteration,
        "attempt_number": attempt,
        "strategy_mode": "same_strategy",
        "check_name": check_name,
        "check_type": "test",
        "result_state": result,
        "error_class": error_class,
        "error_fingerprint": fingerprint,
        "decision": decision.decision,
        "decision_reason": decision.decision_reason,
        "tool_calls_count": 0,
        "tokens_used_input": input_tokens,
        "tokens_used_output": output_tokens,
        "estimated_cost_usd": None,
        "duration_ms": duration_ms,
        "changed_paths": [],
        "stop_reason": decision.stop_reason,
        "created_at": float(attempt + iteration * 10),
    }


def _report_from_rows(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in conn.execute("SELECT * FROM retry_ledger_events ORDER BY id")
    ]
    scenarios: dict[str, dict[str, Any]] = {}
    for row in rows:
        paths = json.loads(row["changed_paths"])
        if not isinstance(paths, list):
            raise RetryLedgerValidationError(
                "stored changed_paths did not deserialize to a list"
            )
        item = scenarios.setdefault(
            row["check_name"],
            {
                "attempt_count": 0,
                "consumed_input": 0,
                "consumed_output": 0,
                "consumed_duration_ms": 0,
                "budget_state": asdict(BudgetConfig()),
                "decision_path": [],
                "terminal_stop_reason": "none",
            },
        )
        item["attempt_count"] += 1
        item["consumed_input"] += row["tokens_used_input"] or 0
        item["consumed_output"] += row["tokens_used_output"] or 0
        item["consumed_duration_ms"] += row["duration_ms"]
        item["decision_path"].append({
            "decision": row["decision"],
            "decision_reason": row["decision_reason"],
            "stop_reason": row["stop_reason"],
        })
        if row["stop_reason"] != "none":
            item["terminal_stop_reason"] = row["stop_reason"]
    return {"synthetic_only": True, "attempt_count": len(rows), "scenarios": scenarios}


def run_synthetic_pilot() -> dict[str, Any]:
    """Run local-memory E2E scenarios through SCHEMA_SQL and the real writer only."""
    from hermes_state_common import SCHEMA_SQL

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('synthetic-session', 'synthetic', 0)"
    )
    writer = RetryLedgerWriter(conn)
    rate_fp = normalize_fp_v1({"kind": "rate", "request_id": "a", "path": "/tmp/a"})
    rate_first = decide_r0_retry(
        {"attempts_for_check": 0}, BudgetConfig(), "rate_limit"
    )
    writer.append(
        _event(
            "rate-run",
            "rate_limit_retry_to_pass",
            1,
            0,
            "rate_limit",
            rate_fp,
            rate_first,
        )
    )
    writer.append(
        _event(
            "rate-run",
            "rate_limit_retry_to_pass",
            2,
            1,
            "none",
            rate_fp,
            RetryDecision("stop", "terminal_pass", "terminal_pass"),
            result="pass",
            input_tokens=None,
            output_tokens=None,
            duration_ms=250,
        )
    )
    validation = decide_r0_retry({}, BudgetConfig(), "validation_schema")
    writer.append(
        _event(
            "validation-run",
            "validation_stop",
            1,
            0,
            "validation_schema",
            normalize_fp_v1({"kind": "validation"}),
            validation,
            result="blocked",
        )
    )
    dup_fp = normalize_fp_v1({"kind": "duplicate", "line": "x.py:1"})
    writer.append(
        _event(
            "duplicate-run",
            "duplicate_stop",
            1,
            0,
            "duplicate_fingerprint",
            dup_fp,
            RetryDecision("retry", "rate_limit", "none"),
        )
    )
    duplicate = decide_r0_retry(
        {"fingerprint_count": 2}, BudgetConfig(), "duplicate_fingerprint"
    )
    writer.append(
        _event(
            "duplicate-run",
            "duplicate_stop",
            2,
            1,
            "duplicate_fingerprint",
            dup_fp,
            duplicate,
        )
    )
    budget = decide_r0_retry(
        {"input_tokens": 75_000}, BudgetConfig(), "transient_network"
    )
    writer.append(
        _event(
            "budget-run",
            "input_budget_stop",
            1,
            0,
            "transient_network",
            normalize_fp_v1({"kind": "budget"}),
            budget,
            result="blocked",
            input_tokens=75_000,
        )
    )
    report = _report_from_rows(conn)
    verification = verify_retry_ledger_code(conn, report)
    return {"report": report, "verifier": verification}


def _schema_enum_values(schema: str, field: str) -> frozenset[str] | None:
    match = re.search(
        rf"{re.escape(field)}\s+TEXT\s+NOT\s+NULL\s+CHECK\s*\(\s*{re.escape(field)}\s+IN\s*\(([^)]*)\)\s*\)",
        schema,
        re.I | re.S,
    )
    if not match:
        return None
    return frozenset(re.findall(r"'([^']+)'", match.group(1)))


def verify_retry_ledger_code(
    conn: sqlite3.Connection, report: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Verify the concrete pilot DB, including behavior that DDL text alone cannot prove."""
    table_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'retry_ledger_events'"
    ).fetchone()
    schema = table_sql[0] if table_sql else ""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(retry_ledger_events)")}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(retry_ledger_events)")}
    expected_enums = {
        "authority_mode": AUTHORITY_MODES,
        "strategy_mode": STRATEGY_MODES,
        "check_type": CHECK_TYPES,
        "result_state": RESULT_STATES,
        "decision": DECISIONS,
        "error_class": ERROR_CLASSES,
        "stop_reason": STOP_REASONS,
    }
    trigger_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'retry_ledger_events_no_%'"
        )
    }
    checks = {
        "schema_exact_enums": all(
            _schema_enum_values(schema, field) == expected
            for field, expected in expected_enums.items()
        ),
        "required_columns": set(REQUIRED_FIELDS) <= columns,
        "nullable_usage_fields": all(
            not row[3]
            for row in conn.execute("PRAGMA table_info(retry_ledger_events)")
            if row[1]
            in {"tokens_used_input", "tokens_used_output", "estimated_cost_usd"}
        ),
        "run_check_index": "idx_retry_ledger_events_run_check" in indexes,
        "append_only_triggers": trigger_names
        == {"retry_ledger_events_no_update", "retry_ledger_events_no_delete"},
        "foreign_key_check": not conn.execute("PRAGMA foreign_key_check").fetchall(),
        "invalid_fk_rejected": False,
        "invalid_enum_rejected": False,
        "append_only": False,
        "sequence_invariants": True,
        "record_completeness": True,
        "decision_report_contract": report is not None,
    }
    row = conn.execute(
        "SELECT * FROM retry_ledger_events ORDER BY id LIMIT 1"
    ).fetchone()
    if row is not None:
        values = dict(row)
        try:
            conn.execute("SAVEPOINT retry_ledger_verify")
            bad_fk = dict(values)
            bad_fk["session_id"] = "missing-session"
            bad_fk["attempt_number"] = 99
            conn.execute(
                f"INSERT INTO retry_ledger_events ({', '.join(REQUIRED_FIELDS)}) VALUES ({', '.join('?' for _ in REQUIRED_FIELDS)})",
                tuple(bad_fk[name] for name in REQUIRED_FIELDS),
            )
        except sqlite3.IntegrityError:
            checks["invalid_fk_rejected"] = True
        finally:
            conn.execute("ROLLBACK TO retry_ledger_verify")
            conn.execute("RELEASE retry_ledger_verify")
        try:
            conn.execute("SAVEPOINT retry_ledger_enum")
            bad_enum = dict(values)
            bad_enum["run_id"] = "invalid-enum-run"
            bad_enum["check_name"] = "invalid-enum-check"
            bad_enum["decision"] = "invalid"
            conn.execute(
                f"INSERT INTO retry_ledger_events ({', '.join(REQUIRED_FIELDS)}) VALUES ({', '.join('?' for _ in REQUIRED_FIELDS)})",
                tuple(bad_enum[name] for name in REQUIRED_FIELDS),
            )
        except sqlite3.IntegrityError:
            checks["invalid_enum_rejected"] = True
        finally:
            conn.execute("ROLLBACK TO retry_ledger_enum")
            conn.execute("RELEASE retry_ledger_enum")
        try:
            conn.execute(
                "UPDATE retry_ledger_events SET decision = 'stop' WHERE id = ?",
                (values["id"],),
            )
        except sqlite3.IntegrityError:
            try:
                conn.execute(
                    "DELETE FROM retry_ledger_events WHERE id = ?", (values["id"],)
                )
            except sqlite3.IntegrityError:
                checks["append_only"] = True
        for stored in conn.execute("SELECT * FROM retry_ledger_events ORDER BY id"):
            event = dict(stored)
            event.pop("id", None)
            paths = json.loads(event["changed_paths"])
            if not isinstance(paths, list):
                checks["record_completeness"] = False
            event["changed_paths"] = paths
            try:
                validate_retry_event(event)
            except RetryLedgerValidationError:
                checks["record_completeness"] = False
        prior_iterations: dict[str, int] = {}
        for event_row in conn.execute(
            "SELECT run_id, loop_iteration FROM retry_ledger_events ORDER BY id"
        ):
            run_id, iteration = event_row
            if run_id in prior_iterations and iteration < prior_iterations[run_id]:
                checks["sequence_invariants"] = False
            prior_iterations[run_id] = iteration
        for run_id, check_name, attempt, iteration in conn.execute(
            "SELECT run_id, check_name, attempt_number, loop_iteration FROM retry_ledger_events ORDER BY run_id, check_name, attempt_number"
        ):
            if (
                attempt
                != conn.execute(
                    "SELECT COUNT(*) FROM retry_ledger_events WHERE run_id = ? AND check_name = ? AND attempt_number <= ?",
                    (run_id, check_name, attempt),
                ).fetchone()[0]
            ):
                checks["sequence_invariants"] = False
    if report is not None:
        needed = {
            "attempt_count",
            "consumed_input",
            "consumed_output",
            "consumed_duration_ms",
            "budget_state",
            "decision_path",
            "terminal_stop_reason",
        }
        checks["decision_report_contract"] = (
            report.get("synthetic_only") is True
            and bool(report.get("scenarios"))
            and all(needed <= set(item) for item in report["scenarios"].values())
        )
    return {"ok": all(checks.values()), "checks": checks}


# Compatibility name for an earlier isolated test surface; it now exercises real persistence.
def synthetic_r0_harness() -> dict[str, Any]:
    return run_synthetic_pilot()["report"]
