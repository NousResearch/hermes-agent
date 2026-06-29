"""SQLite-backed StoreCRM QA job and case store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .models import (
    CaseStatus,
    Evidence,
    FinalReportSummary,
    JobStatus,
    Lease,
    PolicyMetadata,
    QACase,
    QAJob,
    RunnerOutcome,
    RunnerResult,
    from_iso,
    to_iso,
    utc_now,
)
from .redaction import redact_text, redact_value, redacted_json

SCHEMA_VERSION = 1


def default_db_path() -> Path:
    return get_hermes_home() / "storecrm_qa" / "qa_jobs.sqlite3"


class StoreCRMQAStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    allowed_operations_json TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_payload_json TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    allowed_operations_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS case_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    attempt INTEGER NOT NULL,
                    lease_owner TEXT,
                    outcome TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cases_runnable
                    ON cases(status, lease_expires_at, id);
                CREATE INDEX IF NOT EXISTS idx_cases_job_status
                    ON cases(job_id, status);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, to_iso(utc_now())),
            )

    def enqueue_job(
        self,
        *,
        name: str,
        cases: Iterable[dict[str, Any]],
        metadata: PolicyMetadata,
        max_attempts: int = 2,
    ) -> QAJob:
        metadata.validate()
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        case_items = list(cases)
        if not case_items:
            raise ValueError("at least one case is required")
        now = to_iso(utc_now())
        allowed_json = json.dumps(list(metadata.allowed_operations), sort_keys=True)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                INSERT INTO jobs (
                    name, status, tenant_id, store_id, risk, allowed_operations_json,
                    max_attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    JobStatus.QUEUED.value,
                    metadata.tenant_id,
                    metadata.store_id,
                    metadata.risk,
                    allowed_json,
                    max_attempts,
                    now,
                    now,
                ),
            )
            job_id = int(cur.lastrowid)
            for index, item in enumerate(case_items, start=1):
                case_metadata = _metadata_from_case(item, metadata)
                case_metadata.validate()
                conn.execute(
                    """
                    INSERT INTO cases (
                        job_id, name, status, input_payload_json,
                        tenant_id, store_id, risk, allowed_operations_json,
                        attempts, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        job_id,
                        str(item.get("name") or f"case-{index}"),
                        CaseStatus.QUEUED.value,
                        redacted_json(item.get("input", {})),
                        case_metadata.tenant_id,
                        case_metadata.store_id,
                        case_metadata.risk,
                        json.dumps(list(case_metadata.allowed_operations), sort_keys=True),
                        max_attempts,
                        now,
                        now,
                    ),
                )
            conn.execute("COMMIT")
        return self.get_job(job_id)

    def list_jobs(self, status: JobStatus | str | None = None) -> list[QAJob]:
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (_status_value(status),)
        query += " ORDER BY id"
        with self._connect() as conn:
            return [_job_from_row(row) for row in conn.execute(query, params)]

    def list_cases(
        self,
        *,
        job_id: int | None = None,
        status: CaseStatus | str | None = None,
    ) -> list[QACase]:
        clauses = []
        params: list[Any] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(_status_value(status))
        query = "SELECT * FROM cases"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as conn:
            return [_case_from_row(row) for row in conn.execute(query, params)]

    def get_job(self, job_id: int) -> QAJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job {job_id} not found")
        return _job_from_row(row)

    def get_case(self, case_id: int) -> QACase:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"case {case_id} not found")
        return _case_from_row(row)

    def claim_next_case(
        self,
        *,
        owner: str,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> QACase | None:
        if not owner.strip():
            raise ValueError("lease owner is required")
        current = now or utc_now()
        current_iso = to_iso(current)
        expires_at = to_iso(current + timedelta(seconds=lease_seconds))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._recover_stale_locked(conn, current_iso)
            row = conn.execute(
                """
                SELECT * FROM cases
                WHERE status = ?
                  AND attempts < max_attempts
                ORDER BY id
                LIMIT 1
                """,
                (CaseStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            attempts = int(row["attempts"]) + 1
            conn.execute(
                """
                UPDATE cases
                SET status = ?, attempts = ?, lease_owner = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    CaseStatus.LEASED.value,
                    attempts,
                    owner,
                    expires_at,
                    current_iso,
                    row["id"],
                ),
            )
            self._refresh_job_status_locked(conn, int(row["job_id"]), current_iso)
            conn.execute("COMMIT")
        return self.get_case(int(row["id"]))

    def heartbeat(
        self,
        *,
        case_id: int,
        owner: str,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> Lease:
        current = now or utc_now()
        expires_at = current + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(f"case {case_id} not found")
            if row["status"] != CaseStatus.LEASED.value or row["lease_owner"] != owner:
                raise ValueError("case is not leased by this owner")
            if from_iso(row["lease_expires_at"]) <= current:
                raise ValueError("lease already expired")
            conn.execute(
                "UPDATE cases SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (to_iso(expires_at), to_iso(current), case_id),
            )
            conn.execute("COMMIT")
        return Lease(case_id, int(row["job_id"]), owner, expires_at, int(row["attempts"]))

    def complete_case(
        self,
        *,
        case_id: int,
        owner: str,
        result: RunnerResult,
        now: datetime | None = None,
    ) -> QACase:
        current = now or utc_now()
        status = {
            RunnerOutcome.PASS: CaseStatus.PASSED,
            RunnerOutcome.FAIL: CaseStatus.FAILED,
            RunnerOutcome.NEEDS_ENGINEER: CaseStatus.NEEDS_ENGINEER,
        }[result.outcome]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(f"case {case_id} not found")
            if row["status"] != CaseStatus.LEASED.value or row["lease_owner"] != owner:
                raise ValueError("case is not leased by this owner")
            if from_iso(row["lease_expires_at"]) <= current:
                raise ValueError("lease already expired")
            conn.execute(
                """
                UPDATE cases
                SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (status.value, to_iso(current), case_id),
            )
            self._insert_attempt_locked(conn, row, owner, result, current)
            self._refresh_job_status_locked(conn, int(row["job_id"]), to_iso(current))
            conn.execute("COMMIT")
        return self.get_case(case_id)

    def fail_for_retry(
        self,
        *,
        case_id: int,
        owner: str,
        summary: str,
        evidence: Iterable[Evidence] = (),
        now: datetime | None = None,
    ) -> QACase:
        current = now or utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(f"case {case_id} not found")
            if row["status"] != CaseStatus.LEASED.value or row["lease_owner"] != owner:
                raise ValueError("case is not leased by this owner")
            if from_iso(row["lease_expires_at"]) <= current:
                raise ValueError("lease already expired")
            next_status = (
                CaseStatus.QUEUED
                if int(row["attempts"]) < int(row["max_attempts"])
                else CaseStatus.EXHAUSTED
            )
            conn.execute(
                """
                UPDATE cases
                SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_status.value, to_iso(current), case_id),
            )
            result = RunnerResult(
                outcome=RunnerOutcome.FAIL,
                summary=summary,
                evidence=tuple(evidence),
            )
            self._insert_attempt_locked(conn, row, owner, result, current)
            self._refresh_job_status_locked(conn, int(row["job_id"]), to_iso(current))
            conn.execute("COMMIT")
        return self.get_case(case_id)

    def recover_stale_leases(self, *, now: datetime | None = None) -> int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            count = self._recover_stale_locked(conn, to_iso(now or utc_now()))
            conn.execute("COMMIT")
        return count

    def report_summary(self, job_id: int) -> FinalReportSummary:
        with self._connect() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"job {job_id} not found")
            counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM cases WHERE job_id = ? GROUP BY status",
                    (job_id,),
                )
            }
            evidence_count = conn.execute(
                "SELECT COUNT(*) AS count FROM case_attempts WHERE job_id = ?",
                (job_id,),
            ).fetchone()["count"]
        return FinalReportSummary(
            job_id=job_id,
            status=JobStatus(job["status"]),
            total_cases=sum(counts.values()),
            passed=counts.get(CaseStatus.PASSED.value, 0),
            failed=counts.get(CaseStatus.FAILED.value, 0),
            needs_engineer=counts.get(CaseStatus.NEEDS_ENGINEER.value, 0),
            exhausted=counts.get(CaseStatus.EXHAUSTED.value, 0),
            evidence_count=evidence_count,
        )

    def write_report(self, job_id: int, output_path: str | Path) -> Path:
        summary = self.report_summary(job_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": summary.job_id,
            "status": summary.status.value,
            "total_cases": summary.total_cases,
            "passed": summary.passed,
            "failed": summary.failed,
            "needs_engineer": summary.needs_engineer,
            "exhausted": summary.exhausted,
            "evidence_count": summary.evidence_count,
            "cases": [
                _case_to_report(case)
                for case in self.list_cases(job_id=job_id)
            ],
            "attempts": self._attempts_for_job(job_id),
        }
        output.write_text(json.dumps(redact_value(payload), indent=2, sort_keys=True) + "\n")
        return output

    def _attempts_for_job(self, job_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT case_id, attempt, lease_owner, outcome, summary,
                       evidence_json, created_at
                FROM case_attempts
                WHERE job_id = ?
                ORDER BY id
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "case_id": int(row["case_id"]),
                "attempt": int(row["attempt"]),
                "lease_owner": row["lease_owner"],
                "outcome": row["outcome"],
                "summary": redact_text(row["summary"]),
                "evidence": redact_value(json.loads(row["evidence_json"])),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _recover_stale_locked(self, conn: sqlite3.Connection, now_iso: str) -> int:
        rows = conn.execute(
            """
            SELECT id, job_id, attempts, max_attempts
            FROM cases
            WHERE status = ?
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= ?
            """,
            (CaseStatus.LEASED.value, now_iso),
        ).fetchall()
        for row in rows:
            next_status = (
                CaseStatus.QUEUED
                if int(row["attempts"]) < int(row["max_attempts"])
                else CaseStatus.EXHAUSTED
            )
            conn.execute(
                """
                UPDATE cases
                SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_status.value, now_iso, row["id"]),
            )
            self._refresh_job_status_locked(conn, int(row["job_id"]), now_iso)
        return len(rows)

    def _insert_attempt_locked(
        self,
        conn: sqlite3.Connection,
        case_row: sqlite3.Row,
        owner: str,
        result: RunnerResult,
        now: datetime,
    ) -> None:
        evidence = [
            {
                "kind": item.kind,
                "summary": redact_text(item.summary),
                "uri": item.uri,
                "metadata": redact_value(item.metadata),
            }
            for item in result.evidence
        ]
        conn.execute(
            """
            INSERT INTO case_attempts (
                case_id, job_id, attempt, lease_owner, outcome, summary,
                evidence_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_row["id"],
                case_row["job_id"],
                case_row["attempts"],
                owner,
                result.outcome.value,
                redact_text(result.summary),
                json.dumps(evidence, sort_keys=True),
                to_iso(now),
            ),
        )

    def _refresh_job_status_locked(
        self,
        conn: sqlite3.Connection,
        job_id: int,
        now_iso: str,
    ) -> None:
        statuses = [
            row["status"]
            for row in conn.execute("SELECT status FROM cases WHERE job_id = ?", (job_id,))
        ]
        if not statuses:
            status = JobStatus.FAILED
        elif any(item in {CaseStatus.QUEUED.value, CaseStatus.LEASED.value} for item in statuses):
            status = (
                JobStatus.RUNNING
                if any(item == CaseStatus.LEASED.value for item in statuses)
                else JobStatus.QUEUED
            )
        elif any(item == CaseStatus.NEEDS_ENGINEER.value for item in statuses):
            status = JobStatus.NEEDS_ENGINEER
        elif any(item in {CaseStatus.FAILED.value, CaseStatus.EXHAUSTED.value} for item in statuses):
            status = JobStatus.FAILED
        else:
            status = JobStatus.PASSED
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, now_iso, job_id),
        )


def _metadata_from_case(item: dict[str, Any], default: PolicyMetadata) -> PolicyMetadata:
    metadata = item.get("metadata") or {}
    return PolicyMetadata(
        tenant_id=str(metadata.get("tenant_id", default.tenant_id)),
        store_id=str(metadata.get("store_id", default.store_id)),
        risk=str(metadata.get("risk", default.risk)),
        allowed_operations=tuple(metadata.get("allowed_operations", default.allowed_operations)),
    )


def _status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _policy_from_row(row: sqlite3.Row) -> PolicyMetadata:
    return PolicyMetadata(
        tenant_id=row["tenant_id"],
        store_id=row["store_id"],
        risk=row["risk"],
        allowed_operations=tuple(json.loads(row["allowed_operations_json"])),
    )


def _job_from_row(row: sqlite3.Row) -> QAJob:
    return QAJob(
        id=int(row["id"]),
        name=row["name"],
        status=JobStatus(row["status"]),
        metadata=_policy_from_row(row),
        max_attempts=int(row["max_attempts"]),
        created_at=from_iso(row["created_at"]),
        updated_at=from_iso(row["updated_at"]),
    )


def _case_from_row(row: sqlite3.Row) -> QACase:
    return QACase(
        id=int(row["id"]),
        job_id=int(row["job_id"]),
        name=row["name"],
        status=CaseStatus(row["status"]),
        input_payload=json.loads(row["input_payload_json"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=from_iso(row["lease_expires_at"]),
        metadata=_policy_from_row(row),
        created_at=from_iso(row["created_at"]),
        updated_at=from_iso(row["updated_at"]),
    )


def _case_to_report(case: QACase) -> dict[str, Any]:
    return {
        "id": case.id,
        "job_id": case.job_id,
        "name": case.name,
        "status": case.status.value,
        "attempts": case.attempts,
        "max_attempts": case.max_attempts,
        "tenant_id": case.metadata.tenant_id,
        "store_id": case.metadata.store_id,
        "risk": case.metadata.risk,
        "allowed_operations": list(case.metadata.allowed_operations),
    }
