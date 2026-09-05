"""Profile-scoped durable receipt ledger with atomic claim/finalize/retry."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .policy import FeedbackReceipt


_SQLITE_BUSY_TIMEOUT_MS = 5_000
_SQLITE_WAL_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def _enable_wal_with_bounded_retry(connection: sqlite3.Connection) -> None:
    """Enable WAL without losing startup to a concurrent opener.

    SQLite can report ``unable to open database file`` while another process
    is changing the journal mode.  Install the busy timeout before the mode
    change, retry only that bounded initialization step, and still fail closed
    for a persistent or unrelated SQLite error.
    """

    connection.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    delays = (0.0, *_SQLITE_WAL_RETRY_DELAYS)
    last_error: sqlite3.OperationalError | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            current_mode = connection.execute("PRAGMA journal_mode").fetchone()
            if current_mode and str(current_mode[0]).casefold() == "wal":
                return
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
        except sqlite3.OperationalError as error:
            if "unable to open database file" not in str(error).casefold():
                raise
            last_error = error
            continue
        if not mode or str(mode[0]).casefold() != "wal":
            raise sqlite3.OperationalError(
                f"SQLite WAL initialization returned {mode!r}"
            )
        return
    assert last_error is not None
    raise last_error

try:  # Hermes supplies the profile-aware source of truth at runtime.
    from hermes_constants import get_hermes_home
except ImportError:  # Standalone unit tests remain dependency-free.

    def get_hermes_home() -> Path:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


@dataclass(frozen=True, slots=True)
class ClaimLease:
    owner: str
    claimed_at: datetime
    version: int


@dataclass(frozen=True, slots=True)
class PendingTaskBinding:
    receipt: FeedbackReceipt
    task_id: str


@dataclass(frozen=True, slots=True)
class MergeLease:
    repository: str
    pr_number: int
    head_sha: str
    owner: str
    claimed_at: datetime


@dataclass(frozen=True, slots=True)
class CIRunLease:
    run_id: str
    version: int
    supervisor_pid: int


@dataclass(frozen=True, slots=True)
class WorktreeSlotLease:
    slot_id: int
    version: int
    owner_pid: int


@dataclass(frozen=True, slots=True)
class MaintenanceReceipt:
    repository: str
    head_sha: str
    lane: str
    status: str
    summary: str
    completed_at: datetime


class LedgerStateError(RuntimeError):
    """The caller tried to finalize or fail a receipt it does not hold."""


class FeedbackLedger:
    """SQLite-backed state for one profile; all receipt transitions are atomic."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        _enable_wal_with_bounded_retry(self._connection)
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA wal_autocheckpoint=1000")
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback_receipts (
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                feedback_kind TEXT NOT NULL,
                feedback_id TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('claimed', 'completed', 'failed')),
                task_id TEXT,
                last_error TEXT,
                attempts INTEGER NOT NULL,
                claim_owner TEXT,
                claimed_at TEXT,
                lease_version INTEGER NOT NULL DEFAULT 0,
                workspace_path TEXT,
                expected_sha TEXT,
                PRIMARY KEY (repository, pr_number, feedback_kind, feedback_id, head_sha)
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_heads (
                repository TEXT NOT NULL,
                base_branch TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                first_observed_at TEXT NOT NULL,
                PRIMARY KEY (repository, base_branch)
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_receipts (
                repository TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                lane TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('passed', 'failed')),
                summary TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (repository, head_sha, lane)
            )
            """)
        self._migrate_lease_columns()
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS ci_audit_receipts (
                receipt_id TEXT PRIMARY KEY,
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                base_sha TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('passed', 'failed')),
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                UNIQUE (repository, pr_number, head_sha, manifest_digest, completed_at)
            )
            """)
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS ci_receipt_lookup ON ci_audit_receipts "
            "(repository, pr_number, head_sha, manifest_digest, status, completed_at)"
        )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS ci_audit_runs (
                run_id TEXT PRIMARY KEY,
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                base_sha TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                supervisor_pid INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                lease_version INTEGER NOT NULL,
                receipt_id TEXT,
                last_error TEXT,
                UNIQUE (repository, pr_number, base_sha, head_sha, manifest_digest)
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS merge_attempts (
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                head_sha TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('claimed', 'verification_required', 'completed', 'failed')
                ),
                owner TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                receipt_json TEXT,
                last_error TEXT,
                PRIMARY KEY (repository, pr_number, head_sha)
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS deployment_receipts (
                receipt_id TEXT PRIMARY KEY,
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                merge_commit_oid TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
                completed_at TEXT NOT NULL,
                receipt_json TEXT NOT NULL
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback_scan_watermarks (
                repository TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                head_sha TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                PRIMARY KEY (repository, pr_number)
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS worktree_pool_slots (
                slot_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('leased', 'free')),
                owner_pid INTEGER NOT NULL,
                head_sha TEXT,
                task_id TEXT,
                board TEXT,
                claimed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                lease_version INTEGER NOT NULL
            )
            """)
        self._migrate_worktree_pool_columns()

    def _migrate_worktree_pool_columns(self) -> None:
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(worktree_pool_slots)")
        }
        additions = {
            "task_id": "task_id TEXT",
            "board": "board TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE worktree_pool_slots ADD COLUMN {declaration}"
                )

    def _migrate_lease_columns(self) -> None:
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(feedback_receipts)")
        }
        additions = {
            "claim_owner": "claim_owner TEXT",
            "claimed_at": "claimed_at TEXT",
            "lease_version": "lease_version INTEGER NOT NULL DEFAULT 0",
            "workspace_path": "workspace_path TEXT",
            "expected_sha": "expected_sha TEXT",
            "action_status": "action_status TEXT NOT NULL DEFAULT 'pending'",
            "actioned_head_sha": "actioned_head_sha TEXT",
            "actioned_at": "actioned_at TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE feedback_receipts ADD COLUMN {declaration}"
                )

    @classmethod
    def for_current_profile(cls) -> FeedbackLedger:
        return cls(cls.current_profile_path())

    @classmethod
    def current_profile_path(cls) -> Path:
        return get_hermes_home() / "github-pr-feedback" / "ledger.sqlite3"

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def feedback_scan_is_current(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        updated_at: datetime,
    ) -> bool:
        """Return whether feedback was read for this exact current PR update."""

        updated_at = _aware_utc(updated_at, "updated_at")
        row = self._connection.execute(
            "SELECT head_sha, updated_at FROM feedback_scan_watermarks "
            "WHERE repository = ? AND pr_number = ?",
            (repository, pr_number),
        ).fetchone()
        return row is not None and row == (
            head_sha.casefold(),
            updated_at.isoformat(),
        )

    def record_feedback_scan(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        updated_at: datetime,
        *,
        scanned_at: datetime,
    ) -> None:
        """Advance a PR watermark only after its feedback read succeeded."""

        updated_at = _aware_utc(updated_at, "updated_at")
        scanned_at = _aware_utc(scanned_at, "scanned_at")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO feedback_scan_watermarks "
                "(repository, pr_number, head_sha, updated_at, scanned_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(repository, pr_number) DO UPDATE SET "
                "head_sha = excluded.head_sha, updated_at = excluded.updated_at, "
                "scanned_at = excluded.scanned_at",
                (
                    repository,
                    pr_number,
                    head_sha.casefold(),
                    updated_at.isoformat(),
                    scanned_at.isoformat(),
                ),
            )

    def observe_maintenance_head(
        self,
        repository: str,
        base_branch: str,
        head_sha: str,
        *,
        observed_at: datetime,
    ) -> datetime:
        """Return the first time the current exact base head was observed."""

        observed_at = _aware_utc(observed_at, "observed_at")
        identity = tuple(value.strip() for value in (repository, base_branch, head_sha))
        if not all(identity):
            raise ValueError("maintenance head identity must be non-empty")
        with self._transaction():
            row = self._connection.execute(
                "SELECT head_sha, first_observed_at FROM maintenance_heads "
                "WHERE repository = ? AND base_branch = ?",
                identity[:2],
            ).fetchone()
            if row is not None and row[0] == identity[2]:
                return datetime.fromisoformat(row[1])
            self._connection.execute(
                "INSERT INTO maintenance_heads "
                "(repository, base_branch, head_sha, first_observed_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(repository, base_branch) DO UPDATE SET "
                "head_sha = excluded.head_sha, first_observed_at = excluded.first_observed_at",
                (*identity, observed_at.isoformat()),
            )
        return observed_at

    def record_maintenance_receipt(
        self,
        *,
        repository: str,
        head_sha: str,
        lane: str,
        status: str,
        summary: str,
        completed_at: datetime,
    ) -> None:
        """Record one immutable, exact-head lane outcome; identical retries are idempotent."""

        completed_at = _aware_utc(completed_at, "completed_at")
        values = tuple(
            value.strip() for value in (repository, head_sha, lane, status, summary)
        )
        if not all(values[:4]) or status not in {"passed", "failed"}:
            raise ValueError("maintenance receipt is invalid")
        if len(summary) > 4000:
            raise ValueError("maintenance receipt summary is too long")
        row_values = (*values, completed_at.isoformat())
        with self._transaction():
            existing = self._connection.execute(
                "SELECT repository, head_sha, lane, status, summary, completed_at "
                "FROM maintenance_receipts WHERE repository = ? AND head_sha = ? AND lane = ?",
                values[:3],
            ).fetchone()
            if existing is not None:
                if tuple(existing) != row_values:
                    raise LedgerStateError("maintenance receipt is immutable")
                return
            self._connection.execute(
                "INSERT INTO maintenance_receipts "
                "(repository, head_sha, lane, status, summary, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row_values,
            )

    def maintenance_receipts(
        self, repository: str, head_sha: str
    ) -> dict[str, MaintenanceReceipt]:
        rows = self._connection.execute(
            "SELECT lane, status, summary, completed_at FROM maintenance_receipts "
            "WHERE repository = ? AND head_sha = ?",
            (repository, head_sha),
        )
        return {
            row[0]: MaintenanceReceipt(
                repository=repository,
                head_sha=head_sha,
                lane=row[0],
                status=row[1],
                summary=row[2],
                completed_at=datetime.fromisoformat(row[3]),
            )
            for row in rows
        }

    def claim(
        self,
        receipt: FeedbackReceipt,
        *,
        owner: str,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> ClaimLease | None:
        owner = owner.strip() if isinstance(owner, str) else ""
        if not owner:
            raise ValueError("claim owner must be a non-empty string")
        claimed_at = _aware_utc(claimed_at, "claimed_at")
        stale_before = _aware_utc(stale_before, "stale_before")
        with self._transaction():
            serialized_repair = not (
                receipt.feedback_kind == "pr_repair"
                and receipt.feedback_id.startswith("report:")
            )
            if serialized_repair:
                active_repair = self._connection.execute(
                    "SELECT 1 FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
                    "AND head_sha = ? AND feedback_kind != 'pr_local_ci' "
                    "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%') "
                    "AND NOT (feedback_kind = ? AND feedback_id = ?) "
                    "AND status IN ('claimed', 'completed') AND action_status = 'pending' LIMIT 1",
                    (
                        receipt.repository,
                        receipt.pr_number,
                        receipt.head_sha,
                        receipt.feedback_kind,
                        receipt.feedback_id,
                    ),
                ).fetchone()
                if active_repair is not None:
                    return None
            row = self._connection.execute(
                "SELECT status, claimed_at, lease_version FROM feedback_receipts "
                "WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                receipt.key,
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO feedback_receipts "
                    "(repository, pr_number, feedback_kind, feedback_id, head_sha, status, attempts, "
                    "claim_owner, claimed_at, lease_version) "
                    "VALUES (?, ?, ?, ?, ?, 'claimed', 1, ?, ?, 1)",
                    (*receipt.key, owner, claimed_at.isoformat()),
                )
                return ClaimLease(owner, claimed_at, 1)
            status, stored_claimed_at, stored_version = row
            stale = stored_claimed_at is None
            if stored_claimed_at is not None:
                try:
                    stale = (
                        datetime.fromisoformat(stored_claimed_at).astimezone(UTC)
                        <= stale_before
                    )
                except (TypeError, ValueError):
                    stale = True
            if status != "claimed" or not stale:
                return None
            version = int(stored_version or 0) + 1
            self._connection.execute(
                "UPDATE feedback_receipts SET claim_owner = ?, claimed_at = ?, lease_version = ?, "
                "last_error = NULL, attempts = attempts + 1 WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ? AND status = 'claimed'",
                (owner, claimed_at.isoformat(), version, *receipt.key),
            )
            return ClaimLease(owner, claimed_at, version)

    def was_completed_on_any_head(self, receipt: FeedbackReceipt) -> bool:
        """Return whether this immutable feedback item was already queued successfully."""

        row = self._connection.execute(
            "SELECT 1 FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
            "AND feedback_kind = ? AND feedback_id = ? AND status = 'completed' LIMIT 1",
            receipt.key[:4],
        ).fetchone()
        return row is not None

    def exact_receipt_status(self, receipt: FeedbackReceipt) -> str | None:
        """Return the durable status for this exact immutable receipt, if present."""

        row = self._connection.execute(
            "SELECT status FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
            "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
            receipt.key,
        ).fetchone()
        if row is None:
            return None
        status = row[0]
        if status not in {"claimed", "completed", "failed"}:
            raise LedgerStateError("stored feedback receipt status is invalid")
        return str(status)

    def pending_task_bindings_for_head(
        self, receipt: FeedbackReceipt
    ) -> tuple[PendingTaskBinding, ...]:
        """Return every pending dispatch that serializes work on this exact PR head."""

        rows = self._connection.execute(
            "SELECT feedback_kind, feedback_id, task_id FROM feedback_receipts "
            "WHERE repository = ? AND pr_number = ? AND head_sha = ? "
            "AND feedback_kind != 'pr_local_ci' "
            "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%') "
            "AND status = 'completed' "
            "AND action_status = 'pending' ORDER BY claimed_at, feedback_kind, feedback_id",
            (receipt.repository, receipt.pr_number, receipt.head_sha),
        )
        bindings: list[PendingTaskBinding] = []
        for feedback_kind, feedback_id, task_id in rows:
            if not isinstance(task_id, str) or not task_id.strip():
                continue
            bindings.append(
                PendingTaskBinding(
                    FeedbackReceipt(
                        receipt.repository,
                        receipt.pr_number,
                        str(feedback_kind),
                        str(feedback_id),
                        receipt.head_sha,
                    ),
                    task_id.strip(),
                )
            )
        return tuple(bindings)

    def exact_pending_task_binding(
        self, receipt: FeedbackReceipt
    ) -> PendingTaskBinding | None:
        """Return the exact completed dispatch that still awaits a work outcome."""

        row = self._connection.execute(
            "SELECT task_id FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
            "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ? "
            "AND status = 'completed' AND action_status = 'pending'",
            receipt.key,
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0].strip():
            return None
        return PendingTaskBinding(receipt, row[0].strip())

    def reopen_archived_exact_dispatch(
        self,
        receipt: FeedbackReceipt,
        *,
        archived: PendingTaskBinding,
        owner: str,
        claimed_at: datetime,
    ) -> ClaimLease | None:
        """Atomically reopen one exact dispatch after its Kanban card was archived."""

        owner = owner.strip() if isinstance(owner, str) else ""
        if not owner or archived.receipt != receipt or not archived.task_id.strip():
            raise ValueError("claim owner and exact archived binding must be valid")
        claimed_at = _aware_utc(claimed_at, "claimed_at")
        with self._transaction():
            row = self._connection.execute(
                "SELECT task_id, status, action_status, lease_version "
                "FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                receipt.key,
            ).fetchone()
            if (
                row is None
                or row[0] != archived.task_id
                or row[1] != "completed"
                or row[2] != "pending"
            ):
                return None
            version = int(row[3] or 0) + 1
            reopened = self._connection.execute(
                "UPDATE feedback_receipts SET status = 'claimed', task_id = NULL, "
                "last_error = NULL, attempts = attempts + 1, claim_owner = ?, claimed_at = ?, "
                "lease_version = ? WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                "AND feedback_id = ? AND head_sha = ? AND status = 'completed' "
                "AND action_status = 'pending' AND task_id = ?",
                (
                    owner,
                    claimed_at.isoformat(),
                    version,
                    *receipt.key,
                    archived.task_id,
                ),
            )
            if reopened.rowcount != 1:
                raise LedgerStateError("exact archived dispatch changed during replacement")
            return ClaimLease(owner, claimed_at, version)

    def replace_archived_dispatches(
        self,
        receipt: FeedbackReceipt,
        *,
        archived: tuple[PendingTaskBinding, ...],
        owner: str,
        claimed_at: datetime,
    ) -> ClaimLease | None:
        """Atomically supersede a verified archived set and acquire its replacement."""

        owner = owner.strip() if isinstance(owner, str) else ""
        if not owner or not archived:
            raise ValueError("claim owner and archived bindings must be non-empty")
        claimed_at = _aware_utc(claimed_at, "claimed_at")
        expected = {
            (binding.receipt.feedback_kind, binding.receipt.feedback_id, binding.task_id)
            for binding in archived
            if binding.receipt.repository == receipt.repository
            and binding.receipt.pr_number == receipt.pr_number
            and binding.receipt.head_sha == receipt.head_sha
            and binding.task_id
        }
        if len(expected) != len(archived):
            raise ValueError("archived bindings must be unique and match the replacement head")
        with self._transaction():
            rows = tuple(
                self._connection.execute(
                    "SELECT feedback_kind, feedback_id, task_id, status, lease_version "
                    "FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
                    "AND head_sha = ? AND feedback_kind != 'pr_local_ci' "
                    "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%') "
                    "AND status IN ('claimed', 'completed') AND action_status = 'pending'",
                    (receipt.repository, receipt.pr_number, receipt.head_sha),
                )
            )
            if any(row[3] != "completed" or not row[2] for row in rows):
                return None
            actual = {(str(row[0]), str(row[1]), str(row[2])) for row in rows}
            if actual != expected:
                return None
            exact_row = next(
                (
                    row
                    for row in rows
                    if (str(row[0]), str(row[1]))
                    == (receipt.feedback_kind, receipt.feedback_id)
                ),
                None,
            )
            if exact_row is None:
                existing = self._connection.execute(
                    "SELECT 1 FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
                    "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                    receipt.key,
                ).fetchone()
                if existing is not None:
                    return None
            for row in rows:
                if exact_row is not None and row is exact_row:
                    continue
                result = self._connection.execute(
                    "UPDATE feedback_receipts SET action_status = 'superseded', "
                    "last_error = 'archived Kanban task superseded by current exact-head work' "
                    "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                    "AND feedback_id = ? AND head_sha = ? AND status = 'completed' "
                    "AND action_status = 'pending' AND task_id = ?",
                    (
                        receipt.repository,
                        receipt.pr_number,
                        str(row[0]),
                        str(row[1]),
                        receipt.head_sha,
                        str(row[2]),
                    ),
                )
                if result.rowcount != 1:
                    raise LedgerStateError("archived dispatch set changed during replacement")
            if exact_row is None:
                self._connection.execute(
                    "INSERT INTO feedback_receipts "
                    "(repository, pr_number, feedback_kind, feedback_id, head_sha, status, "
                    "attempts, claim_owner, claimed_at, lease_version) "
                    "VALUES (?, ?, ?, ?, ?, 'claimed', 1, ?, ?, 1)",
                    (*receipt.key, owner, claimed_at.isoformat()),
                )
                return ClaimLease(owner, claimed_at, 1)
            version = int(exact_row[4] or 0) + 1
            reopened = self._connection.execute(
                "UPDATE feedback_receipts SET status = 'claimed', task_id = NULL, "
                "last_error = NULL, attempts = attempts + 1, claim_owner = ?, claimed_at = ?, "
                "lease_version = ? WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                "AND feedback_id = ? AND head_sha = ? AND status = 'completed' "
                "AND action_status = 'pending' AND task_id = ?",
                (
                    owner,
                    claimed_at.isoformat(),
                    version,
                    *receipt.key,
                    str(exact_row[2]),
                ),
            )
            if reopened.rowcount != 1:
                raise LedgerStateError("exact archived dispatch changed during replacement")
            return ClaimLease(owner, claimed_at, version)

    def was_actioned_on_any_head(self, receipt: FeedbackReceipt) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
            "AND feedback_kind = ? AND feedback_id = ? AND action_status = 'completed' LIMIT 1",
            receipt.key[:4],
        ).fetchone()
        return row is not None

    def mark_feedback_actioned(
        self,
        receipt: FeedbackReceipt,
        *,
        resolved_head_sha: str,
        actioned_at: datetime,
    ) -> None:
        if receipt.feedback_kind == "pr_local_ci":
            raise ValueError("local CI dispatches are not feedback actions")
        if (
            not isinstance(resolved_head_sha, str)
            or len(resolved_head_sha) != 40
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in resolved_head_sha
            )
        ):
            raise ValueError("resolved_head_sha must be a full hexadecimal SHA")
        actioned_at = _aware_utc(actioned_at, "actioned_at")
        with self._transaction():
            result = self._connection.execute(
                "UPDATE feedback_receipts SET action_status = 'completed', actioned_head_sha = ?, "
                "actioned_at = ? WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                "AND feedback_id = ? AND head_sha = ? AND status = 'completed'",
                (resolved_head_sha.lower(), actioned_at.isoformat(), *receipt.key),
            )
            if result.rowcount != 1:
                raise LedgerStateError("feedback dispatch is not complete")

    def begin_feedback_action(
        self,
        receipt: FeedbackReceipt,
        *,
        resolved_head_sha: str,
        actioned_at: datetime,
    ) -> None:
        """Durably admit an idempotent external feedback reconciliation."""

        if receipt.feedback_kind == "pr_local_ci":
            raise ValueError("local CI dispatches are not feedback actions")
        if (
            not isinstance(resolved_head_sha, str)
            or len(resolved_head_sha) != 40
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in resolved_head_sha
            )
        ):
            raise ValueError("resolved_head_sha must be a full hexadecimal SHA")
        actioned_at = _aware_utc(actioned_at, "actioned_at")
        normalized_head = resolved_head_sha.casefold()
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, action_status, actioned_head_sha FROM feedback_receipts "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
                "AND feedback_id = ? AND head_sha = ?",
                receipt.key,
            ).fetchone()
            if row is None or row[0] != "completed":
                raise LedgerStateError("feedback dispatch is not complete")
            if row[1] in {"resolving", "completed"}:
                if row[2] != normalized_head:
                    raise LedgerStateError("feedback action head changed")
                return
            if row[1] != "pending":
                raise LedgerStateError("feedback action status is invalid")
            self._connection.execute(
                "UPDATE feedback_receipts SET action_status = 'resolving', "
                "actioned_head_sha = ?, actioned_at = ? WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                (normalized_head, actioned_at.isoformat(), *receipt.key),
            )

    def retry(
        self,
        receipt: FeedbackReceipt,
        *,
        owner: str,
        claimed_at: datetime,
        not_before: datetime | None = None,
    ) -> ClaimLease | None:
        """Atomically retry a receipt after an explicitly recorded dispatch failure."""

        owner = owner.strip() if isinstance(owner, str) else ""
        if not owner:
            raise ValueError("claim owner must be a non-empty string")
        claimed_at = _aware_utc(claimed_at, "claimed_at")
        if not_before is not None:
            not_before = _aware_utc(not_before, "not_before")
        with self._transaction():
            row = self._connection.execute(
                "SELECT lease_version, claimed_at FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ? AND status = 'failed'",
                receipt.key,
            ).fetchone()
            if row is None:
                return None
            if not_before is not None and row[1] is not None:
                try:
                    if datetime.fromisoformat(str(row[1])).astimezone(UTC) > not_before:
                        return None
                except (TypeError, ValueError):
                    # A malformed timestamp is conservative: it is eligible for
                    # one bounded retry, which can repair the row's timestamp.
                    pass
            version = int(row[0] or 0) + 1
            result = self._connection.execute(
                "UPDATE feedback_receipts SET status = 'claimed', last_error = NULL, attempts = attempts + 1, "
                "claim_owner = ?, claimed_at = ?, lease_version = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
                "AND head_sha = ? AND status = 'failed'",
                (owner, claimed_at.isoformat(), version, *receipt.key),
            )
            return (
                ClaimLease(owner, claimed_at, version) if result.rowcount == 1 else None
            )

    def failed_retry_state(
        self,
        receipt: FeedbackReceipt,
        *,
        now: datetime,
        backoff: timedelta,
    ) -> str:
        """Classify a failed dispatch without mutating the receipt.

        The scanner uses this to keep a failed row's backoff separate from a
        duplicate.  In particular, a not-yet-due row must not consume the
        bounded local-CI admission slot or prevent older backlog from being
        considered.
        """

        if backoff < timedelta(0):
            raise ValueError("backoff must be non-negative")
        now = _aware_utc(now, "now")
        row = self._connection.execute(
            "SELECT status, claimed_at FROM feedback_receipts "
            "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? "
            "AND feedback_id = ? AND head_sha = ?",
            receipt.key,
        ).fetchone()
        if row is None or row[0] != "failed":
            return "not_failed"
        if row[1] is None:
            return "due"
        try:
            failed_at = datetime.fromisoformat(str(row[1])).astimezone(UTC)
        except (TypeError, ValueError):
            return "due"
        return "due" if now >= failed_at + backoff else "backoff"

    def finalize(
        self, receipt: FeedbackReceipt, task_id: str, lease: ClaimLease
    ) -> None:
        task_id = task_id.strip() if isinstance(task_id, str) else ""
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, task_id, claim_owner, lease_version FROM feedback_receipts "
                "WHERE repository = ? AND pr_number = ? "
                "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                receipt.key,
            ).fetchone()
            if (
                row is None
                or row[0] != "claimed"
                or row[2] != lease.owner
                or int(row[3] or 0) != lease.version
            ):
                raise LedgerStateError("receipt lease is not held")
            if row[0] == "claimed":
                self._connection.execute(
                    "UPDATE feedback_receipts SET status = 'completed', task_id = ? "
                    "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
                    (task_id, *receipt.key),
                )

    def fail(self, receipt: FeedbackReceipt, error: str, lease: ClaimLease) -> None:
        error = error.strip() if isinstance(error, str) else ""
        if not error:
            raise ValueError("error must be a non-empty string")
        with self._transaction():
            result = self._connection.execute(
                "UPDATE feedback_receipts SET status = 'failed', last_error = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
                "AND head_sha = ? AND status = 'claimed' AND claim_owner = ? AND lease_version = ?",
                (error[:1000], *receipt.key, lease.owner, lease.version),
            )
            if result.rowcount != 1:
                raise LedgerStateError("receipt lease is not held")

    def record_workspace(
        self,
        receipt: FeedbackReceipt,
        lease: ClaimLease,
        workspace_path: Path,
        expected_sha: str,
    ) -> None:
        if expected_sha.casefold() != receipt.head_sha.casefold():
            raise ValueError("expected SHA must equal receipt head SHA")
        workspace = Path(workspace_path)
        if not workspace.is_absolute():
            raise ValueError("workspace path must be absolute")
        with self._transaction():
            result = self._connection.execute(
                "UPDATE feedback_receipts SET workspace_path = ?, expected_sha = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
                "AND head_sha = ? AND status = 'claimed' AND claim_owner = ? AND lease_version = ?",
                (
                    str(workspace),
                    expected_sha,
                    *receipt.key,
                    lease.owner,
                    lease.version,
                ),
            )
            if result.rowcount != 1:
                raise LedgerStateError("receipt lease is not held")

    def record_expected_head(
        self,
        receipt: FeedbackReceipt,
        lease: ClaimLease,
        expected_sha: str,
    ) -> None:
        if expected_sha.casefold() != receipt.head_sha.casefold():
            raise ValueError("expected SHA must equal receipt head SHA")
        with self._transaction():
            result = self._connection.execute(
                "UPDATE feedback_receipts SET expected_sha = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
                "AND head_sha = ? AND status = 'claimed' AND claim_owner = ? AND lease_version = ?",
                (expected_sha, *receipt.key, lease.owner, lease.version),
            )
            if result.rowcount != 1:
                raise LedgerStateError("receipt lease is not held")

    def status_counts(self) -> dict[str, int]:
        """Return durable receipt counts without exposing receipt evidence."""

        counts = {"claimed": 0, "completed": 0, "failed": 0}
        for status, count in self._connection.execute(
            "SELECT status, COUNT(*) FROM feedback_receipts GROUP BY status"
        ):
            if status in counts:
                counts[status] = int(count)
        return counts

    def record_ci_receipt(self, receipt: object) -> None:
        """Atomically persist one typed receipt; prose cannot enter this boundary."""

        from .ci_runner import CIAuditReceipt

        if not isinstance(receipt, CIAuditReceipt):
            raise TypeError("receipt must be a CIAuditReceipt")
        payload = json.dumps(
            receipt.to_payload(), sort_keys=True, separators=(",", ":")
        )
        if len(payload.encode("utf-8")) > 1_000_000:
            raise ValueError("CI receipt evidence exceeds its bounded limit")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO ci_audit_receipts "
                "(receipt_id, repository, pr_number, base_sha, head_sha, manifest_digest, status, "
                "started_at, completed_at, evidence_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_id,
                    receipt.identity.repository,
                    receipt.identity.pr_number,
                    receipt.identity.base_sha,
                    receipt.identity.head_sha,
                    receipt.manifest_digest,
                    receipt.status,
                    receipt.started_at.isoformat(),
                    receipt.completed_at.isoformat(),
                    payload,
                ),
            )

    def claim_ci_run(
        self,
        repository: str,
        pr_number: int,
        base_sha: str,
        head_sha: str,
        manifest_digest: str,
        *,
        supervisor_pid: int,
        claimed_at: datetime,
        stale_before: datetime,
        pid_is_alive: Callable[[int], bool | None],
    ) -> CIRunLease | None:
        """Claim one exact CI identity, fencing duplicate or live supervisors."""

        if supervisor_pid < 2:
            raise ValueError("CI supervisor PID must identify a real process")
        claimed = _aware_utc(claimed_at, "claimed_at")
        stale = _aware_utc(stale_before, "stale_before")
        key = (repository, pr_number, base_sha, head_sha, manifest_digest)
        run_id = hashlib.sha256(
            "\0".join(map(str, key)).encode("utf-8")
        ).hexdigest()
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, supervisor_pid, updated_at, lease_version "
                "FROM ci_audit_runs WHERE repository = ? AND pr_number = ? AND base_sha = ? "
                "AND head_sha = ? AND manifest_digest = ?",
                key,
            ).fetchone()
            version = 1
            if row is not None:
                version = int(row[3]) + 1
                if row[0] == "running":
                    updated_at = datetime.fromisoformat(str(row[2]))
                    alive = pid_is_alive(int(row[1]))
                    if alive is True or (alive is None and updated_at >= stale):
                        return None
                self._connection.execute(
                    "UPDATE ci_audit_runs SET status = 'running', supervisor_pid = ?, "
                    "started_at = ?, updated_at = ?, lease_version = ?, receipt_id = NULL, "
                    "last_error = NULL WHERE run_id = ?",
                    (supervisor_pid, claimed.isoformat(), claimed.isoformat(), version, run_id),
                )
            else:
                self._connection.execute(
                    "INSERT INTO ci_audit_runs (run_id, repository, pr_number, base_sha, head_sha, "
                    "manifest_digest, status, supervisor_pid, started_at, updated_at, lease_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
                    (run_id, *key, supervisor_pid, claimed.isoformat(), claimed.isoformat(), version),
                )
        return CIRunLease(run_id, version, supervisor_pid)

    def finish_ci_run(
        self,
        lease: CIRunLease,
        *,
        status: str,
        completed_at: datetime,
        receipt_id: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("CI run terminal status is invalid")
        completed = _aware_utc(completed_at, "completed_at")
        with self._transaction():
            result = self._connection.execute(
                "UPDATE ci_audit_runs SET status = ?, updated_at = ?, receipt_id = ?, "
                "last_error = ? WHERE run_id = ? AND status = 'running' AND lease_version = ? "
                "AND supervisor_pid = ?",
                (
                    status,
                    completed.isoformat(),
                    receipt_id,
                    (error or "")[:1000] or None,
                    lease.run_id,
                    lease.version,
                    lease.supervisor_pid,
                ),
            )
            if result.rowcount != 1:
                raise LedgerStateError("CI run lease is not held")

    def claim_worktree_slot(
        self,
        slot_id: int,
        *,
        owner_pid: int,
        head_sha: str | None,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> WorktreeSlotLease | None:
        """Claim one worktree-pool slot, fencing a not-yet-stale lease.

        Deliberately time-only, unlike claim_ci_run's PID-liveness check: a
        worktree-pool slot is handed off to a *dispatched Kanban agent task*
        that outlives the short-lived dispatcher process making this call, so
        the dispatcher's own PID going away is not evidence the slot is
        free -- only elapsed time against stale_before is. Callers must pass
        a stale_before comfortably longer than the longest task
        max_runtime_seconds that can hold a slot.
        """

        if owner_pid < 2:
            raise ValueError("worktree slot owner PID must identify a real process")
        claimed = _aware_utc(claimed_at, "claimed_at")
        stale = _aware_utc(stale_before, "stale_before")
        with self._transaction():
            row = self._connection.execute(
                "SELECT status, owner_pid, updated_at, lease_version, task_id "
                "FROM worktree_pool_slots WHERE slot_id = ?",
                (slot_id,),
            ).fetchone()
            version = 1
            if row is not None:
                version = int(row[3]) + 1
                if row[0] == "leased":
                    # A bound Kanban task owns this checkout until the board
                    # positively reports that task terminal and
                    # reconcile_leases releases it.  Time-based reclamation is
                    # only safe for an unbound lease stranded mid-dispatch;
                    # retryable blocked/triage cards may legitimately outlive
                    # the timeout and must retain their exact-head workspace.
                    if row[4] is not None:
                        return None
                    updated_at = datetime.fromisoformat(str(row[2]))
                    if updated_at >= stale:
                        return None
                self._connection.execute(
                    "UPDATE worktree_pool_slots SET status = 'leased', owner_pid = ?, "
                    "head_sha = ?, claimed_at = ?, updated_at = ?, lease_version = ? "
                    "WHERE slot_id = ?",
                    (owner_pid, head_sha, claimed.isoformat(), claimed.isoformat(), version, slot_id),
                )
            else:
                self._connection.execute(
                    "INSERT INTO worktree_pool_slots (slot_id, status, owner_pid, head_sha, "
                    "claimed_at, updated_at, lease_version) VALUES (?, 'leased', ?, ?, ?, ?, ?)",
                    (slot_id, owner_pid, head_sha, claimed.isoformat(), claimed.isoformat(), version),
                )
        return WorktreeSlotLease(slot_id, version, owner_pid)

    def finish_worktree_slot(self, lease: WorktreeSlotLease) -> None:
        """Release a held slot back to the free pool. Idempotent no-op if the

        lease was already reclaimed by orphan recovery (never raises --
        releasing a slot you no longer hold is not an error, unlike failing a
        CI run you no longer hold).
        """

        with self._transaction():
            self._connection.execute(
                "UPDATE worktree_pool_slots SET status = 'free', updated_at = ? "
                "WHERE slot_id = ? AND lease_version = ? AND owner_pid = ? AND status = 'leased'",
                (
                    _aware_utc(datetime.now(UTC), "updated_at").isoformat(),
                    lease.slot_id,
                    lease.version,
                    lease.owner_pid,
                ),
            )

    def bind_worktree_slot_task(self, head_sha: str, task_id: str, board: str) -> None:
        """Record which dispatched Kanban task now owns a leased slot.

        Best-effort by design: if the slot was already reconciled away (or a
        non-pooled LocalGit is in use and no such slot exists), this is a
        silent no-op rather than an error -- proactive release is an
        optimization over the lease timeout, not a correctness requirement.
        """

        with self._transaction():
            self._connection.execute(
                "UPDATE worktree_pool_slots SET task_id = ?, board = ? "
                "WHERE head_sha = ? AND status = 'leased'",
                (task_id, board, head_sha),
            )

    def leased_worktree_slots(self) -> tuple[dict[str, object], ...]:
        """List every currently-leased slot with an attached task binding."""

        rows = self._connection.execute(
            "SELECT slot_id, lease_version, owner_pid, task_id, board "
            "FROM worktree_pool_slots WHERE status = 'leased' AND task_id IS NOT NULL "
            "AND board IS NOT NULL"
        ).fetchall()
        return tuple(
            {
                "slot_id": int(row[0]),
                "lease_version": int(row[1]),
                "owner_pid": int(row[2]),
                "task_id": row[3],
                "board": row[4],
            }
            for row in rows
        )

    def latest_ci_run(
        self, repository: str, pr_number: int, head_sha: str
    ) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT run_id, status, supervisor_pid, lease_version, receipt_id, updated_at, last_error "
            "FROM ci_audit_runs WHERE repository = ? AND pr_number = ? AND head_sha = ? "
            "ORDER BY updated_at DESC LIMIT 1",
            (repository, pr_number, head_sha),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "status": row[1],
            "supervisor_pid": int(row[2]),
            "lease_version": int(row[3]),
            "receipt_id": row[4],
            "updated_at": row[5],
            "last_error": row[6],
        }

    def latest_passing_ci_receipt(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        *,
        manifest_digest: str,
        not_before: datetime,
    ) -> object | None:
        """Return only a fresh typed passing receipt for an exact head and manifest."""

        from .ci_runner import CIAuditReceipt

        boundary = _aware_utc(not_before, "not_before")
        row = self._connection.execute(
            "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
            "AND head_sha = ? AND manifest_digest = ? AND status = 'passed' AND completed_at >= ? "
            "ORDER BY completed_at DESC LIMIT 1",
            (repository, pr_number, head_sha, manifest_digest, boundary.isoformat()),
        ).fetchone()
        if row is None:
            return None
        try:
            receipt = CIAuditReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored CI receipt is invalid") from error
        if receipt.status != "passed":
            raise LedgerStateError("stored CI receipt status is inconsistent")
        return receipt

    def latest_ci_receipt(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        *,
        manifest_digest: str,
        not_before: datetime,
    ) -> object | None:
        """Return the latest typed receipt, including a failed audit, for an exact lane."""

        from .ci_runner import CIAuditReceipt

        boundary = _aware_utc(not_before, "not_before")
        row = self._connection.execute(
            "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
            "AND head_sha = ? AND manifest_digest = ? AND completed_at >= ? "
            "ORDER BY completed_at DESC LIMIT 1",
            (repository, pr_number, head_sha, manifest_digest, boundary.isoformat()),
        ).fetchone()
        if row is None:
            return None
        try:
            return CIAuditReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored CI receipt is invalid") from error

    def latest_ci_receipt_for_head(
        self, repository: str, pr_number: int, head_sha: str
    ) -> object | None:
        """Return the newest typed audit receipt for an exact PR head."""

        from .ci_runner import CIAuditReceipt

        row = self._connection.execute(
            "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
            "AND head_sha = ? ORDER BY completed_at DESC LIMIT 1",
            (repository, pr_number, head_sha),
        ).fetchone()
        if row is None:
            return None
        try:
            return CIAuditReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored CI receipt is invalid") from error

    def ci_receipt_by_id(
        self, repository: str, pr_number: int, receipt_id: str
    ) -> object | None:
        """Return one immutable typed audit receipt by its globally unique ID."""

        from .ci_runner import CIAuditReceipt

        row = self._connection.execute(
            "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
            "AND receipt_id = ? LIMIT 1",
            (repository, pr_number, receipt_id.casefold()),
        ).fetchone()
        if row is None:
            return None
        try:
            return CIAuditReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored CI receipt is invalid") from error

    def completed_merge_receipt(self, repository: str, pr_number: int) -> object | None:
        from .merge_controller import MergeReceipt

        row = self._connection.execute(
            "SELECT receipt_json FROM merge_attempts WHERE repository = ? AND pr_number = ? "
            "AND status = 'completed' ORDER BY updated_at DESC LIMIT 1",
            (repository, pr_number),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return MergeReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored merge receipt is invalid") from error

    def verification_required_merge_lease(
        self, repository: str, pr_number: int
    ) -> MergeLease | None:
        """Return the immutable attempt identity awaiting canonical readback."""

        row = self._connection.execute(
            "SELECT head_sha, owner, claimed_at FROM merge_attempts "
            "WHERE repository = ? AND pr_number = ? AND status = 'verification_required' "
            "ORDER BY updated_at DESC LIMIT 1",
            (repository, pr_number),
        ).fetchone()
        if row is None:
            return None
        return MergeLease(
            repository,
            pr_number,
            str(row[0]),
            str(row[1]),
            _aware_utc(datetime.fromisoformat(str(row[2])), "claimed_at"),
        )

    def verification_required_merge_numbers(self, repository: str) -> tuple[int, ...]:
        """List ambiguous merge attempts that still require canonical readback."""

        rows = self._connection.execute(
            "SELECT pr_number FROM merge_attempts WHERE repository = ? "
            "AND status = 'verification_required' ORDER BY pr_number",
            (repository,),
        ).fetchall()
        return tuple(int(row[0]) for row in rows)

    def complete_verified_merge(
        self,
        lease: MergeLease,
        *,
        updated_at: datetime,
        receipt: object,
    ) -> None:
        """Atomically reconcile one ambiguous write from canonical merged truth."""

        from .merge_controller import MergeReceipt

        if not isinstance(receipt, MergeReceipt):
            raise ValueError("verified merge completion requires a typed receipt")
        updated_at = _aware_utc(updated_at, "updated_at")
        receipt_json = json.dumps(
            receipt.to_payload(), sort_keys=True, separators=(",", ":")
        )
        with self._transaction():
            result = self._connection.execute(
                "UPDATE merge_attempts SET status = 'completed', updated_at = ?, "
                "receipt_json = ?, last_error = NULL WHERE repository = ? AND pr_number = ? "
                "AND head_sha = ? AND status = 'verification_required' AND owner = ? "
                "AND claimed_at = ?",
                (
                    updated_at.isoformat(),
                    receipt_json,
                    lease.repository,
                    lease.pr_number,
                    lease.head_sha,
                    lease.owner,
                    lease.claimed_at.isoformat(),
                ),
            )
            if result.rowcount != 1:
                raise LedgerStateError("merge verification attempt changed")

    def claim_merge_lease(
        self,
        repository: str,
        pr_number: int,
        head_sha: str,
        *,
        owner: str,
        claimed_at: datetime,
    ) -> MergeLease | None:
        owner = owner.strip() if isinstance(owner, str) else ""
        if not owner:
            raise ValueError("merge lease owner must be a non-empty string")
        claimed_at = _aware_utc(claimed_at, "claimed_at")
        with self._transaction():
            completed = self._connection.execute(
                "SELECT 1 FROM merge_attempts WHERE repository = ? AND pr_number = ? "
                "AND status = 'completed' LIMIT 1",
                (repository, pr_number),
            ).fetchone()
            if completed is not None:
                return None
            existing = self._connection.execute(
                "SELECT status FROM merge_attempts WHERE repository = ? AND pr_number = ? "
                "AND head_sha = ?",
                (repository, pr_number, head_sha),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    "INSERT INTO merge_attempts "
                    "(repository, pr_number, head_sha, status, owner, claimed_at, updated_at) "
                    "VALUES (?, ?, ?, 'claimed', ?, ?, ?)",
                    (
                        repository,
                        pr_number,
                        head_sha,
                        owner,
                        claimed_at.isoformat(),
                        claimed_at.isoformat(),
                    ),
                )
            elif existing[0] == "failed":
                self._connection.execute(
                    "UPDATE merge_attempts SET status = 'claimed', owner = ?, claimed_at = ?, "
                    "updated_at = ?, receipt_json = NULL, last_error = NULL WHERE repository = ? "
                    "AND pr_number = ? AND head_sha = ? AND status = 'failed'",
                    (
                        owner,
                        claimed_at.isoformat(),
                        claimed_at.isoformat(),
                        repository,
                        pr_number,
                        head_sha,
                    ),
                )
            else:
                return None
        return MergeLease(repository, pr_number, head_sha, owner, claimed_at)

    def finish_merge_lease(
        self,
        lease: MergeLease,
        *,
        status: str,
        updated_at: datetime,
        receipt: object | None = None,
        error: str | None = None,
    ) -> None:
        from .merge_controller import MergeReceipt

        if status not in {"verification_required", "completed", "failed"}:
            raise ValueError("merge terminal status is invalid")
        if status == "completed" and not isinstance(receipt, MergeReceipt):
            raise ValueError("completed merge requires a typed receipt")
        updated_at = _aware_utc(updated_at, "updated_at")
        receipt_json = (
            json.dumps(receipt.to_payload(), sort_keys=True, separators=(",", ":"))
            if isinstance(receipt, MergeReceipt)
            else None
        )
        with self._transaction():
            result = self._connection.execute(
                "UPDATE merge_attempts SET status = ?, updated_at = ?, receipt_json = ?, "
                "last_error = ? WHERE repository = ? AND pr_number = ? AND head_sha = ? "
                "AND status = 'claimed' AND owner = ? AND claimed_at = ?",
                (
                    status,
                    updated_at.isoformat(),
                    receipt_json,
                    (error or "")[:1000] or None,
                    lease.repository,
                    lease.pr_number,
                    lease.head_sha,
                    lease.owner,
                    lease.claimed_at.isoformat(),
                ),
            )
            if result.rowcount != 1:
                raise LedgerStateError("merge lease is not held")

    def merge_status_counts(self) -> dict[str, int]:
        counts = {
            "claimed": 0,
            "verification_required": 0,
            "completed": 0,
            "failed": 0,
        }
        for status, count in self._connection.execute(
            "SELECT status, COUNT(*) FROM merge_attempts GROUP BY status"
        ):
            if status in counts:
                counts[status] = int(count)
        return counts

    def record_deployment_receipt(self, receipt: object) -> None:
        from .post_merge import DeploymentReceipt

        if not isinstance(receipt, DeploymentReceipt):
            raise TypeError("receipt must be a DeploymentReceipt")
        payload = json.dumps(
            receipt.to_payload(), sort_keys=True, separators=(",", ":")
        )
        with self._transaction():
            self._connection.execute(
                "INSERT INTO deployment_receipts "
                "(receipt_id, repository, pr_number, merge_commit_oid, status, completed_at, "
                "receipt_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_id,
                    receipt.repository,
                    receipt.pr_number,
                    receipt.merge_commit_oid,
                    receipt.status,
                    receipt.completed_at.isoformat(),
                    payload,
                ),
            )

    def latest_deployment_receipt(
        self, repository: str, pr_number: int
    ) -> object | None:
        from .post_merge import DeploymentReceipt

        row = self._connection.execute(
            "SELECT receipt_json FROM deployment_receipts WHERE repository = ? AND pr_number = ? "
            "ORDER BY completed_at DESC LIMIT 1",
            (repository, pr_number),
        ).fetchone()
        if row is None:
            return None
        try:
            return DeploymentReceipt.from_payload(json.loads(row[0]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LedgerStateError("stored deployment receipt is invalid") from error

    def close(self) -> None:
        self._connection.close()


def _aware_utc(value: datetime, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)
