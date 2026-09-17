"""Profile-local receipts for Gemini delegation routing and daily reviews."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import time
from contextlib import AbstractContextManager, contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback


UTC = timezone.utc
DEFAULT_TIMEZONE = "America/Los_Angeles"
_JOURNAL_BUSY_RETRY_DELAYS = (0.01, 0.05, 0.1)
_RESPONSE_EXCERPT_MAX_BYTES = 32_768
_ERROR_EXCERPT_MAX_BYTES = 2_048
_TERMINAL_ATTEMPT_STATUSES = frozenset(
    {"completed", "failed", "timeout", "cancelled", "malformed", "denied", "oversized"}
)
_TERMINAL_BATCH_STATUSES = frozenset({"passed", "failed", "pipeline_failed"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | None) -> datetime:
    value = value or _utc_now()
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime | None = None) -> str:
    return _as_utc(value).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bounded_text_evidence(
    value: str | None, *, max_bytes: int
) -> tuple[str | None, str | None, int | None]:
    if value is None:
        return None, None, None
    encoded = value.encode("utf-8")
    excerpt = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return excerpt, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded_envelope_json(envelope: Mapping[str, Any] | None) -> str | None:
    if envelope is None:
        return None
    sanitized = dict(envelope)
    sanitized.pop("response", None)
    raw = _canonical_json(sanitized)
    encoded = raw.encode("utf-8")
    if len(encoded) <= _RESPONSE_EXCERPT_MAX_BYTES:
        return raw
    status = sanitized.get("status")
    return _canonical_json(
        {
            "truncated": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "status": status if isinstance(status, str) and len(status) <= 64 else None,
        }
    )


def _migrate_text_evidence(
    value: Any,
    existing_sha256: Any,
    existing_bytes: Any,
    *,
    max_bytes: int,
) -> tuple[str | None, str | None, int | None]:
    if value is None:
        return None, None, None
    text = value if isinstance(value, str) else str(value)
    excerpt, observed_sha256, observed_bytes = _bounded_text_evidence(
        text, max_bytes=max_bytes
    )
    if (
        isinstance(existing_sha256, str)
        and len(existing_sha256) == 64
        and all(char in "0123456789abcdef" for char in existing_sha256)
        and type(existing_bytes) is int
        and existing_bytes >= (observed_bytes or 0)
    ):
        return excerpt, existing_sha256, existing_bytes
    return excerpt, observed_sha256, observed_bytes


def _migrate_envelope_json(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        return _bounded_envelope_json(parsed)
    encoded = text.encode("utf-8")
    return _canonical_json(
        {
            "truncated": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "status": None,
        }
    )


def resolve_profile_receipt_path(hermes_home: Path, configured_path: str | Path) -> Path:
    """Resolve a configured receipt DB path without leaving the active profile."""
    root = hermes_home.resolve()
    configured = Path(configured_path)
    if configured.is_absolute():
        raise ValueError(
            "delegation.gemini_routing.receipt_db must be relative to HERMES_HOME"
        )
    candidate = (root / configured).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("delegation.gemini_routing.receipt_db escapes HERMES_HOME") from exc
    return candidate


def routing_day_for(when: datetime, timezone_name: str = DEFAULT_TIMEZONE) -> str:
    """Return the immutable local calendar day for an aware UTC instant."""
    return _as_utc(when).astimezone(ZoneInfo(timezone_name)).date().isoformat()


class GeminiReceiptStore:
    """Small SQLite repository isolated from Hermes conversation state."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else get_hermes_home() / "routing" / "gemini-routing.sqlite3"
        self._initialize()

    def _initialize(self) -> None:
        self._reject_unsafe_path(self.path.parent, expected="directory")
        self._reject_unsafe_path(self.path, expected="regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._enforce_private_mode(self.path.parent, 0o700)
        with self._transaction() as conn:
            self._ensure_schema(conn)
        self._enforce_sqlite_artifact_modes()

    @staticmethod
    def _reject_unsafe_path(path: Path, *, expected: str) -> None:
        if path.is_symlink():
            raise RuntimeError(f"receipt {expected} path must not be a symlink: {path}")
        if not path.exists():
            return
        valid = path.is_dir() if expected == "directory" else path.is_file()
        if not valid:
            raise RuntimeError(f"receipt database path must be a regular file: {path}")

    @staticmethod
    def _enforce_private_mode(path: Path, expected_mode: int) -> None:
        try:
            path.chmod(expected_mode)
            actual_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            raise RuntimeError(f"could not enforce private permissions for {path}") from exc
        if actual_mode != expected_mode:
            raise RuntimeError(
                f"could not enforce private permissions for {path}: "
                f"expected {oct(expected_mode)}, got {oct(actual_mode)}"
            )

    def _open_write(self) -> sqlite3.Connection:
        self._reject_unsafe_path(self.path, expected="regular file")
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            actual: str | None = None
            for attempt in range(len(_JOURNAL_BUSY_RETRY_DELAYS) + 1):
                try:
                    actual = apply_wal_with_fallback(
                        conn, db_label="routing/gemini-routing.sqlite3"
                    )
                    break
                except sqlite3.OperationalError as exc:
                    code = getattr(exc, "sqlite_errorcode", None)
                    message = str(exc).lower()
                    busy = code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or any(
                        marker in message for marker in ("locked", "busy")
                    )
                    if not busy or attempt == len(_JOURNAL_BUSY_RETRY_DELAYS):
                        raise
                    time.sleep(_JOURNAL_BUSY_RETRY_DELAYS[attempt])
            if actual not in {"wal", "delete"}:
                raise sqlite3.OperationalError(f"unsupported journal mode returned: {actual}")
            self._enforce_sqlite_artifact_modes()
        except Exception:
            conn.close()
            raise
        return conn

    def _open_readonly(self) -> sqlite3.Connection:
        self._reject_unsafe_path(self.path, expected="regular file")
        uri = f"file:{quote(str(self.path.resolve()), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._open_write()
        try:
            yield conn
        finally:
            try:
                conn.close()
            finally:
                self._enforce_sqlite_artifact_modes()

    def _enforce_sqlite_artifact_modes(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists() or candidate.is_symlink():
                self._reject_unsafe_path(candidate, expected="regular file")
                self._enforce_private_mode(candidate, 0o600)

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._open_readonly()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(
        self,
        commit_fence: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        with self._write_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                if commit_fence is None:
                    conn.execute("COMMIT")
                else:
                    with commit_fence():
                        conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        schema = """
            CREATE TABLE IF NOT EXISTS gemini_attempts (
                receipt_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL,
                parent_turn_id TEXT NOT NULL DEFAULT '',
                child_session_id TEXT NOT NULL,
                task_index INTEGER NOT NULL,
                routing_day TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                process_started_at_utc TEXT,
                completed_at_utc TEXT,
                route_requested TEXT NOT NULL,
                route_decision TEXT NOT NULL,
                route_reason TEXT NOT NULL,
                data_classification TEXT NOT NULL,
                output_contract TEXT NOT NULL,
                goal_text TEXT NOT NULL,
                context_text TEXT NOT NULL DEFAULT '',
                goal_sha256 TEXT NOT NULL,
                context_sha256 TEXT NOT NULL,
                prompt_sha256 TEXT NOT NULL,
                response_text TEXT,
                response_sha256 TEXT,
                response_bytes INTEGER,
                worker_status TEXT NOT NULL,
                process_exit_code INTEGER,
                duration_ms INTEGER NOT NULL,
                requested_provider TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                requested_effort TEXT NOT NULL,
                conversation_id TEXT,
                usage_json TEXT,
                raw_envelope_json TEXT,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                terminal_worker_route TEXT,
                terminal_provider TEXT,
                terminal_model TEXT,
                terminal_worker_status TEXT,
                terminal_response_text TEXT,
                terminal_response_sha256 TEXT,
                terminal_response_bytes INTEGER,
                terminal_error_code TEXT,
                error_code TEXT,
                error_message TEXT,
                error_message_sha256 TEXT,
                error_message_bytes INTEGER,
                created_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gemini_attempts_day
            ON gemini_attempts(routing_day, route_decision, receipt_id);

            CREATE TABLE IF NOT EXISTS gemini_schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_review_batches (
                batch_id TEXT PRIMARY KEY,
                routing_day TEXT NOT NULL UNIQUE,
                timezone TEXT NOT NULL,
                sample_size_requested INTEGER NOT NULL,
                eligible_count INTEGER NOT NULL,
                sample_seed_hex TEXT NOT NULL,
                sample_receipt_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                pipeline_error TEXT,
                alert_status TEXT NOT NULL DEFAULT 'not_needed',
                alert_message TEXT,
                alert_message_sha256 TEXT,
                alert_delivery_key TEXT,
                alert_lease_token TEXT,
                alert_lease_started_at_utc TEXT,
                review_lease_token TEXT,
                review_lease_started_at_utc TEXT,
                slack_channel_id TEXT,
                slack_workspace_id TEXT,
                slack_message_ts TEXT,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_review_items (
                batch_id TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                reviewer_provider TEXT NOT NULL,
                reviewer_model TEXT NOT NULL,
                review_status TEXT NOT NULL,
                verdict TEXT,
                failure_kind TEXT,
                reason TEXT,
                review_json TEXT,
                review_sha256 TEXT,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                error_code TEXT,
                error_message TEXT,
                PRIMARY KEY (batch_id, receipt_id),
                FOREIGN KEY (batch_id) REFERENCES daily_review_batches(batch_id),
                FOREIGN KEY (receipt_id) REFERENCES gemini_attempts(receipt_id)
            );
            """
        for statement in schema.split(";"):
            if statement.strip():
                conn.execute(statement)
        item_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(daily_review_items)")
        }
        if "failure_kind" not in item_columns:
            conn.execute("ALTER TABLE daily_review_items ADD COLUMN failure_kind TEXT")
        batch_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(daily_review_batches)")
        }
        if "alert_delivery_key" not in batch_columns:
            conn.execute("ALTER TABLE daily_review_batches ADD COLUMN alert_delivery_key TEXT")
        if "alert_message" not in batch_columns:
            conn.execute("ALTER TABLE daily_review_batches ADD COLUMN alert_message TEXT")
        if "slack_workspace_id" not in batch_columns:
            conn.execute("ALTER TABLE daily_review_batches ADD COLUMN slack_workspace_id TEXT")
        if "alert_lease_token" not in batch_columns:
            conn.execute("ALTER TABLE daily_review_batches ADD COLUMN alert_lease_token TEXT")
        if "alert_lease_started_at_utc" not in batch_columns:
            conn.execute(
                "ALTER TABLE daily_review_batches ADD COLUMN alert_lease_started_at_utc TEXT"
            )
        if "review_lease_token" not in batch_columns:
            conn.execute(
                "ALTER TABLE daily_review_batches ADD COLUMN review_lease_token TEXT"
            )
        if "review_lease_started_at_utc" not in batch_columns:
            conn.execute(
                "ALTER TABLE daily_review_batches ADD COLUMN review_lease_started_at_utc TEXT"
            )
        attempt_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(gemini_attempts)")
        }
        attempt_migrations = {
            "terminal_worker_route": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_worker_route TEXT"
            ),
            "terminal_provider": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_provider TEXT"
            ),
            "terminal_model": "ALTER TABLE gemini_attempts ADD COLUMN terminal_model TEXT",
            "terminal_worker_status": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_worker_status TEXT"
            ),
            "terminal_response_text": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_response_text TEXT"
            ),
            "terminal_response_sha256": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_response_sha256 TEXT"
            ),
            "terminal_response_bytes": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_response_bytes INTEGER"
            ),
            "terminal_error_code": (
                "ALTER TABLE gemini_attempts ADD COLUMN terminal_error_code TEXT"
            ),
            "response_bytes": "ALTER TABLE gemini_attempts ADD COLUMN response_bytes INTEGER",
            "error_message_sha256": (
                "ALTER TABLE gemini_attempts ADD COLUMN error_message_sha256 TEXT"
            ),
            "error_message_bytes": (
                "ALTER TABLE gemini_attempts ADD COLUMN error_message_bytes INTEGER"
            ),
        }
        for column, statement in attempt_migrations.items():
            if column not in attempt_columns:
                conn.execute(statement)

        if "fallback_used" in attempt_columns:
            conn.execute(
                """UPDATE gemini_attempts
                   SET terminal_worker_route=NULL,
                       terminal_provider=NULL,
                       terminal_model=NULL,
                       terminal_worker_status=NULL,
                       terminal_response_text=NULL,
                       terminal_response_sha256=NULL,
                       terminal_response_bytes=NULL,
                       terminal_error_code=NULL
                   WHERE fallback_used=1 AND terminal_worker_route='gemini'"""
            )

        evidence_columns = attempt_columns | set(attempt_migrations)
        evidence_required = {
            "response_text",
            "response_sha256",
            "response_bytes",
            "error_message",
            "error_message_sha256",
            "error_message_bytes",
            "raw_envelope_json",
            "terminal_response_text",
            "terminal_response_sha256",
            "terminal_response_bytes",
        }
        migration_name = "bound_attempt_evidence_v1"
        migration_applied = conn.execute(
            "SELECT 1 FROM gemini_schema_migrations WHERE name=?",
            (migration_name,),
        ).fetchone()
        if migration_applied is None and evidence_required <= evidence_columns:
            rows = conn.execute(
                """SELECT receipt_id,
                          response_text, response_sha256, response_bytes,
                          error_message, error_message_sha256, error_message_bytes,
                          raw_envelope_json,
                          terminal_response_text, terminal_response_sha256,
                          terminal_response_bytes
                   FROM gemini_attempts"""
            ).fetchall()
            for row in rows:
                response = _migrate_text_evidence(
                    row["response_text"],
                    row["response_sha256"],
                    row["response_bytes"],
                    max_bytes=_RESPONSE_EXCERPT_MAX_BYTES,
                )
                error = _migrate_text_evidence(
                    row["error_message"],
                    row["error_message_sha256"],
                    row["error_message_bytes"],
                    max_bytes=_ERROR_EXCERPT_MAX_BYTES,
                )
                terminal = _migrate_text_evidence(
                    row["terminal_response_text"],
                    row["terminal_response_sha256"],
                    row["terminal_response_bytes"],
                    max_bytes=_RESPONSE_EXCERPT_MAX_BYTES,
                )
                conn.execute(
                    """UPDATE gemini_attempts
                       SET response_text=?, response_sha256=?, response_bytes=?,
                           error_message=?, error_message_sha256=?, error_message_bytes=?,
                           raw_envelope_json=?,
                           terminal_response_text=?, terminal_response_sha256=?,
                           terminal_response_bytes=?
                       WHERE receipt_id=?""",
                    (
                        *response,
                        *error,
                        _migrate_envelope_json(row["raw_envelope_json"]),
                        *terminal,
                        row["receipt_id"],
                    ),
                )
            conn.execute(
                "INSERT INTO gemini_schema_migrations (name, applied_at_utc) VALUES (?, ?)",
                (migration_name, _iso()),
            )

        metadata_only_required = {
            "goal_text",
            "context_text",
            "goal_sha256",
            "context_sha256",
            "response_text",
            "response_sha256",
            "response_bytes",
            "error_message",
            "error_message_sha256",
            "error_message_bytes",
            "raw_envelope_json",
            "terminal_response_text",
            "terminal_response_sha256",
            "terminal_response_bytes",
            "conversation_id",
        }
        if metadata_only_required <= evidence_columns:
            rows = conn.execute(
                """SELECT receipt_id, goal_text, context_text,
                          goal_sha256, context_sha256,
                          response_text, response_sha256, response_bytes,
                          error_message, error_message_sha256, error_message_bytes,
                          terminal_response_text, terminal_response_sha256,
                          terminal_response_bytes
                   FROM gemini_attempts
                   WHERE goal_text != '' OR context_text != ''
                      OR response_text IS NOT NULL OR error_message IS NOT NULL
                      OR raw_envelope_json IS NOT NULL
                      OR terminal_response_text IS NOT NULL
                      OR conversation_id IS NOT NULL"""
            ).fetchall()
            for row in rows:
                response = _migrate_text_evidence(
                    row["response_text"],
                    row["response_sha256"],
                    row["response_bytes"],
                    max_bytes=_RESPONSE_EXCERPT_MAX_BYTES,
                )
                error = _migrate_text_evidence(
                    row["error_message"],
                    row["error_message_sha256"],
                    row["error_message_bytes"],
                    max_bytes=_ERROR_EXCERPT_MAX_BYTES,
                )
                terminal = _migrate_text_evidence(
                    row["terminal_response_text"],
                    row["terminal_response_sha256"],
                    row["terminal_response_bytes"],
                    max_bytes=_RESPONSE_EXCERPT_MAX_BYTES,
                )
                conn.execute(
                    """UPDATE gemini_attempts
                       SET goal_text='', context_text='',
                           goal_sha256=?, context_sha256=?,
                           response_text=NULL, response_sha256=?, response_bytes=?,
                           error_message=NULL, error_message_sha256=?, error_message_bytes=?,
                           raw_envelope_json=NULL, conversation_id=NULL,
                           terminal_response_text=NULL,
                           terminal_response_sha256=?, terminal_response_bytes=?
                       WHERE receipt_id=?""",
                    (
                        row["goal_sha256"] or _sha256(str(row["goal_text"] or "")),
                        row["context_sha256"] or _sha256(str(row["context_text"] or "")),
                        response[1],
                        response[2],
                        error[1],
                        error[2],
                        terminal[1],
                        terminal[2],
                        row["receipt_id"],
                    ),
                )
            conn.execute(
                "INSERT OR IGNORE INTO gemini_schema_migrations (name, applied_at_utc) VALUES (?, ?)",
                ("metadata_only_attempt_evidence_v1", _iso()),
            )

        review_metadata_required = {
            "batch_id",
            "receipt_id",
            "reason",
            "review_json",
            "review_sha256",
            "error_message",
        }
        if review_metadata_required <= item_columns:
            rows = conn.execute(
                """SELECT batch_id, receipt_id, review_json, review_sha256
                   FROM daily_review_items
                   WHERE reason IS NOT NULL OR review_json IS NOT NULL
                      OR error_message IS NOT NULL"""
            ).fetchall()
            for row in rows:
                raw = row["review_json"]
                conn.execute(
                    """UPDATE daily_review_items
                       SET reason=NULL, review_json=NULL, error_message=NULL,
                           review_sha256=?
                       WHERE batch_id=? AND receipt_id=?""",
                    (
                        row["review_sha256"]
                        or (_sha256(str(raw)) if raw is not None else None),
                        row["batch_id"],
                        row["receipt_id"],
                    ),
                )
            conn.execute(
                "INSERT OR IGNORE INTO gemini_schema_migrations (name, applied_at_utc) VALUES (?, ?)",
                ("metadata_only_review_evidence_v1", _iso()),
            )

    def prepare_attempt(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        task_index: int,
        route_requested: str,
        route_decision: str,
        route_reason: str,
        data_classification: str,
        output_contract: str,
        goal_text: str,
        requested_provider: str,
        requested_model: str,
        requested_effort: str,
        parent_turn_id: str = "",
        context_text: str = "",
        started_at: datetime | None = None,
        receipt_id: str | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> str:
        started = _as_utc(started_at)
        rid = receipt_id or f"grt_{secrets.token_hex(16)}"
        prompt = _canonical_json(
            {"goal": goal_text, "context": context_text, "output_contract": output_contract}
        )
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO gemini_attempts (
                    receipt_id, parent_session_id, parent_turn_id, child_session_id,
                    task_index, routing_day, started_at_utc, route_requested,
                    route_decision, route_reason, data_classification, output_contract,
                    goal_text, context_text, goal_sha256, context_sha256, prompt_sha256,
                    worker_status, duration_ms, requested_provider, requested_model,
                    requested_effort, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'prepared', 0, ?, ?, ?, ?)
                """,
                (
                    rid,
                    parent_session_id,
                    parent_turn_id,
                    child_session_id,
                    int(task_index),
                    routing_day_for(started, timezone_name),
                    started.isoformat(),
                    route_requested,
                    route_decision,
                    route_reason,
                    data_classification,
                    output_contract,
                    "",
                    "",
                    _sha256(goal_text),
                    _sha256(context_text),
                    _sha256(prompt),
                    requested_provider,
                    requested_model,
                    requested_effort,
                    _iso(),
                ),
            )
        return rid

    def mark_process_started(self, receipt_id: str, *, when: datetime | None = None) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE gemini_attempts
                   SET process_started_at_utc=?, worker_status='running'
                   WHERE receipt_id=? AND worker_status='prepared'
                """,
                (_iso(when), receipt_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("attempt is missing or process was already started")

    def complete_attempt(
        self,
        receipt_id: str,
        *,
        worker_status: str,
        duration_ms: int,
        response_text: str | None = None,
        process_exit_code: int | None = None,
        conversation_id: str | None = None,
        usage: Mapping[str, Any] | None = None,
        raw_envelope: Mapping[str, Any] | None = None,
        fallback_used: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
        response_sha256: str | None = None,
        response_bytes: int | None = None,
        completed_at: datetime | None = None,
        commit_fence: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        if worker_status not in _TERMINAL_ATTEMPT_STATUSES:
            raise ValueError(f"worker_status is not terminal: {worker_status}")
        _response_excerpt, computed_response_sha256, computed_response_bytes = _bounded_text_evidence(
            response_text, max_bytes=_RESPONSE_EXCERPT_MAX_BYTES
        )
        if (response_sha256 is None) != (response_bytes is None):
            raise ValueError("response_sha256 and response_bytes must be provided together")
        if response_sha256 is None:
            response_sha256 = computed_response_sha256
            response_bytes = computed_response_bytes
        else:
            if len(response_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in response_sha256
            ):
                raise ValueError("response_sha256 must be lowercase SHA-256 hex")
            if type(response_bytes) is not int:
                raise ValueError("response_bytes must be an integer")
            if response_text is not None and response_bytes < len(response_text.encode("utf-8")):
                raise ValueError("response_bytes cannot be smaller than the supplied response")
        _error_excerpt, error_sha256, error_bytes = _bounded_text_evidence(
            error_message, max_bytes=_ERROR_EXCERPT_MAX_BYTES
        )
        with self._transaction(commit_fence) as conn:
            current = conn.execute(
                """SELECT worker_status, requested_provider, requested_model
                   FROM gemini_attempts WHERE receipt_id=?""",
                (receipt_id,),
            ).fetchone()
            if current is None:
                raise KeyError(receipt_id)
            if current["worker_status"] in _TERMINAL_ATTEMPT_STATUSES:
                raise ValueError("attempt is already terminal")
            terminal_route = None if fallback_used else "gemini"
            terminal_provider = None if fallback_used else current["requested_provider"]
            terminal_model = None if fallback_used else current["requested_model"]
            terminal_status = None if fallback_used else worker_status
            terminal_response = None
            terminal_response_sha256 = None if fallback_used else response_sha256
            terminal_response_bytes = None if fallback_used else response_bytes
            terminal_error_code = None if fallback_used else error_code
            cursor = conn.execute(
                """
                UPDATE gemini_attempts SET
                    completed_at_utc=?, response_text=?, response_sha256=?, response_bytes=?,
                    worker_status=?, process_exit_code=?, duration_ms=?, conversation_id=?,
                    usage_json=?, raw_envelope_json=?, fallback_used=?,
                    terminal_worker_route=?, terminal_provider=?, terminal_model=?,
                    terminal_worker_status=?, terminal_response_text=?,
                    terminal_response_sha256=?, terminal_response_bytes=?, terminal_error_code=?,
                    error_code=?, error_message=?, error_message_sha256=?, error_message_bytes=?
                WHERE receipt_id=?
                """,
                (
                    _iso(completed_at),
                    None,
                    response_sha256,
                    response_bytes,
                    worker_status,
                    process_exit_code,
                    max(0, int(duration_ms)),
                    None,
                    _canonical_json(dict(usage)) if usage is not None else None,
                    None,
                    int(bool(fallback_used)),
                    terminal_route,
                    terminal_provider,
                    terminal_model,
                    terminal_status,
                    terminal_response,
                    terminal_response_sha256,
                    terminal_response_bytes,
                    terminal_error_code,
                    error_code,
                    None,
                    error_sha256,
                    error_bytes,
                    receipt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(receipt_id)

    def record_fallback_outcome(
        self,
        receipt_id: str,
        *,
        worker_route: str,
        provider: str,
        model: str,
        worker_status: str,
        response_text: str | None,
        error_code: str | None,
        response_sha256: str | None = None,
        response_bytes: int | None = None,
        commit_fence: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        """Persist the actual terminal Sol identity without replacing Gemini audit data."""
        if worker_status not in _TERMINAL_ATTEMPT_STATUSES:
            raise ValueError(f"fallback worker_status is not terminal: {worker_status}")
        if worker_route != "sol" or not provider or not model:
            raise ValueError("fallback route, provider, and model must identify Sol")
        _response_excerpt, computed_response_sha256, computed_response_bytes = _bounded_text_evidence(
            response_text, max_bytes=_RESPONSE_EXCERPT_MAX_BYTES
        )
        if (response_sha256 is None) != (response_bytes is None):
            raise ValueError("response_sha256 and response_bytes must be provided together")
        if response_sha256 is None:
            response_sha256 = computed_response_sha256
            response_bytes = computed_response_bytes
        else:
            if len(response_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in response_sha256
            ):
                raise ValueError("response_sha256 must be lowercase SHA-256 hex")
            if type(response_bytes) is not int or response_bytes < 0:
                raise ValueError("response_bytes must be a non-negative integer")
            if response_text is not None and response_bytes < len(response_text.encode("utf-8")):
                raise ValueError("response_bytes cannot be smaller than the supplied response")
        with self._transaction(commit_fence) as conn:
            cursor = conn.execute(
                """UPDATE gemini_attempts
                   SET terminal_worker_route=?, terminal_provider=?, terminal_model=?,
                       terminal_worker_status=?, terminal_response_text=?,
                       terminal_response_sha256=?, terminal_response_bytes=?, terminal_error_code=?
                   WHERE receipt_id=? AND fallback_used=1
                      AND completed_at_utc IS NOT NULL
                      AND terminal_worker_route IS NULL
                """,
                (
                    worker_route,
                    provider,
                    model,
                    worker_status,
                    None,
                    response_sha256,
                    response_bytes,
                    error_code,
                    receipt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(receipt_id)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def get_attempt(self, receipt_id: str) -> dict[str, Any]:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM gemini_attempts WHERE receipt_id=?", (receipt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        return dict(row)

    def count_attempts(self) -> int:
        with self._read_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gemini_attempts").fetchone()[0])

    def list_started_attempts_for_day(self, routing_day: str | date) -> list[dict[str, Any]]:
        day = routing_day.isoformat() if isinstance(routing_day, date) else str(routing_day)
        with self._read_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM gemini_attempts
                   WHERE routing_day=? AND route_decision='gemini'
                     AND process_started_at_utc IS NOT NULL
                   ORDER BY receipt_id
                """,
                (day,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_or_get_review_batch(
        self,
        *,
        routing_day: str,
        timezone_name: str,
        sample_size_requested: int,
        eligible_count: int,
        sample_seed_hex: str,
        sample_receipt_ids: Sequence[str],
        started_at: datetime | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        bid = batch_id or f"grb_{secrets.token_hex(16)}"
        now = _iso(started_at)
        sample_json = _canonical_json(list(sample_receipt_ids))
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO daily_review_batches (
                       batch_id, routing_day, timezone, sample_size_requested,
                       eligible_count, sample_seed_hex, sample_receipt_ids_json,
                       status, started_at_utc, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?)
                """,
                (
                    bid,
                    routing_day,
                    timezone_name,
                    int(sample_size_requested),
                    int(eligible_count),
                    sample_seed_hex,
                    sample_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM daily_review_batches WHERE routing_day=?", (routing_day,)
            ).fetchone()
        if row is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("review batch insert vanished")
        return dict(row)

    def get_review_batch(self, routing_day: str) -> dict[str, Any] | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_review_batches WHERE routing_day=?", (routing_day,)
            ).fetchone()
        return self._row(row)

    def claim_review_batch(
        self, batch_id: str, *, lease_token: str, now: datetime
    ) -> bool:
        """Atomically grant one runner authority to execute a prepared batch."""
        if not lease_token:
            raise ValueError("review lease_token must be non-empty")
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET status='reviewing', review_lease_token=?,
                       review_lease_started_at_utc=?
                   WHERE batch_id=? AND status='preparing'
                """,
                (lease_token, _iso(now), batch_id),
            )
        return cursor.rowcount == 1

    def fail_stale_review_batch(
        self,
        batch_id: str,
        *,
        stale_before: datetime,
        pipeline_error: str,
        alert_message: str,
        slack_channel_id: str,
        completed_at: datetime,
        slack_workspace_id: str | None = None,
    ) -> bool:
        """Atomically terminalize an abandoned review lease exactly once."""
        alert_sha256 = _sha256(alert_message)
        delivery_key = f"gemini-daily-review:{batch_id}:{alert_sha256}"
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET status='pipeline_failed', completed_at_utc=?, pipeline_error=?,
                       alert_status='pending', alert_message=?, alert_message_sha256=?,
                       alert_delivery_key=?, slack_channel_id=?, slack_workspace_id=?
                   WHERE batch_id=?
                     AND status IN ('preparing', 'reviewing')
                     AND (
                         (status='preparing' AND started_at_utc <= ?)
                         OR (
                             status='reviewing'
                             AND COALESCE(review_lease_started_at_utc, started_at_utc) <= ?
                         )
                     )
                """,
                (
                    _iso(completed_at),
                    pipeline_error,
                    alert_message,
                    alert_sha256,
                    delivery_key,
                    slack_channel_id,
                    slack_workspace_id,
                    batch_id,
                    _iso(stale_before),
                    _iso(stale_before),
                ),
            )
        return cursor.rowcount == 1

    def count_review_batches(self) -> int:
        with self._read_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM daily_review_batches").fetchone()[0])

    def claim_alert_delivery(
        self,
        batch_id: str,
        *,
        lease_token: str,
        now: datetime,
        stale_before: datetime,
    ) -> bool:
        """Atomically admit one pending sender or reclaim one stale sender lease."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET alert_status='sending', alert_lease_token=?,
                       alert_lease_started_at_utc=?
                   WHERE batch_id=?
                     AND (
                         alert_status='pending'
                         OR (
                             alert_status='sending'
                             AND alert_lease_started_at_utc IS NOT NULL
                             AND alert_lease_started_at_utc <= ?
                         )
                     )
                """,
                (lease_token, _iso(now), batch_id, _iso(stale_before)),
            )
        return cursor.rowcount == 1

    def release_alert_delivery(self, batch_id: str, *, lease_token: str) -> bool:
        """Return an owned failed delivery attempt to the durable pending state."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET alert_status='pending', alert_lease_token=NULL,
                       alert_lease_started_at_utc=NULL
                   WHERE batch_id=? AND alert_status='sending' AND alert_lease_token=?
                """,
                (batch_id, lease_token),
            )
        return cursor.rowcount == 1

    def complete_alert_delivery(
        self,
        batch_id: str,
        *,
        lease_token: str,
        slack_channel_id: str,
        slack_message_ts: str,
    ) -> bool:
        """Persist a confirmed Slack receipt only for the current lease owner."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches
                   SET alert_status='sent', slack_channel_id=?, slack_message_ts=?,
                       alert_lease_token=NULL, alert_lease_started_at_utc=NULL
                   WHERE batch_id=? AND alert_status='sending' AND alert_lease_token=?
                """,
                (slack_channel_id, slack_message_ts, batch_id, lease_token),
            )
        return cursor.rowcount == 1

    def update_review_batch(
        self,
        batch_id: str,
        *,
        lease_token: str,
        status: str,
        pipeline_error: str | None = None,
        alert_status: str | None = None,
        alert_message: str | None = None,
        alert_delivery_key: str | None = None,
        slack_channel_id: str | None = None,
        slack_workspace_id: str | None = None,
        slack_message_ts: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if not lease_token:
            raise ValueError("review lease_token must be non-empty")
        if status not in {"preparing", "reviewing", *_TERMINAL_BATCH_STATUSES}:
            raise ValueError(f"invalid batch status: {status}")
        terminal_at = _iso(completed_at) if status in _TERMINAL_BATCH_STATUSES else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches SET
                       status=?, completed_at_utc=?, pipeline_error=?,
                       alert_status=COALESCE(?, alert_status),
                       alert_message=COALESCE(?, alert_message),
                       alert_message_sha256=COALESCE(?, alert_message_sha256),
                       alert_delivery_key=COALESCE(?, alert_delivery_key),
                       slack_channel_id=COALESCE(?, slack_channel_id),
                       slack_workspace_id=COALESCE(?, slack_workspace_id),
                       slack_message_ts=COALESCE(?, slack_message_ts),
                       review_lease_token=NULL, review_lease_started_at_utc=NULL
                   WHERE batch_id=? AND status='reviewing' AND review_lease_token=?
                """,
                (
                    status,
                    terminal_at,
                    pipeline_error,
                    alert_status,
                    alert_message,
                    _sha256(alert_message) if alert_message is not None else None,
                    alert_delivery_key,
                    slack_channel_id,
                    slack_workspace_id,
                    slack_message_ts,
                    batch_id,
                    lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"review batch missing or review authority lost: {batch_id}")

    def initialize_alert_outbox(
        self,
        batch_id: str,
        *,
        alert_message: str,
        alert_delivery_key: str,
        slack_channel_id: str,
        slack_workspace_id: str | None = None,
    ) -> bool:
        """Fill pending legacy outbox fields without disturbing an active lease."""
        alert_sha256 = _sha256(alert_message)
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE daily_review_batches SET
                       alert_message=COALESCE(alert_message, ?),
                       alert_message_sha256=COALESCE(alert_message_sha256, ?),
                       alert_delivery_key=COALESCE(alert_delivery_key, ?),
                       slack_channel_id=COALESCE(slack_channel_id, ?),
                       slack_workspace_id=COALESCE(slack_workspace_id, ?)
                   WHERE batch_id=? AND alert_status='pending'
                     AND (alert_message IS NULL OR alert_message=?)
                     AND (alert_message_sha256 IS NULL OR alert_message_sha256=?)
                     AND (alert_delivery_key IS NULL OR alert_delivery_key=?)
                     AND (slack_channel_id IS NULL OR slack_channel_id=?)
                     AND (slack_workspace_id IS NULL OR slack_workspace_id IS ?)
                """,
                (
                    alert_message,
                    alert_sha256,
                    alert_delivery_key,
                    slack_channel_id,
                    slack_workspace_id,
                    batch_id,
                    alert_message,
                    alert_sha256,
                    alert_delivery_key,
                    slack_channel_id,
                    slack_workspace_id,
                ),
            )
        return cursor.rowcount == 1

    def add_review_item(
        self,
        *,
        batch_id: str,
        lease_token: str,
        receipt_id: str,
        ordinal: int,
        reviewer_provider: str,
        reviewer_model: str,
        review_status: str,
        verdict: str | None = None,
        failure_kind: str | None = None,
        reason: str | None = None,
        review_json: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if not lease_token:
            raise ValueError("review lease_token must be non-empty")
        raw = _canonical_json(dict(review_json)) if review_json is not None else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO daily_review_items (
                       batch_id, receipt_id, ordinal, reviewer_provider, reviewer_model,
                       review_status, verdict, failure_kind, reason, review_json, review_sha256,
                       started_at_utc, completed_at_utc, error_code, error_message
                   )
                   SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                   FROM daily_review_batches
                   WHERE batch_id=? AND status='reviewing' AND review_lease_token=?
                """,
                (
                    batch_id,
                    receipt_id,
                    int(ordinal),
                    reviewer_provider,
                    reviewer_model,
                    review_status,
                    verdict,
                    failure_kind,
                    None,
                    None,
                    _sha256(raw) if raw is not None else None,
                    _iso(started_at),
                    _iso(completed_at) if completed_at is not None else None,
                    error_code,
                    None,
                    batch_id,
                    lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"review batch missing or review authority lost: {batch_id}")

    def list_review_items(self, batch_id: str) -> list[dict[str, Any]]:
        with self._read_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_review_items WHERE batch_id=? ORDER BY ordinal, receipt_id",
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def apply_retention(
        self,
        *,
        now: datetime | None = None,
        raw_days: int = 30,
        aggregate_days: int = 180,
    ) -> dict[str, int]:
        current = _as_utc(now)
        raw_cutoff = (current - timedelta(days=max(0, int(raw_days)))).isoformat()
        aggregate_cutoff = (current - timedelta(days=max(0, int(aggregate_days)))).isoformat()
        with self._transaction() as conn:
            raw_cursor = conn.execute(
                """UPDATE gemini_attempts SET
                       goal_text='', context_text='', response_text=NULL,
                       usage_json=NULL, raw_envelope_json=NULL, error_message=NULL,
                       terminal_response_text=NULL
                   WHERE started_at_utc < ?
                      AND (goal_text != '' OR context_text != '' OR response_text IS NOT NULL
                           OR usage_json IS NOT NULL OR raw_envelope_json IS NOT NULL
                           OR error_message IS NOT NULL
                           OR terminal_response_text IS NOT NULL)
                """,
                (raw_cutoff,),
            )
            review_raw_cursor = conn.execute(
                """UPDATE daily_review_items
                   SET reason='', review_json=NULL
                   WHERE batch_id IN (
                       SELECT batch_id FROM daily_review_batches
                       WHERE created_at_utc < ?
                   )
                     AND (reason != '' OR review_json IS NOT NULL)
                """,
                (raw_cutoff,),
            )
            old_batches = [
                row[0]
                for row in conn.execute(
                    "SELECT batch_id FROM daily_review_batches WHERE created_at_utc < ?",
                    (aggregate_cutoff,),
                ).fetchall()
            ]
            deleted_items = 0
            deleted_batches = 0
            if old_batches:
                marks = ",".join("?" for _ in old_batches)
                deleted_items = conn.execute(
                    f"DELETE FROM daily_review_items WHERE batch_id IN ({marks})", old_batches
                ).rowcount
                deleted_batches = conn.execute(
                    f"DELETE FROM daily_review_batches WHERE batch_id IN ({marks})", old_batches
                ).rowcount
            deleted_attempts = conn.execute(
                """DELETE FROM gemini_attempts
                   WHERE created_at_utc < ?
                     AND receipt_id NOT IN (SELECT receipt_id FROM daily_review_items)
                """,
                (aggregate_cutoff,),
            ).rowcount
        return {
            "raw_redacted": raw_cursor.rowcount,
            "review_raw_redacted": review_raw_cursor.rowcount,
            "review_items_deleted": deleted_items,
            "review_batches_deleted": deleted_batches,
            "attempts_deleted": deleted_attempts,
        }
