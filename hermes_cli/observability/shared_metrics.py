"""Durable aggregation and local export for Hermes shared metrics."""

from __future__ import annotations

import atexit
import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from typing import Any

from hermes_cli.sqlite_util import add_column_if_missing, write_txn
from hermes_constants import get_hermes_home
from utils import atomic_json_write

from .shared_metrics_contract import (
    CLIENT_ACTIVE_METRIC,
    COUNTER_METRICS,
    FEATURE_ADOPTION_METRIC,
    INSTALL_SNAPSHOT_METRIC,
    MILESTONE_METRIC,
    MODEL_ROUTE_METRIC,
    SUM_METRICS,
    client_resource_is_valid,
    counter_dimensions_are_valid,
)


_PACKAGE_SCHEMA_VERSION = "hermes.shared_metrics.v4"
_STORE_SCHEMA_VERSION = "2"
_BUSY_TIMEOUT_MS = 250
_SCHEMA_BUSY_TIMEOUT_MS = 5_000
# Writes a busy store could not take within _BUSY_TIMEOUT_MS, kept in memory for the next write.
_DEFERRED_WRITES_MAX = 1_024
_LOCAL_HISTORY_RETENTION_DAYS = 30
_ACTIVE_INSTALL_STATE_KEY = "client_active_recorded_at"
_INSTALL_SNAPSHOT_STATE_KEY = "install_snapshot_recorded_at"
_MILESTONE_STATE_PREFIX = "milestone:"
_FEATURE_STATE_PREFIX = "feature:"
_DAILY_ONCE_STATE_PREFIX = "daily_once:"
_ACTIVE_INSTALL_INTERVAL = timedelta(hours=24)
# Column order of the client resource in every counter_aggregates statement.
_RESOURCE_COLUMNS = ("hermes_version", "os_family", "architecture", "install_method")

_CREATE_TELEMETRY_STATE_SQL = """
            CREATE TABLE IF NOT EXISTS telemetry_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
_CREATE_COUNTER_AGGREGATES_SQL = """
            CREATE TABLE IF NOT EXISTS counter_aggregates (
                period_start TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                hermes_version TEXT NOT NULL,
                os_family TEXT NOT NULL,
                architecture TEXT NOT NULL,
                install_method TEXT NOT NULL,
                dimensions_json TEXT NOT NULL,
                value INTEGER NOT NULL,
                packaged_value INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    period_start, metric_name, hermes_version, os_family, architecture,
                    install_method, dimensions_json
                )
            )
            """
_CREATE_PACKAGE_OUTBOX_SQL = """
            CREATE TABLE IF NOT EXISTS package_outbox (
                package_id TEXT PRIMARY KEY,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                exported_at TEXT
            )
            """
# ``send_consent_windows`` records consent as explicit intervals rather than a moving
# day-stamp: opened when send consent is observed, heartbeat-confirmed on every later
# observation, closed at the LAST CONFIRMED moment (never "now") on withdrawal. Additive
# like the send columns; old readers never touch these tables.
_CREATE_CONSENT_TABLES_SQL = (
    """
            CREATE TABLE IF NOT EXISTS send_consent_windows (
                opened_at TEXT NOT NULL,
                last_confirmed_at TEXT NOT NULL,
                closed_at TEXT
            )
            """,
    """
            CREATE TABLE IF NOT EXISTS consent_marks (
                name TEXT PRIMARY KEY CHECK (name IN ('obs', 'data')),
                stamp TEXT NOT NULL
            )
            """,
)
# v1 -> v2: the resource columns were added to the counter primary key.
_MIGRATE_V1_SQL = (
    "ALTER TABLE counter_aggregates RENAME TO counter_aggregates_v1",
    _CREATE_COUNTER_AGGREGATES_SQL,
    """
            INSERT INTO counter_aggregates(
                period_start, metric_name, hermes_version, os_family, architecture,
                install_method, dimensions_json, value, packaged_value
            )
            SELECT
                period_start, metric_name, hermes_version, 'unknown', 'unknown', 'unknown',
                dimensions_json, value, packaged_value
            FROM counter_aggregates_v1
            """,
    "DROP TABLE counter_aggregates_v1",
)
_PENDING_PERIOD_COUNT_SQL = """
                SELECT COUNT(*) AS period_count FROM (
                    SELECT period_start, hermes_version, os_family, architecture, install_method
                    FROM counter_aggregates WHERE value > packaged_value
                    GROUP BY period_start, hermes_version, os_family, architecture, install_method
                )
                """
_INCREMENT_COUNTER_SQL = """
            INSERT INTO counter_aggregates(
                period_start, metric_name, hermes_version, os_family, architecture,
                install_method, dimensions_json, value, packaged_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(
                period_start, metric_name, hermes_version, os_family, architecture,
                install_method, dimensions_json
            )
            DO UPDATE SET value = value + excluded.value
            """
# (column, declaration) added to package_outbox for transmission bookkeeping.
_SEND_COLUMNS = (
    # When the 202 was received. NULL = never acknowledged.
    ("sent_at", "TEXT"),
    # NULL/'pending' = eligible, 'sent' = done, 'rejected' = permanent 400.
    ("send_state", "TEXT"),
    ("send_attempts", "INTEGER NOT NULL DEFAULT 0"),
    # Earliest next attempt; enforces backoff across process restarts.
    ("next_attempt_at", "TEXT"),
    ("last_error", "TEXT"),
    # The install_id actually transmitted, frozen on the first attempt so retries stay
    # byte-identical; the body is recomputed from payload_json.
    ("sent_install_id", "TEXT"),
    # Rewritten on every claim. Settlement and the pre-POST revalidation are compare-and-set
    # on it, so a claimant whose lease lapsed loses authority the moment another process
    # reclaims (else a suspended sender double-POSTs).
    ("claim_token", "TEXT"),
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row_resource(row: sqlite3.Row) -> dict[str, str]:
    return {column: row[column] for column in _RESOURCE_COLUMNS}


def _ensure_private(path: Path, mode: int) -> None:
    if mode == 0o700:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
    else:
        path.touch(mode=mode, exist_ok=True)
    try:
        path.chmod(mode)
    except OSError:
        pass


class SharedMetricsStore:
    """Persist allowlisted counters and export immutable delta packages."""

    def __init__(
        self, database_path: Path | None = None, outbox_directory: Path | None = None
    ) -> None:
        root = get_hermes_home() / "telemetry" / "shared_metrics"
        self.database_path = database_path or root / "metrics.sqlite3"
        self.outbox_directory = outbox_directory or root / "outbox"
        _ensure_private(self.database_path.parent, 0o700)
        _ensure_private(self.outbox_directory, 0o700)
        _ensure_private(self.database_path, 0o600)
        self._ensure_schema()
        self._deferred: list[Callable[[sqlite3.Connection], None]] = []
        self._deferred_lock = threading.Lock()
        self._drain_at_exit = False

    def record_model_call(self, dimensions: dict[str, str], resource: dict[str, str]) -> None:
        """Increment the terminal model-call counter for the current UTC day."""
        self.record_counter(MODEL_ROUTE_METRIC, dimensions, resource)

    def record_client_active(self, resource: dict[str, str]) -> bool:
        """Record this install at most once in any rolling 24-hour window."""
        return self._record_once_per_interval(
            _ACTIVE_INSTALL_STATE_KEY, CLIENT_ACTIVE_METRIC, {}, resource
        )

    def record_install_snapshot(self, dimensions: dict[str, str], resource: dict[str, str]) -> bool:
        """Record the configuration snapshot at most once in any rolling 24-hour window."""
        return self._record_once_per_interval(
            _INSTALL_SNAPSHOT_STATE_KEY, INSTALL_SNAPSHOT_METRIC, dimensions, resource
        )

    def install_snapshot_due(self) -> bool:
        """Cheap read so the producer skips collecting a snapshot the latch would drop."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM telemetry_state WHERE key = ?", (_INSTALL_SNAPSHOT_STATE_KEY,)
            ).fetchone()
        last = self._parse_state_timestamp(row["value"]) if row is not None else None
        now = _utc_now()
        return last is None or last > now or now >= last + _ACTIVE_INSTALL_INTERVAL

    def _record_once_per_interval(
        self, state_key: str, metric_name: str, dimensions: dict[str, str],
        resource: dict[str, str],
    ) -> bool:
        """Transactional compare-and-set latch shared by concurrent Hermes processes."""
        self._validate_counter(metric_name, dimensions, resource)
        now = _utc_now()
        with self._write() as connection:
            row = connection.execute(
                "SELECT value FROM telemetry_state WHERE key = ?", (state_key,)
            ).fetchone()
            last_recorded = self._parse_state_timestamp(row["value"]) if row is not None else None
            if last_recorded is not None and last_recorded > now:
                # A wall-clock correction must not suppress activity until the stale future
                # timestamp plus another full interval.
                connection.execute(
                    "UPDATE telemetry_state SET value = ? WHERE key = ?",
                    (_isoformat(now), state_key),
                )
                return False
            if last_recorded is not None and now < last_recorded + _ACTIVE_INSTALL_INTERVAL:
                return False
            self._install_id(connection)
            self._record_counter_in_transaction(
                connection, metric_name, dimensions, resource, now.date().isoformat()
            )
            connection.execute(
                "INSERT INTO telemetry_state(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (state_key, _isoformat(now)),
            )
        return True

    def record_counter(
        self, metric_name: str, dimensions: dict[str, str], resource: dict[str, str],
        amount: int = 1,
    ) -> None:
        """Add ``amount`` (1 for event counters, a token count for sums) for the current UTC day."""
        self._validate_counter(metric_name, dimensions, resource)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1 or (
            amount != 1 and metric_name not in SUM_METRICS
        ):
            raise ValueError(f"Unsupported amount for shared metric: {metric_name}")
        period_start = _utc_now().date().isoformat()
        self._write_or_defer(lambda connection: self._record_counter_in_transaction(
            connection, metric_name, dimensions, resource, period_start, amount
        ))

    def update_rollup_state(
        self, state_key: str,
        update: Callable[[dict[str, Any] | None], tuple[dict[str, Any], list[tuple[str, dict, dict, str]]]],
    ) -> None:
        """Read-modify-write one JSON rollup state and record the ``(metric, dimensions, resource,
        period_start)`` rows ``update`` closes, in one write transaction: every process of the profile
        serializes here, so a closed period is emitted exactly once. Rows the contract rejects are
        dropped (never retried) so one bad row cannot wedge the rollup. A busy store defers the update
        to the next write (it then runs with that write's clock)."""
        self._write_or_defer(lambda connection: self._update_rollup_state(connection, state_key, update))

    def _update_rollup_state(
        self, connection: sqlite3.Connection, state_key: str,
        update: Callable[[dict[str, Any] | None], tuple[dict[str, Any], list[tuple[str, dict, dict, str]]]],
    ) -> None:
        row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = ?", (state_key,)
        ).fetchone()
        try:
            state = json.loads(row["value"]) if row is not None else None
        except (TypeError, ValueError):
            state = None
        new_state, rows = update(state if isinstance(state, dict) else None)
        valid = [
            r for r in rows
            if counter_dimensions_are_valid(r[0], r[1]) and client_resource_is_valid(r[2])
        ]
        if valid:
            self._install_id(connection)
        for metric_name, dimensions, resource, period_start in valid:
            self._record_counter_in_transaction(connection, metric_name, dimensions, resource, period_start)
        connection.execute(
            "INSERT INTO telemetry_state(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (state_key, _compact_json(new_state)),
        )

    def recorded_milestones(self) -> frozenset[str]:
        return self._latched(_MILESTONE_STATE_PREFIX)

    def record_milestone(self, milestone: str, install_age_bucket: str, resource: dict[str, str]) -> bool:
        """Record an install milestone at most once, ever (compare-and-set across processes)."""
        dimensions = {"install_age_bucket": install_age_bucket, "milestone": milestone}
        return self._record_once_ever(f"{_MILESTONE_STATE_PREFIX}{milestone}", MILESTONE_METRIC, dimensions, resource)

    # ---- v5 signals ----
    def recorded_features(self) -> frozenset[str]:
        return self._latched(_FEATURE_STATE_PREFIX)

    def record_feature_adoption(self, feature: str, days_bucket: str, resource: dict[str, str]) -> bool:
        """Record a feature's first use at most once per install, ever (same latch as milestones)."""
        dimensions = {"days_since_install_bucket": days_bucket, "feature": feature}
        return self._record_once_ever(f"{_FEATURE_STATE_PREFIX}{feature}", FEATURE_ADOPTION_METRIC, dimensions, resource)

    def record_counter_once_per_day(self, metric_name: str, dimensions: dict[str, str], resource: dict[str, str]) -> bool:
        """Count at most once per UTC day per dimension set, whatever the surface (``surface`` is
        attribution of the first one that day). One state row per dimension set, overwritten daily."""
        self._validate_counter(metric_name, dimensions, resource)
        now = _utc_now()
        today = now.date().isoformat()
        key = f"{_DAILY_ONCE_STATE_PREFIX}{metric_name}:" + _compact_json(
            {k: v for k, v in dimensions.items() if k != "surface"})
        with self._write() as connection:
            changed = connection.execute(
                "INSERT INTO telemetry_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE"
                " SET value = excluded.value WHERE telemetry_state.value != excluded.value",
                (key, today),
            ).rowcount
            if not changed:
                return False
            self._install_id(connection)
            self._record_counter_in_transaction(connection, metric_name, dimensions, resource, today)
        return True
    # ---- end v5 signals ----

    def _latched(self, prefix: str) -> frozenset[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT key FROM telemetry_state WHERE key LIKE ?", (f"{prefix}%",)
            ).fetchall()
        return frozenset(row["key"][len(prefix):] for row in rows)

    def _record_once_ever(
        self, state_key: str, metric_name: str, dimensions: dict[str, str], resource: dict[str, str],
    ) -> bool:
        self._validate_counter(metric_name, dimensions, resource)
        now = _utc_now()
        with self._write() as connection:
            inserted = connection.execute(
                "INSERT INTO telemetry_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO NOTHING",
                (state_key, _isoformat(now)),
            ).rowcount
            if not inserted:
                return False
            self._install_id(connection)
            self._record_counter_in_transaction(
                connection, metric_name, dimensions, resource, now.date().isoformat()
            )
        return True

    @staticmethod
    def _validate_counter(
        metric_name: str, dimensions: dict[str, str], resource: dict[str, str]
    ) -> None:
        if metric_name not in COUNTER_METRICS:
            raise ValueError(f"Unsupported shared metric: {metric_name}")
        if not counter_dimensions_are_valid(metric_name, dimensions):
            raise ValueError(f"Unsupported dimensions for shared metric: {metric_name}")
        if not client_resource_is_valid(resource):
            raise ValueError("Unsupported shared-metrics client resource")

    @staticmethod
    def _record_counter_in_transaction(
        connection: sqlite3.Connection,
        metric_name: str,
        dimensions: dict[str, str],
        resource: dict[str, str],
        period_start: str,
        amount: int = 1,
    ) -> None:
        connection.execute(
            _INCREMENT_COUNTER_SQL,
            (
                period_start, metric_name, *(resource[column] for column in _RESOURCE_COLUMNS),
                _compact_json(dimensions), amount,
            ),
        )

    def create_and_export_package(self) -> list[Path]:
        """Commit every pending delta package (one per period), then export the outbox."""
        with self._connection() as connection:
            row = connection.execute(_PENDING_PERIOD_COUNT_SQL).fetchone()
        for _ in range(int(row["period_count"]) if row is not None else 0):
            if self._create_package() is None:
                break
        return self._export_and_prune()

    def _write_or_defer(self, write: Callable[[sqlite3.Connection], None]) -> None:
        """Run ``write`` (after any deferred ones) in one write transaction. A store another writer
        holds past the short busy timeout keeps them in memory for the next write or process exit
        instead of losing them or stalling the caller."""
        with self._deferred_lock:
            pending, self._deferred = [*self._deferred, write], []
        try:
            with self._write() as connection:
                for item in pending:
                    item(connection)
        except sqlite3.OperationalError as exc:
            busy = "locked" in str(exc) or "busy" in str(exc)
            with self._deferred_lock:
                # A failure that is not contention drops only this write: the earlier ones stay queued.
                self._deferred = [*(pending if busy else pending[:-1]), *self._deferred][-_DEFERRED_WRITES_MAX:]
                register, self._drain_at_exit = not self._drain_at_exit, True
            if register:
                atexit.register(self.drain_deferred)
            if not busy:
                raise

    def drain_deferred(self, busy_timeout_ms: int = _SCHEMA_BUSY_TIMEOUT_MS) -> None:
        """Write the deferred writes (process exit / export waits longer than the user's path does)."""
        with self._deferred_lock:
            pending, self._deferred = self._deferred, []
        if not pending:
            return
        try:
            with self._write(busy_timeout_ms=busy_timeout_ms) as connection:
                for item in pending:
                    item(connection)
        except Exception:
            logger.warning("Unable to write %d deferred shared-metrics updates", len(pending), exc_info=True)

    def create_and_export_package_if_due(self) -> list[Path]:
        """Create pending packages at most once per UTC day, then export them."""
        self.drain_deferred(_BUSY_TIMEOUT_MS)
        self._create_pending_packages_if_due()
        return self._export_and_prune()

    def _export_and_prune(self) -> list[Path]:
        exported = self._export_pending_packages()
        try:
            self._prune_expired_history()
        except Exception:
            logger.warning("Unable to prune expired shared-metrics history", exc_info=True)
        return exported

    def counter_snapshot(self) -> list[dict[str, Any]]:
        """Return cumulative counters for focused tests and local inspection."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT period_start, metric_name, hermes_version, os_family, architecture,
                    install_method, dimensions_json, value, packaged_value
                FROM counter_aggregates
                ORDER BY period_start, hermes_version, os_family, architecture, install_method,
                    metric_name, dimensions_json
                """
            ).fetchall()
        return [
            {
                "period_start": row["period_start"], "metric_name": row["metric_name"],
                "resource": _row_resource(row), "dimensions": json.loads(row["dimensions_json"]),
                "value": row["value"], "packaged_value": row["packaged_value"],
            }
            for row in rows
        ]

    @contextmanager
    def _connection(
        self, *, busy_timeout_ms: int = _BUSY_TIMEOUT_MS
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=busy_timeout_ms / 1000)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self, *, busy_timeout_ms: int = _BUSY_TIMEOUT_MS) -> Iterator[sqlite3.Connection]:
        """A connection inside one write transaction."""
        with self._connection(busy_timeout_ms=busy_timeout_ms) as connection:
            with write_txn(connection):
                yield connection

    def _ensure_schema(self) -> None:
        # Serialize first-run creation and upgrades across Hermes processes.
        with self._write(busy_timeout_ms=_SCHEMA_BUSY_TIMEOUT_MS) as connection:
            connection.execute(_CREATE_TELEMETRY_STATE_SQL)
            schema_row = connection.execute(
                "SELECT value FROM telemetry_state WHERE key = 'schema_version'"
            ).fetchone()
            schema_version = str(schema_row["value"]) if schema_row is not None else None
            if schema_version == "1":
                for statement in _MIGRATE_V1_SQL:
                    connection.execute(statement)
                schema_version = _STORE_SCHEMA_VERSION
            if schema_version is not None and schema_version != _STORE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported shared-metrics store schema version: {schema_version}"
                )
            connection.execute(_CREATE_COUNTER_AGGREGATES_SQL)
            connection.execute(_CREATE_PACKAGE_OUTBOX_SQL)
            # Send bookkeeping columns are ADDITIVE and nullable; the schema version is
            # deliberately NOT bumped because the check above raises on unknown versions,
            # so a bump would hard-fail an older Hermes (second profile, rollback) sharing
            # the database. Old readers select named columns, never ``SELECT *``.
            existing = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(package_outbox)")
            }
            for column, declaration in _SEND_COLUMNS:
                if column not in existing:
                    add_column_if_missing(connection, "package_outbox", column, f"{column} {declaration}")
            for statement in _CREATE_CONSENT_TABLES_SQL:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO telemetry_state(key, value) VALUES ('schema_version', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_STORE_SCHEMA_VERSION,),
            )

    def _install_id(self, connection: sqlite3.Connection) -> str:
        query = "SELECT value FROM telemetry_state WHERE key = 'install_id'"
        row = connection.execute(query).fetchone()
        if row is None:
            connection.execute(
                "INSERT OR IGNORE INTO telemetry_state(key, value) VALUES ('install_id', ?)",
                (str(uuid.uuid4()),),
            )
            row = connection.execute(query).fetchone()
        if row is None:
            raise RuntimeError("Unable to create the shared-metrics install identity")
        return str(row["value"])

    @staticmethod
    def _parse_state_timestamp(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        return None if parsed.tzinfo is None else parsed.astimezone(UTC)

    def _create_pending_packages_if_due(self) -> None:
        now = _utc_now()
        with self._write() as connection:
            # Gate on the committed package, not its file write, so a failed
            # outbox export can be retried without packaging deltas twice.
            package_created_today = connection.execute(
                "SELECT 1 FROM package_outbox WHERE substr(created_at, 1, 10) >= ? LIMIT 1",
                (now.date().isoformat(),),
            ).fetchone()
            if package_created_today is None:
                while self._create_package_in_transaction(connection, now) is not None:
                    pass

    def _create_package(self) -> dict[str, Any] | None:
        with self._write() as connection:
            return self._create_package_in_transaction(connection, _utc_now())

    def _create_package_in_transaction(
        self, connection: sqlite3.Connection, now: datetime
    ) -> dict[str, Any] | None:
        period_row = connection.execute(
            """
                SELECT period_start, hermes_version, os_family, architecture, install_method
                FROM counter_aggregates WHERE value > packaged_value
                ORDER BY period_start, hermes_version, os_family, architecture, install_method
                LIMIT 1
                """
        ).fetchone()
        if period_row is None or not period_row["period_start"]:
            return None
        period_value = period_row["period_start"]

        resource = _row_resource(period_row)
        resource_values = tuple(resource[column] for column in _RESOURCE_COLUMNS)
        rows = connection.execute(
            """
                SELECT metric_name, dimensions_json, value, packaged_value FROM counter_aggregates
                WHERE period_start = ? AND hermes_version = ? AND os_family = ?
                  AND architecture = ? AND install_method = ? AND value > packaged_value
                ORDER BY metric_name, dimensions_json
                """,
            (period_value, *resource_values),
        ).fetchall()
        period_start = datetime.fromisoformat(str(period_value)).replace(tzinfo=UTC)
        if not client_resource_is_valid(resource):
            raise ValueError("Unsupported shared-metrics client resource")
        payload = {
            "schema_version": _PACKAGE_SCHEMA_VERSION,
            "package_id": str(uuid.uuid4()),
            "install_id": self._install_id(connection),
            "period_start": _isoformat(period_start),
            "period_end": _isoformat(period_start + timedelta(days=1)),
            "generated_at": _isoformat(now),
            "resource": resource,
            "metrics": [self._package_metric(row) for row in rows],
        }
        connection.execute(
            "INSERT INTO package_outbox( package_id, period_start, period_end, payload_json,"
            " created_at ) VALUES (?, ?, ?, ?, ?)",
            (
                payload["package_id"], payload["period_start"], payload["period_end"],
                _compact_json(payload), payload["generated_at"],
            ),
        )
        # Advance the data high-water mark. This is the ONLY writer of the 'data' mark:
        # it clamps consent-window opens so a rolled-back clock can never open a window
        # underneath packages that already exist.
        connection.execute(
            "INSERT INTO consent_marks(name, stamp) VALUES ('data', ?)"
            " ON CONFLICT(name) DO UPDATE SET stamp = MAX(stamp, excluded.stamp)",
            (payload["period_end"],),
        )
        for row in rows:
            connection.execute(
                """
                    UPDATE counter_aggregates SET packaged_value = value
                    WHERE period_start = ? AND metric_name = ? AND hermes_version = ?
                      AND os_family = ? AND architecture = ? AND install_method = ?
                      AND dimensions_json = ?
                    """,
                (period_value, row["metric_name"], *resource_values, row["dimensions_json"]),
            )
        return payload

    @staticmethod
    def _package_metric(row: sqlite3.Row) -> dict[str, Any]:
        metric_name = str(row["metric_name"])
        dimensions = json.loads(row["dimensions_json"])
        if not isinstance(dimensions, dict) or not counter_dimensions_are_valid(
            metric_name, dimensions
        ):
            raise ValueError(f"Unsupported dimensions for shared metric: {metric_name}")
        return {
            "name": metric_name, "type": "counter", "dimensions": dimensions,
            "value": row["value"] - row["packaged_value"],
        }

    def _export_pending_packages(self) -> list[Path]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT package_id, payload_json FROM package_outbox"
                " WHERE exported_at IS NULL ORDER BY created_at, package_id"
            ).fetchall()

        exported: list[Path] = []
        for row in rows:
            package_id = str(row["package_id"])
            path = self.outbox_directory / f"{package_id}.json"
            atomic_json_write(
                path, json.loads(row["payload_json"]), indent=None, separators=(",", ":"),
                sort_keys=True, mode=0o600,
            )
            with self._connection() as connection:
                connection.execute(
                    "UPDATE package_outbox SET exported_at = ?"
                    " WHERE package_id = ? AND exported_at IS NULL",
                    (_isoformat(_utc_now()), package_id),
                )
            exported.append(path)
        return exported

    def _prune_expired_history(self, *, now: datetime | None = None) -> None:
        """Remove exported local history after the bounded retention window."""
        cutoff = (now or _utc_now()) - timedelta(days=_LOCAL_HISTORY_RETENTION_DAYS)
        cutoff_timestamp = _isoformat(cutoff)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT package_id FROM package_outbox"
                " WHERE exported_at IS NOT NULL AND exported_at < ?"
                " ORDER BY exported_at, package_id",
                (cutoff_timestamp,),
            ).fetchall()

        removable_package_ids: list[str] = []
        for row in rows:
            package_id = str(row["package_id"])
            try:
                (self.outbox_directory / f"{package_id}.json").unlink(missing_ok=True)
                removable_package_ids.append(package_id)
            except OSError:
                logger.warning(
                    "Unable to prune expired shared-metrics package %s", package_id, exc_info=True
                )

        with self._write() as connection:
            # A package the ingest accepted or refused is never sent again, and its outbox file is
            # the local history copy, so the database drops its duplicate of the body.
            connection.execute(
                "UPDATE package_outbox SET payload_json = ''"
                " WHERE exported_at IS NOT NULL AND send_state IN ('sent', 'rejected')"
                " AND payload_json != ''"
            )
            for package_id in removable_package_ids:
                connection.execute(
                    "DELETE FROM package_outbox"
                    " WHERE package_id = ? AND exported_at IS NOT NULL AND exported_at < ?",
                    (package_id, cutoff_timestamp),
                )
            connection.execute(
                """
                    DELETE FROM counter_aggregates
                    WHERE period_start < ? AND value = packaged_value AND NOT EXISTS (
                        SELECT 1 FROM package_outbox WHERE exported_at IS NULL
                          AND substr(package_outbox.period_start, 1, 10)
                              = counter_aggregates.period_start
                    )
                    """,
                (cutoff.date().isoformat(),),
            )
