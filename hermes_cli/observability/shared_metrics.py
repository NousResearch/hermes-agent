"""Durable aggregation and local export for Hermes shared metrics."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_cli.sqlite_util import write_txn
from hermes_constants import get_hermes_home
from utils import atomic_json_write

from .shared_metrics_contract import (
    CLIENT_ACTIVE_METRIC,
    COUNTER_METRICS,
    MODEL_ROUTE_METRIC,
    client_resource_is_valid,
    counter_dimensions_are_valid,
)


_PACKAGE_SCHEMA_VERSION = "hermes.shared_metrics.v2"
_STORE_SCHEMA_VERSION = "2"
_BUSY_TIMEOUT_MS = 250
_SCHEMA_BUSY_TIMEOUT_MS = 5_000
_LOCAL_HISTORY_RETENTION_DAYS = 30
_ACTIVE_INSTALL_STATE_KEY = "client_active_recorded_at"
_ACTIVE_INSTALL_INTERVAL = timedelta(hours=24)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class SharedMetricsStore:
    """Persist allowlisted counters and export immutable delta packages."""

    def __init__(
        self,
        database_path: Path | None = None,
        outbox_directory: Path | None = None,
    ) -> None:
        root = get_hermes_home() / "telemetry" / "shared_metrics"
        self.database_path = database_path or root / "metrics.sqlite3"
        self.outbox_directory = outbox_directory or root / "outbox"
        self._ensure_private_directory(self.database_path.parent)
        self._ensure_private_directory(self.outbox_directory)
        self._ensure_private_file(self.database_path)
        self._ensure_schema()

    def record_model_call(
        self,
        dimensions: dict[str, str],
        resource: dict[str, str],
    ) -> None:
        """Increment the terminal model-call counter for the current UTC day."""
        self.record_counter(MODEL_ROUTE_METRIC, dimensions, resource)

    def record_client_active(self, resource: dict[str, str]) -> bool:
        """Record this install at most once in any rolling 24-hour window."""
        dimensions: dict[str, str] = {}
        self._validate_counter(CLIENT_ACTIVE_METRIC, dimensions, resource)
        now = _utc_now()
        with self._connection() as connection:
            with write_txn(connection):
                row = connection.execute(
                    "SELECT value FROM telemetry_state WHERE key = ?",
                    (_ACTIVE_INSTALL_STATE_KEY,),
                ).fetchone()
                if row is not None:
                    last_recorded = self._parse_state_timestamp(row["value"])
                    if last_recorded is not None and last_recorded > now:
                        # A wall-clock correction must not suppress activity until
                        # the stale future timestamp plus another full interval.
                        connection.execute(
                            """
                            UPDATE telemetry_state
                            SET value = ?
                            WHERE key = ?
                            """,
                            (_isoformat(now), _ACTIVE_INSTALL_STATE_KEY),
                        )
                        return False
                    if (
                        last_recorded is not None
                        and now < last_recorded + _ACTIVE_INSTALL_INTERVAL
                    ):
                        return False

                self._install_id(connection)
                self._record_counter_in_transaction(
                    connection,
                    CLIENT_ACTIVE_METRIC,
                    dimensions,
                    resource,
                    period_start=now.date().isoformat(),
                )
                connection.execute(
                    """
                    INSERT INTO telemetry_state(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (_ACTIVE_INSTALL_STATE_KEY, _isoformat(now)),
                )
        return True

    def record_counter(
        self,
        metric_name: str,
        dimensions: dict[str, str],
        resource: dict[str, str],
    ) -> None:
        """Increment one allowlisted counter for the current UTC day."""
        self._validate_counter(metric_name, dimensions, resource)
        with self._connection() as connection:
            self._record_counter_in_transaction(
                connection,
                metric_name,
                dimensions,
                resource,
                period_start=_utc_now().date().isoformat(),
            )

    @staticmethod
    def _validate_counter(
        metric_name: str,
        dimensions: dict[str, str],
        resource: dict[str, str],
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
        *,
        period_start: str,
    ) -> None:
        dimensions_json = json.dumps(
            dimensions,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            """
            INSERT INTO counter_aggregates(
                period_start,
                metric_name,
                hermes_version,
                os_family,
                architecture,
                install_method,
                dimensions_json,
                value,
                packaged_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0)
            ON CONFLICT(
                period_start,
                metric_name,
                hermes_version,
                os_family,
                architecture,
                install_method,
                dimensions_json
            )
            DO UPDATE SET value = value + 1
            """,
            (
                period_start,
                metric_name,
                resource["hermes_version"],
                resource["os_family"],
                resource["architecture"],
                resource["install_method"],
                dimensions_json,
            ),
        )

    def create_and_export_package(self) -> list[Path]:
        """Commit one pending delta package, then atomically export the outbox."""
        pending_periods = self._pending_period_count()
        for _ in range(pending_periods):
            if self._create_package() is None:
                break
        return self._export_and_prune()

    def create_and_export_package_if_due(self) -> list[Path]:
        """Create pending packages at most once per UTC day, then export them."""
        self._create_pending_packages_if_due()
        return self._export_and_prune()

    def _export_and_prune(self) -> list[Path]:
        exported = self._export_pending_packages()
        try:
            self._prune_expired_history()
        except Exception:
            logger.warning(
                "Unable to prune expired shared-metrics history",
                exc_info=True,
            )
        return exported

    def counter_snapshot(self) -> list[dict[str, Any]]:
        """Return cumulative counters for focused tests and local inspection."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    period_start,
                    metric_name,
                    hermes_version,
                    os_family,
                    architecture,
                    install_method,
                    dimensions_json,
                    value,
                    packaged_value
                FROM counter_aggregates
                ORDER BY
                    period_start,
                    hermes_version,
                    os_family,
                    architecture,
                    install_method,
                    metric_name,
                    dimensions_json
                """
            ).fetchall()
        return [
            {
                "period_start": row["period_start"],
                "metric_name": row["metric_name"],
                "resource": {
                    "hermes_version": row["hermes_version"],
                    "os_family": row["os_family"],
                    "architecture": row["architecture"],
                    "install_method": row["install_method"],
                },
                "dimensions": json.loads(row["dimensions_json"]),
                "value": row["value"],
                "packaged_value": row["packaged_value"],
            }
            for row in rows
        ]

    @contextmanager
    def _connection(
        self,
        *,
        busy_timeout_ms: int = _BUSY_TIMEOUT_MS,
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=busy_timeout_ms / 1000,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _ensure_private_file(path: Path) -> None:
        path.touch(mode=0o600, exist_ok=True)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _ensure_schema(self) -> None:
        with self._connection(busy_timeout_ms=_SCHEMA_BUSY_TIMEOUT_MS) as connection:
            # Serialize first-run creation and upgrades across Hermes processes.
            with write_txn(connection):
                self._ensure_schema_in_transaction(connection)

    @staticmethod
    def _ensure_schema_in_transaction(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        schema_row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = 'schema_version'"
        ).fetchone()
        schema_version = str(schema_row["value"]) if schema_row is not None else None
        if schema_version == "1":
            SharedMetricsStore._migrate_v1_counter_aggregates(connection)
            schema_version = _STORE_SCHEMA_VERSION
        if schema_version is not None and schema_version != _STORE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported shared-metrics store schema version: {schema_version}"
            )
        SharedMetricsStore._create_counter_aggregates_table(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS package_outbox (
                package_id TEXT PRIMARY KEY,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                exported_at TEXT
            )
            """
        )
        SharedMetricsStore._add_send_columns(connection)
        SharedMetricsStore._add_consent_tables(connection)
        connection.execute(
            """
            INSERT INTO telemetry_state(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_STORE_SCHEMA_VERSION,),
        )

    @staticmethod
    def _add_send_columns(connection: sqlite3.Connection) -> None:
        """Add transmission bookkeeping to ``package_outbox``, idempotently.

        These columns are ADDITIVE and nullable, and the store schema version
        is deliberately NOT bumped. ``_ensure_schema_in_transaction`` raises on
        any version it does not recognise and has no forward-compatibility
        branch, so bumping would make an older Hermes — a second profile on an
        older build, or a rollback — hard-fail against the same database file.
        Old readers select named columns and never ``SELECT *``, so extra
        columns are invisible to them.
        """
        existing = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(package_outbox)")
        }
        for column, declaration in (
            # When the 202 was received. NULL = never acknowledged.
            ("sent_at", "TEXT"),
            # NULL/'pending' = eligible, 'sent' = done, 'rejected' = permanent 400.
            ("send_state", "TEXT"),
            ("send_attempts", "INTEGER NOT NULL DEFAULT 0"),
            # Earliest next attempt; enforces backoff across process restarts.
            ("next_attempt_at", "TEXT"),
            ("last_error", "TEXT"),
            # The identifier actually transmitted, frozen on the first
            # attempt so retries stay byte-identical. Since the 2026-08-27
            # product decision this is the stable install_id itself.
            # Only the ~36-byte id is stored: the body is recomputed from
            # payload_json, whose serialisation is deterministic.
            ("sent_install_id", "TEXT"),
            # NULL until first claimed; rewritten on every claim. Settlement
            # and the pre-POST revalidation are compare-and-set on this, so a
            # claimant whose lease lapsed loses authority the moment another
            # process reclaims (PR-review finding: without it, a suspended
            # sender resuming after a reclaim double-POSTs the package).
            ("claim_token", "TEXT"),
        ):
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE package_outbox ADD COLUMN {column} {declaration}"
                )

    @staticmethod
    def _add_consent_tables(connection: sqlite3.Connection) -> None:
        """Create the consent-window tables, idempotently.

        Additive like ``_add_send_columns`` — the schema version is
        deliberately NOT bumped, and old readers never touch these tables.

        ``send_consent_windows`` records consent as explicit intervals rather
        than a moving day-stamp: a window is opened when send consent is
        observed, heartbeat-confirmed on every later observation, and closed
        at the LAST CONFIRMED moment (never "now") when consent is observed
        withdrawn. Consent is asserted only for time that was actually
        observed, so unobserved gaps — a hand-edited config with no process
        running — fail closed by construction.

        ``consent_marks`` holds two monotonic high-water marks with strictly
        separated roles:

        - ``obs``: the latest observation stamp ever seen. Advanced only by
          the reconciler. Confirms consent and clamps window closes.
        - ``data``: the latest package ``period_end`` ever stored. Advanced
          only by the package writer. Clamps window OPENS, so a rolled-back
          clock can never open a window underneath packages that already
          exist on disk.

        The separation is load-bearing: letting data stamps confirm consent
        re-created a refused-window leak (packages stored during an off
        window would vouch for it), and letting observation stamps clamp
        opens is not enough on its own to stop a rollback sliding a window
        under existing refused data.
        """
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS send_consent_windows (
                opened_at TEXT NOT NULL,
                last_confirmed_at TEXT NOT NULL,
                closed_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consent_marks (
                name TEXT PRIMARY KEY CHECK (name IN ('obs', 'data')),
                stamp TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _create_counter_aggregates_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
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
                    period_start,
                    metric_name,
                    hermes_version,
                    os_family,
                    architecture,
                    install_method,
                    dimensions_json
                )
            )
            """
        )

    @staticmethod
    def _migrate_v1_counter_aggregates(connection: sqlite3.Connection) -> None:
        connection.execute(
            "ALTER TABLE counter_aggregates RENAME TO counter_aggregates_v1"
        )
        SharedMetricsStore._create_counter_aggregates_table(connection)
        connection.execute(
            """
            INSERT INTO counter_aggregates(
                period_start,
                metric_name,
                hermes_version,
                os_family,
                architecture,
                install_method,
                dimensions_json,
                value,
                packaged_value
            )
            SELECT
                period_start,
                metric_name,
                hermes_version,
                'unknown',
                'unknown',
                'unknown',
                dimensions_json,
                value,
                packaged_value
            FROM counter_aggregates_v1
            """
        )
        connection.execute("DROP TABLE counter_aggregates_v1")

    def _install_id(self, connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = 'install_id'"
        ).fetchone()
        if row is not None:
            return str(row["value"])
        candidate = str(uuid.uuid4())
        connection.execute(
            "INSERT OR IGNORE INTO telemetry_state(key, value) VALUES ('install_id', ?)",
            (candidate,),
        )
        row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = 'install_id'"
        ).fetchone()
        if row is None:
            raise RuntimeError("Unable to create the shared-metrics install identity")
        return str(row["value"])

    @staticmethod
    def _parse_state_timestamp(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _pending_period_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS period_count
                FROM (
                    SELECT
                        period_start,
                        hermes_version,
                        os_family,
                        architecture,
                        install_method
                    FROM counter_aggregates
                    WHERE value > packaged_value
                    GROUP BY
                        period_start,
                        hermes_version,
                        os_family,
                        architecture,
                        install_method
                )
                """
            ).fetchone()
        return int(row["period_count"]) if row is not None else 0

    def _create_pending_packages_if_due(self) -> None:
        now = _utc_now()
        with self._connection() as connection:
            with write_txn(connection):
                # Gate on the committed package, not its file write, so a failed
                # outbox export can be retried without packaging deltas twice.
                package_created_today = connection.execute(
                    """
                    SELECT 1
                    FROM package_outbox
                    WHERE substr(created_at, 1, 10) >= ?
                    LIMIT 1
                    """,
                    (now.date().isoformat(),),
                ).fetchone()
                if package_created_today is not None:
                    return
                while self._create_package_in_transaction(connection, now) is not None:
                    pass

    def _create_package(self) -> dict[str, Any] | None:
        now = _utc_now()
        with self._connection() as connection:
            with write_txn(connection):
                return self._create_package_in_transaction(connection, now)

    def _create_package_in_transaction(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> dict[str, Any] | None:
        period_row = connection.execute(
            """
                SELECT
                    period_start,
                    hermes_version,
                    os_family,
                    architecture,
                    install_method
                FROM counter_aggregates
                WHERE value > packaged_value
                ORDER BY
                    period_start,
                    hermes_version,
                    os_family,
                    architecture,
                    install_method
                LIMIT 1
                """
        ).fetchone()
        period_value = period_row["period_start"] if period_row is not None else None
        if not period_value:
            return None

        rows = connection.execute(
            """
                SELECT metric_name, dimensions_json, value, packaged_value
                FROM counter_aggregates
                WHERE period_start = ?
                  AND hermes_version = ?
                  AND os_family = ?
                  AND architecture = ?
                  AND install_method = ?
                  AND value > packaged_value
                ORDER BY metric_name, dimensions_json
                """,
            (
                period_value,
                period_row["hermes_version"],
                period_row["os_family"],
                period_row["architecture"],
                period_row["install_method"],
            ),
        ).fetchall()
        period_start = datetime.fromisoformat(str(period_value)).replace(
            tzinfo=timezone.utc
        )
        period_end = period_start + timedelta(days=1)
        package_id = str(uuid.uuid4())
        resource = {
            "hermes_version": period_row["hermes_version"],
            "os_family": period_row["os_family"],
            "architecture": period_row["architecture"],
            "install_method": period_row["install_method"],
        }
        if not client_resource_is_valid(resource):
            raise ValueError("Unsupported shared-metrics client resource")
        payload = {
            "schema_version": _PACKAGE_SCHEMA_VERSION,
            "package_id": package_id,
            "install_id": self._install_id(connection),
            "period_start": _isoformat(period_start),
            "period_end": _isoformat(period_end),
            "generated_at": _isoformat(now),
            "resource": resource,
            "metrics": [self._package_metric(row) for row in rows],
        }
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            """
                INSERT INTO package_outbox(
                    package_id,
                    period_start,
                    period_end,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
            (
                package_id,
                payload["period_start"],
                payload["period_end"],
                payload_json,
                payload["generated_at"],
            ),
        )
        # Advance the data high-water mark. This is the ONLY writer of the
        # 'data' mark: it clamps consent-window opens so a rolled-back clock
        # can never open a window underneath packages that already exist.
        connection.execute(
            """
            INSERT INTO consent_marks(name, stamp) VALUES ('data', ?)
            ON CONFLICT(name) DO UPDATE SET stamp = MAX(stamp, excluded.stamp)
            """,
            (payload["period_end"],),
        )
        for row in rows:
            connection.execute(
                """
                    UPDATE counter_aggregates
                    SET packaged_value = value
                    WHERE period_start = ?
                      AND metric_name = ?
                      AND hermes_version = ?
                      AND os_family = ?
                      AND architecture = ?
                      AND install_method = ?
                      AND dimensions_json = ?
                    """,
                (
                    period_value,
                    row["metric_name"],
                    period_row["hermes_version"],
                    period_row["os_family"],
                    period_row["architecture"],
                    period_row["install_method"],
                    row["dimensions_json"],
                ),
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
            "name": metric_name,
            "type": "counter",
            "dimensions": dimensions,
            "value": row["value"] - row["packaged_value"],
        }

    def _export_pending_packages(self) -> list[Path]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT package_id, payload_json
                FROM package_outbox
                WHERE exported_at IS NULL
                ORDER BY created_at, package_id
                """
            ).fetchall()

        exported: list[Path] = []
        for row in rows:
            package_id = str(row["package_id"])
            path = self.outbox_directory / f"{package_id}.json"
            atomic_json_write(
                path,
                json.loads(row["payload_json"]),
                indent=2,
                sort_keys=True,
                mode=0o600,
            )
            with self._connection() as connection:
                connection.execute(
                    """
                    UPDATE package_outbox
                    SET exported_at = ?
                    WHERE package_id = ? AND exported_at IS NULL
                    """,
                    (_isoformat(_utc_now()), package_id),
                )
            exported.append(path)
        return exported

    def _prune_expired_history(self, *, now: datetime | None = None) -> None:
        """Remove exported local history after the bounded retention window."""
        cutoff = (now or _utc_now()) - timedelta(
            days=_LOCAL_HISTORY_RETENTION_DAYS
        )
        cutoff_timestamp = _isoformat(cutoff)
        cutoff_period = cutoff.date().isoformat()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT package_id
                FROM package_outbox
                WHERE exported_at IS NOT NULL
                  AND exported_at < ?
                ORDER BY exported_at, package_id
                """,
                (cutoff_timestamp,),
            ).fetchall()

        removable_package_ids: list[str] = []
        for row in rows:
            package_id = str(row["package_id"])
            try:
                (self.outbox_directory / f"{package_id}.json").unlink(
                    missing_ok=True
                )
            except OSError:
                logger.warning(
                    "Unable to prune expired shared-metrics package %s",
                    package_id,
                    exc_info=True,
                )
                continue
            removable_package_ids.append(package_id)

        with self._connection() as connection:
            with write_txn(connection):
                for package_id in removable_package_ids:
                    connection.execute(
                        """
                        DELETE FROM package_outbox
                        WHERE package_id = ?
                          AND exported_at IS NOT NULL
                          AND exported_at < ?
                        """,
                        (package_id, cutoff_timestamp),
                    )
                connection.execute(
                    """
                    DELETE FROM counter_aggregates
                    WHERE period_start < ?
                      AND value = packaged_value
                      AND NOT EXISTS (
                          SELECT 1
                          FROM package_outbox
                          WHERE exported_at IS NULL
                            AND substr(package_outbox.period_start, 1, 10)
                                = counter_aggregates.period_start
                      )
                    """,
                    (cutoff_period,),
                )
