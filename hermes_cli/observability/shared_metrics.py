"""Durable aggregation and local export for Hermes shared metrics."""

from __future__ import annotations

import json
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
    COUNTER_METRICS,
    MODEL_CALL_METRIC,
    counter_dimensions_are_valid,
)


_PACKAGE_SCHEMA_VERSION = "hermes.shared_metrics.v1"
_STORE_SCHEMA_VERSION = "1"
_BUSY_TIMEOUT_MS = 250


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
        hermes_version: str,
    ) -> None:
        """Increment the terminal model-call counter for the current UTC day."""
        self.record_counter(MODEL_CALL_METRIC, dimensions, hermes_version)

    def record_counter(
        self,
        metric_name: str,
        dimensions: dict[str, str],
        hermes_version: str,
    ) -> None:
        """Increment one allowlisted counter for the current UTC day."""
        if metric_name not in COUNTER_METRICS:
            raise ValueError(f"Unsupported shared metric: {metric_name}")
        if not counter_dimensions_are_valid(metric_name, dimensions):
            raise ValueError(f"Unsupported dimensions for shared metric: {metric_name}")
        dimensions_json = json.dumps(
            dimensions,
            sort_keys=True,
            separators=(",", ":"),
        )
        period_start = _utc_now().date().isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO counter_aggregates(
                    period_start,
                    metric_name,
                    hermes_version,
                    dimensions_json,
                    value,
                    packaged_value
                ) VALUES (?, ?, ?, ?, 1, 0)
                ON CONFLICT(
                    period_start,
                    metric_name,
                    hermes_version,
                    dimensions_json
                )
                DO UPDATE SET value = value + 1
                """,
                (
                    period_start,
                    metric_name,
                    hermes_version or "unknown",
                    dimensions_json,
                ),
            )

    def create_and_export_package(self) -> list[Path]:
        """Commit one pending delta package, then atomically export the outbox."""
        pending_periods = self._pending_period_count()
        for _ in range(pending_periods):
            if self._create_package() is None:
                break
        return self._export_pending_packages()

    def counter_snapshot(self) -> list[dict[str, Any]]:
        """Return cumulative counters for focused tests and local inspection."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    period_start,
                    metric_name,
                    hermes_version,
                    dimensions_json,
                    value,
                    packaged_value
                FROM counter_aggregates
                ORDER BY period_start, hermes_version, metric_name, dimensions_json
                """
            ).fetchall()
        return [
            {
                "period_start": row["period_start"],
                "metric_name": row["metric_name"],
                "hermes_version": row["hermes_version"],
                "dimensions": json.loads(row["dimensions_json"]),
                "value": row["value"],
                "packaged_value": row["packaged_value"],
            }
            for row in rows
        ]

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
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
        with self._connection() as connection:
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
        if schema_row is not None and str(schema_row["value"]) != _STORE_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported shared-metrics store schema version: "
                f"{schema_row['value']}"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS counter_aggregates (
                period_start TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                hermes_version TEXT NOT NULL,
                dimensions_json TEXT NOT NULL,
                value INTEGER NOT NULL,
                packaged_value INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    period_start,
                    metric_name,
                    hermes_version,
                    dimensions_json
                )
            )
            """
        )
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
        connection.execute(
            """
            INSERT OR IGNORE INTO telemetry_state(key, value)
            VALUES ('schema_version', ?)
            """,
            (_STORE_SCHEMA_VERSION,),
        )

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

    def _pending_period_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS period_count
                FROM (
                    SELECT period_start, hermes_version
                    FROM counter_aggregates
                    WHERE value > packaged_value
                    GROUP BY period_start, hermes_version
                )
                """
            ).fetchone()
        return int(row["period_count"]) if row is not None else 0

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
                SELECT period_start, hermes_version
                FROM counter_aggregates
                WHERE value > packaged_value
                ORDER BY period_start, hermes_version
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
                  AND value > packaged_value
                ORDER BY metric_name, dimensions_json
                """,
            (period_value, period_row["hermes_version"]),
        ).fetchall()
        period_start = datetime.fromisoformat(str(period_value)).replace(
            tzinfo=timezone.utc
        )
        period_end = period_start + timedelta(days=1)
        package_id = str(uuid.uuid4())
        payload = {
            "schema_version": _PACKAGE_SCHEMA_VERSION,
            "package_id": package_id,
            "install_id": self._install_id(connection),
            "period_start": _isoformat(period_start),
            "period_end": _isoformat(period_end),
            "generated_at": _isoformat(now),
            "resource": {"hermes_version": period_row["hermes_version"]},
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
        for row in rows:
            connection.execute(
                """
                    UPDATE counter_aggregates
                    SET packaged_value = value
                    WHERE period_start = ?
                      AND metric_name = ?
                      AND hermes_version = ?
                      AND dimensions_json = ?
                    """,
                (
                    period_value,
                    row["metric_name"],
                    period_row["hermes_version"],
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
