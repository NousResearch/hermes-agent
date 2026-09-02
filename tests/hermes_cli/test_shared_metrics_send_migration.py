"""Tests for the additive send-state migration on ``package_outbox``.

The store schema version must NOT move when these columns are added: the
existing loader raises on any version it does not recognise, so bumping it
would hard-fail an older Hermes (a second profile on an older build, or a
rollback) against the same database file.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_cli.observability.shared_metrics import SharedMetricsStore

SEND_COLUMNS = {
    "sent_at",
    "send_state",
    "send_attempts",
    "next_attempt_at",
    "last_error",
    "sent_install_id",
}


def _columns(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return {row[1] for row in connection.execute("PRAGMA table_info(package_outbox)")}
    finally:
        connection.close()


def _schema_version(db_path):
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = 'schema_version'"
        ).fetchone()
        return row[0] if row else None
    finally:
        connection.close()


@pytest.fixture
def store(tmp_path):
    return SharedMetricsStore(
        database_path=tmp_path / "metrics.sqlite3",
        outbox_directory=tmp_path / "outbox",
    )


class TestFreshDatabase:
    def test_send_columns_exist(self, store):
        assert SEND_COLUMNS <= _columns(store.database_path)

    def test_original_columns_survive(self, store):
        assert {
            "package_id",
            "period_start",
            "period_end",
            "payload_json",
            "created_at",
            "exported_at",
        } <= _columns(store.database_path)

    def test_send_attempts_defaults_to_zero(self, store):
        connection = sqlite3.connect(store.database_path)
        try:
            connection.execute(
                """
                INSERT INTO package_outbox(
                    package_id, period_start, period_end, payload_json, created_at
                ) VALUES ('p', '2026-01-01', '2026-01-02', '{}', '2026-01-01T00:00:00Z')
                """
            )
            connection.commit()
            row = connection.execute(
                "SELECT send_attempts, send_state, sent_install_id FROM package_outbox"
            ).fetchone()
        finally:
            connection.close()
        assert row[0] == 0
        assert row[1] is None
        assert row[2] is None


class TestUpgradeFromPreSendDatabase:
    """The real-world case: a database written before this feature existed."""

    @pytest.fixture
    def legacy_db(self, tmp_path):
        path = tmp_path / "metrics.sqlite3"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE telemetry_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO telemetry_state(key, value) VALUES ('schema_version', '2')"
            )
            connection.execute(
                """
                CREATE TABLE package_outbox (
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
                CREATE TABLE counter_aggregates (
                    period_start TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    hermes_version TEXT NOT NULL,
                    os_family TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    install_method TEXT NOT NULL,
                    dimensions_json TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    packaged_value INTEGER NOT NULL,
                    PRIMARY KEY (
                        period_start, metric_name, hermes_version, os_family,
                        architecture, install_method, dimensions_json
                    )
                )
                """
            )
            for i in range(3):
                connection.execute(
                    """
                    INSERT INTO package_outbox(
                        package_id, period_start, period_end, payload_json,
                        created_at, exported_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"pkg-{i}",
                        "2026-08-2%d" % i,
                        "2026-08-2%d" % (i + 1),
                        json.dumps({"package_id": f"pkg-{i}"}),
                        "2026-08-2%dT00:00:00Z" % i,
                        "2026-08-2%dT01:00:00Z" % i,
                    ),
                )
            connection.commit()
        finally:
            connection.close()
        return path

    def test_upgrade_preserves_every_row(self, legacy_db, tmp_path):
        SharedMetricsStore(
            database_path=legacy_db, outbox_directory=tmp_path / "outbox"
        )
        connection = sqlite3.connect(legacy_db)
        try:
            count = connection.execute("SELECT COUNT(*) FROM package_outbox").fetchone()[0]
            payloads = connection.execute(
                "SELECT package_id, payload_json FROM package_outbox ORDER BY package_id"
            ).fetchall()
        finally:
            connection.close()
        assert count == 3
        assert payloads == [
            ("pkg-0", '{"package_id": "pkg-0"}'),
            ("pkg-1", '{"package_id": "pkg-1"}'),
            ("pkg-2", '{"package_id": "pkg-2"}'),
        ]

    def test_upgrade_adds_the_send_columns(self, legacy_db, tmp_path):
        SharedMetricsStore(
            database_path=legacy_db, outbox_directory=tmp_path / "outbox"
        )
        assert SEND_COLUMNS <= _columns(legacy_db)

    def test_upgrade_does_not_move_the_schema_version(self, legacy_db, tmp_path):
        """Bumping would make older builds refuse the same file."""
        SharedMetricsStore(
            database_path=legacy_db, outbox_directory=tmp_path / "outbox"
        )
        assert _schema_version(legacy_db) == "2"

    def test_migration_is_idempotent(self, legacy_db, tmp_path):
        for _ in range(3):
            SharedMetricsStore(
                database_path=legacy_db, outbox_directory=tmp_path / "outbox"
            )
        columns = [
            row[1]
            for row in sqlite3.connect(legacy_db).execute(
                "PRAGMA table_info(package_outbox)"
            )
        ]
        assert len(columns) == len(set(columns)), "columns were added more than once"

    def test_queries_written_before_this_change_still_work(self, legacy_db, tmp_path):
        """The shipped export query selects named columns; it must be unaffected."""
        SharedMetricsStore(
            database_path=legacy_db, outbox_directory=tmp_path / "outbox"
        )
        connection = sqlite3.connect(legacy_db)
        try:
            rows = connection.execute(
                """
                SELECT package_id, payload_json
                FROM package_outbox
                WHERE exported_at IS NULL
                ORDER BY created_at, package_id
                """
            ).fetchall()
        finally:
            connection.close()
        assert rows == []
