"""Logical durable-schema contracts shared by future state backends."""

from collections.abc import Iterable
from dataclasses import dataclass
import sqlite3
from typing import FrozenSet


@dataclass(frozen=True)
class SchemaV22Manifest:
    """Backend-neutral durable tables for the logical state schema v22."""

    version: int
    core_tables: FrozenSet[str]
    telegram_tables: FrozenSet[str]


SCHEMA_V22_MANIFEST = SchemaV22Manifest(
    version=22,
    core_tables=frozenset(
        {
            "schema_version",
            "sessions",
            "messages",
            "session_model_usage",
            "state_meta",
            "gateway_routing",
            "compression_locks",
            "async_delegations",
        }
    ),
    telegram_tables=frozenset(
        {
            "telegram_dm_topic_mode",
            "telegram_dm_topic_bindings",
        }
    ),
)


@dataclass(frozen=True)
class SchemaManifestParity:
    """Difference between a concrete schema and the v22 logical manifest."""

    missing_core: FrozenSet[str]
    unexpected_core: FrozenSet[str]
    missing_telegram: FrozenSet[str]
    unexpected_telegram: FrozenSet[str]

    @property
    def matches(self) -> bool:
        return not (
            self.missing_core
            or self.unexpected_core
            or self.missing_telegram
            or self.unexpected_telegram
        )


def sqlite_relational_table_names(schema_sql: str) -> FrozenSet[str]:
    """Return non-virtual tables declared by SQLite schema DDL.

    FTS tables are SQLite-specific derived search indexes, not part of the
    portable durable-table contract.
    """

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_sql)
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return frozenset(
            name
            for name, table_sql in rows
            if "VIRTUAL TABLE" not in (table_sql or "").upper()
        )
    finally:
        conn.close()


def schema_v22_manifest_parity(
    *,
    core_tables: Iterable[str],
    telegram_tables: Iterable[str] = (),
) -> SchemaManifestParity:
    """Compare concrete durable-table names with the logical v22 manifest."""

    actual_core = frozenset(core_tables)
    actual_telegram = frozenset(telegram_tables)
    manifest = SCHEMA_V22_MANIFEST
    return SchemaManifestParity(
        missing_core=manifest.core_tables - actual_core,
        unexpected_core=actual_core - manifest.core_tables,
        missing_telegram=manifest.telegram_tables - actual_telegram,
        unexpected_telegram=actual_telegram - manifest.telegram_tables,
    )
