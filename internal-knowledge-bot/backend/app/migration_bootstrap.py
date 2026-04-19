from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import text
from sqlalchemy.engine import Engine


# SQLite-focused lightweight bootstrap migrations for legacy local DB files.
# We intentionally keep this simple and idempotent.

_COLUMN_PATCHES: dict[str, list[tuple[str, str]]] = {
    "tenant_policies": [
        ("policy_rules_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("policy_pack", "VARCHAR(32) NOT NULL DEFAULT 'balanced'"),
        ("daily_query_budget", "INTEGER NOT NULL DEFAULT 1000"),
        ("daily_run_budget", "INTEGER NOT NULL DEFAULT 1000"),
        ("daily_cost_budget_usd", "FLOAT NOT NULL DEFAULT 25.0"),
        ("max_top_k", "INTEGER NOT NULL DEFAULT 8"),
        ("max_question_chars", "INTEGER NOT NULL DEFAULT 4000"),
        ("daily_external_api_budget", "INTEGER NOT NULL DEFAULT 200"),
        ("external_api_timeout_cap_seconds", "INTEGER NOT NULL DEFAULT 8"),
        ("public_api_allowlist_json", "TEXT NOT NULL DEFAULT '[]'"),
    ],
    "document_policies": [
        ("freshness_last_checked_at", "DATETIME"),
        ("freshness_last_updated_at", "DATETIME"),
        ("freshness_check_interval_hours", "INTEGER NOT NULL DEFAULT 24"),
        ("freshness_stale_after_hours", "INTEGER NOT NULL DEFAULT 168"),
        ("auto_refresh_enabled", "BOOLEAN NOT NULL DEFAULT 0"),
        ("citation_anchor_mode", "VARCHAR(32) NOT NULL DEFAULT 'char_offsets'"),
    ],
    "chunks": [
        ("start_char", "INTEGER NOT NULL DEFAULT 0"),
        ("end_char", "INTEGER NOT NULL DEFAULT 0"),
        ("page_number", "INTEGER"),
        ("section_label", "VARCHAR(255) NOT NULL DEFAULT ''"),
    ],
    "handoff_tickets": [
        ("sla_target_minutes", "INTEGER NOT NULL DEFAULT 1440"),
        ("due_at", "DATETIME"),
        ("first_response_at", "DATETIME"),
        ("breached_at", "DATETIME"),
    ],
    "query_logs": [
        ("run_id", "INTEGER"),
        ("estimated_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("estimated_output_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("estimated_cost_usd", "FLOAT NOT NULL DEFAULT 0.0"),
    ],
}


def _sqlite_table_names(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    return {str(r[0]) for r in rows}


def _sqlite_table_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info('{table_name}')"))
    # PRAGMA table_info row schema: cid, name, type, notnull, dflt_value, pk
    return {str(r[1]) for r in rows}


def _ensure_columns(conn, table_name: str, patches: Iterable[tuple[str, str]]) -> int:
    existing_columns = _sqlite_table_columns(conn, table_name)
    applied = 0

    for column_name, column_sql in patches:
        if column_name in existing_columns:
            continue
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))
        applied += 1

    return applied


def run_startup_schema_bootstrap(engine: Engine) -> dict[str, int | str]:
    """
    Apply idempotent schema patches for legacy SQLite DBs.

    Returns a small summary for observability (not currently exposed publicly).
    """
    if engine.dialect.name != "sqlite":
        return {"dialect": engine.dialect.name, "applied": 0}

    applied_total = 0
    with engine.begin() as conn:
        table_names = _sqlite_table_names(conn)

        for table_name, patches in _COLUMN_PATCHES.items():
            if table_name not in table_names:
                continue
            applied_total += _ensure_columns(conn, table_name, patches)

        if "public_api_providers" not in table_names:
            conn.execute(
                text(
                    """
                    CREATE TABLE public_api_providers (
                        id INTEGER PRIMARY KEY,
                        name VARCHAR(128) NOT NULL,
                        category VARCHAR(64) NOT NULL DEFAULT 'open-data',
                        base_url VARCHAR(1000) NOT NULL,
                        auth_type VARCHAR(32) NOT NULL DEFAULT 'none',
                        docs_url VARCHAR(1000) NOT NULL DEFAULT '',
                        cors VARCHAR(32) NOT NULL DEFAULT 'unknown',
                        enabled BOOLEAN NOT NULL DEFAULT 1,
                        tenant_scope VARCHAR(32) NOT NULL DEFAULT 'global',
                        default_timeout_seconds INTEGER NOT NULL DEFAULT 5,
                        rate_limit_hint VARCHAR(255) NOT NULL DEFAULT '',
                        normalization_strategy VARCHAR(64) NOT NULL DEFAULT 'auto',
                        sample_query_json TEXT NOT NULL DEFAULT '{}',
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    """
                )
            )
            conn.execute(text("CREATE UNIQUE INDEX uq_public_api_provider_name ON public_api_providers (name)"))
            conn.execute(text("CREATE INDEX ix_public_api_providers_name ON public_api_providers (name)"))
            applied_total += 1

    return {"dialect": "sqlite", "applied": applied_total}
