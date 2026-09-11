"""The usage mixin's INSERT must bind one placeholder per column (fleet, 2026-09-12).

``_MODEL_USAGE_UPSERT_SQL`` in ``hermes_state_usage.py`` listed 19 columns against 18
placeholders. It is dead code today — ``SessionDB`` overrides ``update_token_counts`` /
``_record_model_usage`` / ``record_auxiliary_usage`` in ``hermes_state.py`` — so nothing
executes it. If a future upstream merge drops one of those overrides, every usage write
fails with ``18 values for 19 columns`` and the cost ledger silently stops recording.

Two tests: a structural parity check on the SQL text, and an execution of the mixin's own
writer against a real database. Both fail on the unfixed statement.
"""
from __future__ import annotations

import re
import sqlite3

import hermes_state_usage as hsu


def _columns_and_placeholders() -> tuple[int, int]:
    sql = hsu._MODEL_USAGE_UPSERT_SQL
    cols = re.search(r"session_model_usage \((.*?)\)\s*VALUES", sql, re.S).group(1)
    ncols = len([c for c in cols.replace("\n", " ").split(",") if c.strip()])
    return ncols, re.search(r"VALUES \((.*?)\)", sql, re.S).group(1).count("?")


def test_upsert_binds_one_placeholder_per_column():
    ncols, nplaceholders = _columns_and_placeholders()
    assert ncols == nplaceholders, (
        f"_MODEL_USAGE_UPSERT_SQL lists {ncols} columns but binds {nplaceholders} placeholders; "
        "the mixin's writer would raise 'N values for M columns' the moment it is reached."
    )


def test_mixin_writer_actually_records_a_row(tmp_path):
    """Execute the mixin's own ``_record_model_usage`` — not SessionDB's override."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, billing_provider TEXT,"
        " billing_base_url TEXT, billing_mode TEXT);"
        "CREATE TABLE session_model_usage ("
        " session_id TEXT, model TEXT, billing_provider TEXT, billing_base_url TEXT,"
        " billing_mode TEXT, task TEXT, provider_name TEXT, api_call_count INTEGER,"
        " input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,"
        " cache_write_tokens INTEGER, reasoning_tokens INTEGER, estimated_cost_usd REAL,"
        " actual_cost_usd REAL, cost_status TEXT, cost_source TEXT, first_seen REAL, last_seen REAL,"
        " PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task, provider_name));"
    )
    conn.execute("INSERT INTO sessions (id) VALUES ('s1')")

    shim = object.__new__(hsu.SessionUsageMixin)
    shim._record_model_usage(
        conn, "s1", model="deepseek-v4-flash", billing_provider="modelark", provider_name="ModelArk",
        input_tokens=10, output_tokens=3, api_call_count=1, actual_cost_usd=0.0, cost_status="included",
    )

    row = conn.execute("SELECT * FROM session_model_usage").fetchone()
    assert row is not None, "the mixin's writer inserted nothing"
    assert row["provider_name"] == "ModelArk"
    assert row["input_tokens"] == 10 and row["output_tokens"] == 3
    assert row["cost_status"] == "included"
