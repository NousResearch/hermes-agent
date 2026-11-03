"""Spend surfaces must read the billed column, not only the estimate.

Moving a provider-reported cost into ``actual_cost_usd`` is only useful if the
surfaces that report spend read it. Each test pins one surface: the sessions
filter SQL (which also drives prune/archive selection), the store-wide summary,
and the web analytics rollups.
"""

import sqlite3

import pytest

import hermes_state_maintenance as hsm
import hermes_state_usage as hsu


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, parent_session_id TEXT, message_count INTEGER,
            archived INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            estimated_cost_usd REAL, actual_cost_usd REAL,
            cost_status TEXT, cost_source TEXT, started_at REAL
        )
    """)
    return conn


def _cost_of(conn, row_id):
    """Evaluate _COST_SQL the way the sessions filters / prune selection do."""
    return conn.execute(
        f"SELECT {hsm._COST_SQL} FROM sessions s WHERE s.id = ?", (row_id,)
    ).fetchone()[0]


def _insert(conn, row_id, estimated, actual):
    conn.execute(
        "INSERT INTO sessions (id, message_count, archived, estimated_cost_usd, actual_cost_usd)"
        " VALUES (?,?,?,?,?)",
        (row_id, 1, 0, estimated, actual),
    )


def test_cost_sql_reads_a_billed_session():
    """The motivating case: a priced OpenRouter call lives only in actual_cost_usd."""
    conn = _db()
    _insert(conn, "billed", 0.0, 0.00592)
    assert _cost_of(conn, "billed") == pytest.approx(0.00592)


def test_cost_sql_reads_an_estimated_session():
    conn = _db()
    _insert(conn, "estimated", 0.0042, 0.0)
    assert _cost_of(conn, "estimated") == pytest.approx(0.0042)


def test_cost_sql_sums_a_mixed_session_instead_of_picking_one_half():
    """A session with an estimated call then a billed one spent BOTH amounts."""
    conn = _db()
    _insert(conn, "mixed", 0.0042, 0.000635248)
    # Sum, not COALESCE: the estimated half must not disappear from filters/prune.
    assert _cost_of(conn, "mixed") == pytest.approx(0.0042 + 0.000635248, rel=1e-9)


def test_cost_sql_treats_null_columns_as_zero():
    conn = _db()
    _insert(conn, "empty", None, None)
    assert _cost_of(conn, "empty") == 0


class _FakeDB:
    """Just enough SessionDB surface for usage_totals (one read)."""

    def __init__(self, conn):
        self._conn = conn

    def _read_one(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()


def test_store_wide_summary_includes_billed_spend():
    conn = _db()
    _insert(conn, "a", 0.0042, 0.000635248)
    totals = hsu.SessionUsageMixin.usage_totals(_FakeDB(conn))
    assert totals["cost_usd"] == pytest.approx(0.0042 + 0.000635248, rel=1e-9)


def test_analytics_rollups_expose_the_billed_column():
    """Daily, per-model and totals all carry actual_cost; a UI can then show real spend."""
    import inspect

    from hermes_cli.web_routers import analytics

    src = inspect.getsource(analytics)
    assert src.count("COALESCE(SUM(actual_cost_usd), 0)") >= 3, (
        "daily, by_model and totals must each expose the billed column"
    )
