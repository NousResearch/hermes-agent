"""Pattern-B perf-regression guards: hot paths must not degrade to O(N·M).

Pattern B ("rebuild everything per delta/row") has no lintable signature, so
the durable guard is behavioral: pin the *scaling shape* of a known hot path
with a deterministic operation count (SQL statements via sqlite trace
callbacks), never wall-clock timing.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest


class TestListSessionsRichQueryBound:
    """Listing sessions must not walk compression chains with nested per-row queries.

    Today ``list_sessions_rich`` still issues ~2 statements per listed
    compression root (the N+1 tracked in #95380). This guard allows that legacy
    budget but fails on a regression to N·M (nested walks, per-hop re-queries).
    Statements are counted on every connection the call can use: pooled read
    connections from ``_read_ctx()`` and the writer connection.
    """

    N_CHAINS = 12

    @pytest.fixture()
    def chain_db(self, tmp_path: Path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        for i in range(self.N_CHAINS):
            root, child = f"root_{i}", f"child_{i}"
            db.create_session(root, source="cli")
            db.create_session(child, source="cli", parent_session_id=root)
            db.end_session(root, end_reason="compression")
        yield db
        db.close()

    @staticmethod
    def _list_and_count_statements(db, monkeypatch):
        statements: list[str] = []
        real_read_ctx = db._read_ctx

        @contextlib.contextmanager
        def traced_read_ctx():
            with real_read_ctx() as conn:
                conn.set_trace_callback(statements.append)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)

        monkeypatch.setattr(db, "_read_ctx", traced_read_ctx)
        db._conn.set_trace_callback(statements.append)
        try:
            rows = db.list_sessions_rich(limit=50)
        finally:
            db._conn.set_trace_callback(None)
        return rows, statements

    def test_statement_count_does_not_scale_with_sessions(self, chain_db, monkeypatch):
        rows, statements = self._list_and_count_statements(chain_db, monkeypatch)

        assert len(rows) == self.N_CHAINS
        assert statements, "trace captured nothing: the guard is no longer observing list_sessions_rich"
        budget = 4 * self.N_CHAINS + 8
        assert len(statements) <= budget, (
            f"list_sessions_rich issued {len(statements)} statements for "
            f"{self.N_CHAINS} sessions, beyond even the legacy N+1 budget "
            f"({budget}). A nested per-row walk has been introduced. "
            f"Captured SQL: {[s[:80] for s in statements[:10]]}"
        )
