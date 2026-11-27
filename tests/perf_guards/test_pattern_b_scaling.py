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

    Today ``sessions.list_sessions`` still issues ~2 statements per listed
    compression root (the N+1 tracked in #95380). This guard allows that legacy
    budget but fails on a regression to N·M (nested walks, per-hop re-queries).
    Statements are counted on every connection the call can use: pooled read
    connections from the store holder's ``_read_ctx()`` and its writer connection.
    """

    N_CHAINS = 12

    @pytest.fixture()
    def chain_db(self, tmp_path: Path):
        from persistence_assembly import open_persistence, profile_scope
        from tests.fixtures.storage_seed import seed_session

        db = open_persistence(profile_scope(home=tmp_path))
        for i in range(self.N_CHAINS):
            root, child = f"root_{i}", f"child_{i}"
            seed_session(db, root, "cli")
            seed_session(db, child, "cli", parent_session_id=root)
            db.sessions.end(root, "compression")
        yield db
        db.close()

    @staticmethod
    def _list_and_count_statements(db, monkeypatch):
        from persistence.conversation.sessions import SessionQuery, SessionWindow
        from tests.fixtures.storage_holder import raw_holder

        holder = raw_holder(db)
        statements: list[str] = []
        real_read_ctx = holder._read_ctx

        @contextlib.contextmanager
        def traced_read_ctx():
            with real_read_ctx() as conn:
                conn.set_trace_callback(statements.append)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)

        monkeypatch.setattr(holder, "_read_ctx", traced_read_ctx)
        holder._conn.set_trace_callback(statements.append)
        try:
            rows = db.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))
        finally:
            holder._conn.set_trace_callback(None)
        return rows, statements

    def test_statement_count_does_not_scale_with_sessions(self, chain_db, monkeypatch):
        rows, statements = self._list_and_count_statements(chain_db, monkeypatch)

        assert len(rows) == self.N_CHAINS
        assert statements, "trace captured nothing: the guard is no longer observing sessions.list_sessions"
        budget = 4 * self.N_CHAINS + 8
        assert len(statements) <= budget, (
            f"sessions.list_sessions issued {len(statements)} statements for "
            f"{self.N_CHAINS} sessions, beyond even the legacy N+1 budget "
            f"({budget}). A nested per-row walk has been introduced. "
            f"Captured SQL: {[s[:80] for s in statements[:10]]}"
        )
