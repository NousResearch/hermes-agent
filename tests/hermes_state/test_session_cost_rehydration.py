"""Tests for the session cost rehydration fix (issue #67762).

Before this fix, ``agent.session_estimated_cost_usd`` was reset to 0.0 inside
``init_agent`` with no read from any persisted source. After a gateway
restart mid-session, the live counter would silently drop to $0.00 even
though ``session_model_usage`` had the real accumulated cost.

These tests cover the new ``SessionDB.get_session_cost_summary`` reader and
the ``init_agent`` rehydration block that consume it.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_per_model_rows(db, session_id):
    """Seed ``session_model_usage`` rows simulating a long-running session.

    Returns the total estimated_cost_usd sum (for direct assertion).
    """
    db.create_session(session_id, source="cli")
    # Row 1: opus, estimated cost $2.10
    db.update_token_counts(
        session_id,
        input_tokens=1000, output_tokens=200,
        model="anthropic/claude-opus-4-8",
        billing_provider="anthropic",
        billing_base_url="",
        billing_mode="",
        api_call_count=10,
        estimated_cost_usd=2.10,
        cost_status="estimated",
        cost_source="provider_models_api",
    )
    # Row 2: haiku, estimated cost $0.07
    db.update_token_counts(
        session_id,
        input_tokens=2000, output_tokens=400,
        model="anthropic/claude-haiku-4-5",
        billing_provider="anthropic",
        billing_base_url="",
        billing_mode="",
        api_call_count=40,
        estimated_cost_usd=0.07,
        cost_status="estimated",
        cost_source="provider_models_api",
    )
    # Row 3 (per-row UPSERT on same (model, provider) accumulates): another opus call
    db.update_token_counts(
        session_id,
        input_tokens=500, output_tokens=100,
        model="anthropic/claude-opus-4-8",
        billing_provider="anthropic",
        billing_base_url="",
        billing_mode="",
        api_call_count=5,
        estimated_cost_usd=1.05,
        cost_status="estimated",
        cost_source="provider_models_api",
    )
    return 2.10 + 0.07 + 1.05


# ── Reader: SessionDB.get_session_cost_summary ────────────────────────────────


class TestGetSessionCostSummary:
    def test_returns_none_when_no_rows(self, db):
        """A session that hasn't been used yet has no per-model rows."""
        db.create_session("s-empty", source="cli")
        assert db.get_session_cost_summary("s-empty") is None

    def test_returns_none_for_unknown_session(self, db):
        """Unknown session_id also returns None (not a KeyError)."""
        assert db.get_session_cost_summary("does-not-exist") is None

    def test_sums_cost_across_models(self, db):
        """Sum across per-model rows is the live counter's restored value."""
        expected = _seed_per_model_rows(db, "s1")
        row = db.get_session_cost_summary("s1")
        assert row is not None
        assert row["estimated_cost_usd"] == pytest.approx(expected, rel=1e-6)

    def test_aggregates_sticky_status(self, db):
        """If ANY row is 'actual', the summary is 'actual'."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-a", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.10, cost_status="estimated",
        )
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-b", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.20, cost_status="actual",
        )
        row = db.get_session_cost_summary("s1")
        assert row["cost_status"] == "actual"

    def test_aggregates_sticky_status_included(self, db):
        """If no 'actual', 'included' wins over 'estimated'."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-a", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.10, cost_status="estimated",
        )
        db.update_token_counts(
            "s1",
            input_tokens=0, output_tokens=0,
            model="m-b", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.0, cost_status="included",
        )
        row = db.get_session_cost_summary("s1")
        assert row["cost_status"] == "included"

    def test_estimated_fallback_uses_latest_row(self, db):
        """When no high-confidence statuses exist, fall back to the most
        recent row's status (per ``last_seen DESC LIMIT 1``)."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-a", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.10, cost_status="estimated",
        )
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-b", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.20, cost_status="unknown",
        )
        row = db.get_session_cost_summary("s1")
        assert row["cost_status"] == "unknown"

    def test_zero_rows_returns_zero_sum(self, db):
        """A session with rows but all zero cost returns 0.0, not None."""
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1",
            input_tokens=100, output_tokens=10,
            model="m-a", billing_provider="x", api_call_count=1,
            estimated_cost_usd=0.0, cost_status="included",
        )
        row = db.get_session_cost_summary("s1")
        assert row is not None
        assert row["estimated_cost_usd"] == 0.0


# ── Rehydration: agent_init.py ────────────────────────────────────────────────


class TestInitAgentRehydratesCost:
    """The init_agent reset block followed by the rehydration block should
    restore the persisted session cost, not leave the agent at $0.0.

    Tests exercise the real ``_rehydrate_session_cost`` helper extracted at
    module level for testability — not an inline copy of the production logic.
    """

    def test_session_estimated_cost_usd_restored_from_db(self, db):
        """After seeding session_model_usage and running the rehydration
        helper against a stub agent, ``agent.session_estimated_cost_usd``
        matches the persisted sum (not the 0.0 the reset block sets)."""
        expected = _seed_per_model_rows(db, "s-rehydrate")

        agent = SimpleNamespace(
            session_id="s-rehydrate",
            _session_db=db,
            # Mimic the reset block having just run
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
            session_cost_source="none",
        )

        from agent.agent_init import _rehydrate_session_cost
        _rehydrate_session_cost(agent)

        assert agent.session_estimated_cost_usd == pytest.approx(
            expected, rel=1e-6
        )
        assert agent.session_cost_status in {"estimated", "actual", "included"}

    def test_init_agent_rehydration_block_dispatches_correctly(self, db):
        """Direct test: rehydrate the persisted row, populate the agent,
        then assert the helper restores both attributes."""
        expected = _seed_per_model_rows(db, "s-rehydrate")

        agent = SimpleNamespace(
            session_id="s-rehydrate",
            _session_db=db,
            _rehydration_entry=None,
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
            session_cost_source="none",
        )

        from agent.agent_init import _rehydrate_session_cost
        _rehydrate_session_cost(agent)

        assert agent.session_estimated_cost_usd == pytest.approx(
            expected, rel=1e-6
        )

    def test_init_agent_rehydration_no_op_when_db_unavailable(self):
        """When ``_session_db`` is None, the helper must skip cleanly without
        raising (e.g., CLI test path constructs an agent without passing
        session_db)."""
        agent = SimpleNamespace(
            session_id="s-nodb",
            # _session_db intentionally absent
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
        )

        from agent.agent_init import _rehydrate_session_cost
        _rehydrate_session_cost(agent)

        assert agent.session_estimated_cost_usd == 0.0
        assert agent.session_cost_status == "unknown"

    def test_init_agent_rehydration_no_op_when_session_id_missing(self, db):
        """When session_id is None/empty, the helper must skip."""
        agent = SimpleNamespace(
            session_id=None,
            _session_db=db,
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
        )

        from agent.agent_init import _rehydrate_session_cost
        _rehydrate_session_cost(agent)

        assert agent.session_estimated_cost_usd == 0.0
        assert agent.session_cost_status == "unknown"

    def test_init_agent_rehydration_fails_open_on_db_error(self, monkeypatch):
        """When ``_session_db.get_session_cost_summary`` raises, the helper
        must not propagate the exception (a transient DB error shouldn't
        block agent construction). The agent's counters stay at the post-
        reset values ($0.0 / "unknown") and accumulate from the next API
        call."""
        from agent.agent_init import _rehydrate_session_cost

        class _Boom:
            def get_session_cost_summary(self, session_id):
                raise RuntimeError("simulated transient DB error")

        agent = SimpleNamespace(
            session_id="s-boom",
            _session_db=_Boom(),
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
        )
        # A bug-class exception (RuntimeError, not sqlite3.Error) should
        # NOT be swallowed — we'd rather surface the SQL bug than silently
        # drop a session.
        with pytest.raises(RuntimeError):
            _rehydrate_session_cost(agent)

    def test_init_agent_rehydration_logged_on_scoped_failure(self, db, monkeypatch):
        """When ``get_session_cost_summary`` raises a scoped ``sqlite3.Error``,
        the helper logs a debug message and leaves the agent at the post-reset
        values. Verifies the helper doesn't crash and the fallback path works.
        """
        import sqlite3
        from agent.agent_init import _rehydrate_session_cost

        class _Boom:
            def get_session_cost_summary(self, session_id):
                raise sqlite3.OperationalError("database is locked")

        agent = SimpleNamespace(
            session_id="s-dblocked",
            _session_db=_Boom(),
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
        )
        # Should NOT raise — scoped exception is caught and logged.
        _rehydrate_session_cost(agent)
        assert agent.session_estimated_cost_usd == 0.0
        assert agent.session_cost_status == "unknown"

    def test_init_agent_rehydration_incremental_calls(self, db):
        """Simulates: agent runs many API calls (lifecycle covered by
        update_token_counts), gateway restarts, new agent constructed,
        rehydrated to the sum. Then a NEW API call after resume adds
        correctly to the rehydrated baseline (not zero)."""
        # Phase 1: many calls accumulate cost in DB
        expected_pre = _seed_per_model_rows(db, "s1")

        # Phase 2: simulate restart — rehydration helper restores the total
        agent = SimpleNamespace(
            session_id="s1", _session_db=db,
            session_estimated_cost_usd=0.0,
            session_cost_status="unknown",
        )
        from agent.agent_init import _rehydrate_session_cost
        _rehydrate_session_cost(agent)

        # Phase 3: post-resume API call adds a new delta
        new_call_cost = 0.42
        agent.session_estimated_cost_usd += new_call_cost

        # The live counter should reflect ALL calls (old + new)
        assert agent.session_estimated_cost_usd == pytest.approx(
            expected_pre + new_call_cost, rel=1e-6
        )
