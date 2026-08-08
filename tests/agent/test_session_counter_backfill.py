"""Regression tests for session-counter backfill on agent rebuild (#50675).

When the gateway invalidates its agent cache mid-session (stale message_count
after a large flush / context compression), it rebuilds the ``AIAgent`` for the
same ``session_id``. The rebuilt instance initializes every session counter to
0, so ``/usage`` reported only the activity since the rebuild and silently
undercounted the whole-session totals. The SessionDB row already holds the
correct cumulative totals, so the rebuild now backfills the in-memory counters
from it.
"""
import sys
import types
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub optional heavy deps not installed in the test environment so importing
# agent.agent_init does not pull in network/SDK packages.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.agent_init import backfill_session_counters_from_row  # noqa: E402
from hermes_state import SessionDB  # noqa: E402


def _zeroed_agent():
    """A bare object with all session counters initialized to their defaults."""
    agent = types.SimpleNamespace()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_api_calls = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    return agent


def test_backfill_restores_cumulative_counters_from_row():
    """A persisted row's totals are mirrored back onto a freshly-zeroed agent."""
    agent = _zeroed_agent()
    row = {
        "input_tokens": 35_000,
        "output_tokens": 10_000,
        "cache_read_tokens": 5_000,
        "cache_write_tokens": 2_000,
        "reasoning_tokens": 1_500,
        "api_call_count": 7,
        "estimated_cost_usd": 0.42,
        "cost_status": "estimated",
        "cost_source": "pricing_table",
    }

    backfill_session_counters_from_row(agent, row)

    assert agent.session_input_tokens == 35_000
    assert agent.session_output_tokens == 10_000
    assert agent.session_cache_read_tokens == 5_000
    assert agent.session_cache_write_tokens == 2_000
    assert agent.session_reasoning_tokens == 1_500
    assert agent.session_api_calls == 7
    # total / prompt / completion are derived from input+output (no DB columns)
    assert agent.session_total_tokens == 45_000
    assert agent.session_prompt_tokens == 35_000
    assert agent.session_completion_tokens == 10_000
    assert agent.session_estimated_cost_usd == 0.42
    assert agent.session_cost_status == "estimated"
    assert agent.session_cost_source == "pricing_table"


def test_backfill_tolerates_null_and_missing_fields():
    """NULL/absent columns fall back to 0 / existing defaults (no crash)."""
    agent = _zeroed_agent()
    row = {"input_tokens": None, "output_tokens": 5}  # cost fields absent

    backfill_session_counters_from_row(agent, row)

    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 5
    assert agent.session_total_tokens == 5
    assert agent.session_api_calls == 0
    # Untouched because the row carried no cost data.
    assert agent.session_estimated_cost_usd == 0.0
    assert agent.session_cost_status == "unknown"
    assert agent.session_cost_source == "none"


def test_backfill_matches_db_accumulated_totals(tmp_path):
    """End-to-end: per-call increments in SessionDB are recovered on rebuild.

    This reproduces the #50675 scenario: a session accrues several API calls,
    the agent is rebuilt (counters reset to 0), and the rebuild must recover the
    cumulative totals from the persisted row.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="whatsapp")
    # Simulate three API calls' worth of per-call increments (the CLI/gateway
    # persistence path, update_token_counts in increment mode).
    for _ in range(3):
        db.update_token_counts(
            "s1",
            input_tokens=1_000,
            output_tokens=400,
            cache_read_tokens=200,
            reasoning_tokens=50,
            api_call_count=1,
        )

    # Rebuilt agent starts at zero, then backfills from the row.
    agent = _zeroed_agent()
    backfill_session_counters_from_row(agent, db.get_session("s1"))

    assert agent.session_api_calls == 3
    assert agent.session_input_tokens == 3_000
    assert agent.session_output_tokens == 1_200
    assert agent.session_cache_read_tokens == 600
    assert agent.session_reasoning_tokens == 150
    assert agent.session_total_tokens == 4_200
