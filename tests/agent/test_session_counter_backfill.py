"""Regression tests for session-counter backfill on agent rebuild (#50675).

When the gateway invalidates its agent cache mid-session it rebuilds the
``AIAgent`` for the same ``session_id``. Init zeros every session counter, so
``/usage`` reported only the activity since the rebuild. The SessionDB row
already holds the cumulative totals (per-call increments). The rebuild seeds
the in-memory counters from that row by assignment — it does not add on top of
a total upstream already restored, and it does not fold cache tokens into
input a second time.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.agent_init import (
    _backfill_rebuilt_session_counters,
    backfill_session_counters_from_row,
)
from hermes_state import SessionDB


def _zeroed_agent():
    """A bare object with session counters at their init defaults."""
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
    """A persisted row is mirrored onto a freshly zeroed agent."""
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
    # CanonicalUsage: input is uncached; prompt adds both cache buckets once.
    assert agent.session_prompt_tokens == 42_000
    assert agent.session_completion_tokens == 10_000
    assert agent.session_total_tokens == 52_000
    assert agent.session_estimated_cost_usd == 0.42
    assert agent.session_cost_status == "estimated"
    assert agent.session_cost_source == "pricing_table"


def test_backfill_tolerates_null_and_missing_fields():
    """NULL or absent columns fall back to 0 / existing defaults."""
    agent = _zeroed_agent()
    row = {"input_tokens": None, "output_tokens": 5}

    backfill_session_counters_from_row(agent, row)

    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 5
    assert agent.session_prompt_tokens == 0
    assert agent.session_completion_tokens == 5
    assert agent.session_total_tokens == 5
    assert agent.session_api_calls == 0
    assert agent.session_estimated_cost_usd == 0.0
    assert agent.session_cost_status == "unknown"
    assert agent.session_cost_source == "none"


def test_backfill_is_assignment_not_accumulation():
    """A second seed replaces the counters; it does not add the row again."""
    agent = _zeroed_agent()
    row = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_tokens": 3,
        "api_call_count": 1,
    }

    backfill_session_counters_from_row(agent, row)
    backfill_session_counters_from_row(agent, row)

    assert agent.session_api_calls == 1
    assert agent.session_input_tokens == 10
    assert agent.session_cache_read_tokens == 3
    assert agent.session_prompt_tokens == 13
    assert agent.session_total_tokens == 17


def test_rebuild_skips_when_counters_already_live():
    """Do not replace totals upstream already maintains, and do not read the DB."""
    agent = _zeroed_agent()
    agent.session_id = "s1"
    agent.session_api_calls = 4
    agent.session_input_tokens = 100
    agent.session_total_tokens = 150

    def _boom(_session_id):
        raise AssertionError("SessionDB should not be read when counters are live")

    agent._session_db = types.SimpleNamespace(get_session=_boom)
    _backfill_rebuilt_session_counters(agent)

    assert agent.session_api_calls == 4
    assert agent.session_input_tokens == 100
    assert agent.session_total_tokens == 150


def test_backfill_matches_db_accumulated_totals(tmp_path):
    """Per-call SessionDB increments are recovered on a zeroed rebuild."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="whatsapp")
    for _ in range(3):
        db.update_token_counts(
            "s1",
            input_tokens=1_000,
            output_tokens=400,
            cache_read_tokens=200,
            reasoning_tokens=50,
            api_call_count=1,
        )

    agent = _zeroed_agent()
    agent.session_id = "s1"
    agent._session_db = db
    _backfill_rebuilt_session_counters(agent)

    assert agent.session_api_calls == 3
    assert agent.session_input_tokens == 3_000
    assert agent.session_output_tokens == 1_200
    assert agent.session_cache_read_tokens == 600
    assert agent.session_reasoning_tokens == 150
    assert agent.session_prompt_tokens == 3_600
    assert agent.session_total_tokens == 4_800
    db.close()
