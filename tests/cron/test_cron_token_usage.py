"""Per-job token-spend budget, step 1: usage recorded beside the terminal ledger row.

Accounting only — no enforcement. Verifies that a finished cron execution persists the agent's
token/cost counters, that absent counters stay NULL (no fabricated zeros), and that a no_agent
script run records no usage.
"""

import pytest

from cron import executions


class _FakeAgent:
    """Minimal stand-in for a live AIAgent's session usage counters."""

    def __init__(self, **counters):
        for k, v in counters.items():
            setattr(self, k, v)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return executions


def _finish(executions, *, success=True, usage=None):
    row = executions.create_execution("job-usage", source="builtin")
    executions.mark_execution_running(row["id"])
    executions.finish_execution(row["id"], success=success, usage=usage)
    return row["id"]


def test_usage_from_agent_reads_counters(ledger):
    agent = _FakeAgent(
        session_prompt_tokens=1200,
        session_completion_tokens=350,
        session_total_tokens=1550,
        session_reasoning_tokens=100,
        session_estimated_cost_usd=0.0042,
        session_cost_status="estimated",
    )
    usage = executions.usage_from_agent(agent)
    assert usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 350,
        "total_tokens": 1550,
        "reasoning_tokens": 100,
        "estimated_cost_usd": 0.0042,
        "cost_status": "estimated",
    }


def test_usage_from_agent_backfills_total_when_missing(ledger):
    agent = _FakeAgent(session_prompt_tokens=800, session_completion_tokens=200)
    usage = executions.usage_from_agent(agent)
    assert usage["total_tokens"] == 1000


def test_usage_from_agent_none_returns_empty(ledger):
    # no_agent script run: no agent at all -> no usage, not a zero.
    assert executions.usage_from_agent(None) == {}


def test_usage_from_agent_never_fabricates(ledger):
    # A bare agent exposing nothing yields an empty snapshot, not zeros.
    assert executions.usage_from_agent(_FakeAgent()) == {}
    # A provider that reports cost_status='unknown' still records the status, but only real numbers.
    agent = _FakeAgent(session_cost_status="unknown")
    usage = executions.usage_from_agent(agent)
    assert usage == {"cost_status": "unknown"}


def test_finished_execution_persists_usage(ledger):
    exec_id = _finish(
        ledger, usage={
            "prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550,
            "reasoning_tokens": 100, "estimated_cost_usd": 0.0042, "cost_status": "estimated",
        })
    record = ledger.get_execution(exec_id)
    assert record["total_tokens"] == 1550
    assert record["prompt_tokens"] == 1200
    assert record["completion_tokens"] == 350
    assert record["reasoning_tokens"] == 100
    assert record["estimated_cost_usd"] == pytest.approx(0.0042)
    assert record["cost_status"] == "estimated"


def test_finished_execution_without_usage_leaves_columns_null(ledger):
    # A no_agent script run or a pre-model failure records NULL, never 0.
    exec_id = _finish(ledger)
    record = ledger.get_execution(exec_id)
    assert record["total_tokens"] is None
    assert record["estimated_cost_usd"] is None
    assert record["cost_status"] is None
