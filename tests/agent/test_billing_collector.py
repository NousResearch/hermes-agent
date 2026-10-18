"""Unit tests for the server-side billing collector (agent/billing_collector.py)."""

import importlib
import os
import tempfile
from pathlib import Path

import pytest

# Feature flag must be controlled deterministically. Force the env override ON so the
# collector is active regardless of config.yaml state during these tests.
os.environ["HERMES_BILLING_COLLECTOR"] = "1"

from decimal import Decimal

from agent import billing_collector as bc


@pytest.fixture
def tmp_db():
    """Collector backed by a throwaway DB file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    c = bc.BillingCollector(path, ttl_days=90)
    yield c
    try:
        os.unlink(path)
    except OSError:
        pass


class _Usage:
    """Minimal OpenAI-style usage mock with an OpenRouter `cost` field."""

    def __init__(self, *, prompt_tokens=10, completion_tokens=5, cache_read_tokens=0,
                 cache_write_tokens=0, reasoning_tokens=0, cost=0.0042, currency="USD"):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.reasoning_tokens = reasoning_tokens
        # OpenRouter emits usage.cost as a float USD.
        self.cost = cost
        self.currency = currency


class _Response:
    def __init__(self, model="openai/gpt-4o", usage=None):
        self.model = model
        self.usage = usage


class _Agent:
    def __init__(self, model="openai/gpt-4o", provider="openrouter", api_mode="chat"):
        self.model = model
        self.provider = provider
        self.api_mode = api_mode


def test_feature_flag_env_override_on():
    saved = os.environ.get("HERMES_BILLING_COLLECTOR")
    os.environ["HERMES_BILLING_COLLECTOR"] = "1"
    assert bc.is_collector_enabled() is True
    os.environ["HERMES_BILLING_COLLECTOR"] = "0"
    assert bc.is_collector_enabled() is False
    # Restore the module-level ON state so subsequent tests keep the collector active.
    if saved is None:
        os.environ["HERMES_BILLING_COLLECTOR"] = "1"
    else:
        os.environ["HERMES_BILLING_COLLECTOR"] = saved


def test_record_response_captures_usage_and_cost(tmp_db, monkeypatch):
    monkeypatch.setattr(bc, "get_collector", lambda: tmp_db)
    agent = _Agent()
    resp = _Response(model="openai/gpt-4o", usage=_Usage(prompt_tokens=100, completion_tokens=50, cost=0.0123))
    bc.record_response(agent, resp, api_request_id="req-1", task_id="t1", turn_id="turn-1")

    rows = tmp_db.query()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "openai/gpt-4o"
    assert r["provider"] == "openrouter"
    # Key requirement: actual_cost_usd is NOT empty when the upstream reports it.
    assert r["actual_cost_usd"] == Decimal("0.0123")
    assert r["prompt_tokens"] == 100
    assert r["completion_tokens"] == 50
    assert r["has_usage"] is True
    assert r["is_error"] is False
    assert r["billing_source"] == "server_response"
    assert r["request_id"] == "req-1"
    assert r["task_id"] == "t1"


def test_record_response_cache_tokens(tmp_db, monkeypatch):
    monkeypatch.setattr(bc, "get_collector", lambda: tmp_db)
    agent = _Agent(provider="anthropic", api_mode="anthropic_messages")
    # Anthropic reports cache_read_input_tokens; cost absent (no OpenRouter cost field).
    usage = type("_U", (), {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 100,
    })()
    resp = _Response(model="claude-opus-4-8", usage=usage)
    bc.record_response(agent, resp)
    rows = tmp_db.query()
    assert len(rows) == 1
    r = rows[0]
    # CanonicalUsage.prompt_tokens aggregates input + cache_read + cache_write (billing basis).
    assert r["prompt_tokens"] == 1900
    assert r["cache_read_tokens"] == 800
    assert r["cache_write_tokens"] == 100
    assert r["completion_tokens"] == 200
    # No OpenRouter cost field -> cost stays NULL but the row is still recorded.
    assert r["actual_cost_usd"] is None
    assert r["has_usage"] is True


def test_record_response_cost_nested_dict(tmp_db, monkeypatch):
    monkeypatch.setattr(bc, "get_collector", lambda: tmp_db)
    agent = _Agent()
    usage = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "cost_details": {"total": "0.0075", "currency": "USD"},
    }
    resp = _Response(usage=usage)
    bc.record_response(agent, resp)
    rows = tmp_db.query()
    assert rows[0]["actual_cost_usd"] == Decimal("0.0075")


def test_record_response_no_usage_still_recorded(tmp_db, monkeypatch):
    monkeypatch.setattr(bc, "get_collector", lambda: tmp_db)
    agent = _Agent()
    resp = _Response(usage=None)
    bc.record_response(agent, resp)
    rows = tmp_db.query()
    assert len(rows) == 1
    assert rows[0]["has_usage"] is False
    assert rows[0]["actual_cost_usd"] is None


def test_record_error(tmp_db):
    tmp_db.record_error(model="m", provider="p", error_type="rate_limit", error_message="429")
    rows = tmp_db.query()
    assert len(rows) == 1
    assert rows[0]["is_error"] is True
    assert rows[0]["error_type"] == "rate_limit"
    assert rows[0]["has_usage"] is False


def test_stats_group_by_model(tmp_db):
    tmp_db.record_call(bc.BillingRecord(
        request_id="r1", timestamp="2026-10-18T10:00:00Z", model="a", provider="openrouter",
        agent="walter", prompt_tokens=10, completion_tokens=5, actual_cost_usd=Decimal("0.01"),
        has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="r2", timestamp="2026-10-18T11:00:00Z", model="a", provider="openrouter",
        agent="walter", prompt_tokens=20, completion_tokens=10, actual_cost_usd=Decimal("0.02"),
        has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="r3", timestamp="2026-10-18T12:00:00Z", model="b", provider="openai",
        agent="heisenberg", prompt_tokens=5, completion_tokens=2, actual_cost_usd=Decimal("0.005"),
        has_usage=True,
    ))
    stats = tmp_db.stats(group_by="model")
    by_model = {s["bucket"]: s for s in stats}
    assert by_model["a"]["calls"] == 2
    assert by_model["a"]["total_cost_usd"] == Decimal("0.03")
    assert by_model["b"]["total_cost_usd"] == Decimal("0.005")


def test_stats_group_by_agent(tmp_db):
    tmp_db.record_call(bc.BillingRecord(
        request_id="r1", timestamp="2026-10-18T10:00:00Z", model="a", provider="openrouter",
        agent="walter", prompt_tokens=10, completion_tokens=5, actual_cost_usd=Decimal("0.01"), has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="r2", timestamp="2026-10-18T10:30:00Z", model="a", provider="openrouter",
        agent="heisenberg", prompt_tokens=10, completion_tokens=5, actual_cost_usd=Decimal("0.01"), has_usage=True,
    ))
    stats = tmp_db.stats(group_by="agent")
    by_agent = {s["bucket"]: s for s in stats}
    assert by_agent["walter"]["calls"] == 1
    assert by_agent["heisenberg"]["calls"] == 1


def test_stats_group_by_task(tmp_db):
    tmp_db.record_call(bc.BillingRecord(
        request_id="r1", timestamp="2026-10-18T10:00:00Z", model="a", provider="openrouter",
        agent="walter", task_id="t_abc", prompt_tokens=10, completion_tokens=5,
        actual_cost_usd=Decimal("0.01"), has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="r2", timestamp="2026-10-18T10:30:00Z", model="a", provider="openrouter",
        agent="walter", task_id="t_abc", prompt_tokens=10, completion_tokens=5,
        actual_cost_usd=Decimal("0.02"), has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="r3", timestamp="2026-10-18T10:45:00Z", model="a", provider="openrouter",
        agent="walter", task_id="t_def", prompt_tokens=10, completion_tokens=5,
        actual_cost_usd=Decimal("0.03"), has_usage=True,
    ))
    stats = tmp_db.stats(group_by="task_id")
    by_task = {s["bucket"]: s for s in stats}
    assert by_task["t_abc"]["total_cost_usd"] == Decimal("0.03")
    assert by_task["t_def"]["total_cost_usd"] == Decimal("0.03")


def test_total_cost_usd(tmp_db):
    for c in (Decimal("0.01"), Decimal("0.02"), None):
        tmp_db.record_call(bc.BillingRecord(
            request_id="x", timestamp="2026-10-18T10:00:00Z", model="a", provider="openrouter",
            agent="walter", prompt_tokens=1, completion_tokens=1, actual_cost_usd=c, has_usage=True,
        ))
    # None rows contribute 0; total = 0.03.
    assert tmp_db.total_cost_usd() == Decimal("0.03")


def test_cleanup_ttl(tmp_db):
    tmp_db.record_call(bc.BillingRecord(
        request_id="old", timestamp="2020-01-01T00:00:00Z", model="a", provider="openrouter",
        agent="walter", prompt_tokens=1, completion_tokens=1, actual_cost_usd=Decimal("0.01"), has_usage=True,
    ))
    tmp_db.record_call(bc.BillingRecord(
        request_id="new", timestamp="2099-01-01T00:00:00Z", model="a", provider="openrouter",
        agent="walter", prompt_tokens=1, completion_tokens=1, actual_cost_usd=Decimal("0.01"), has_usage=True,
    ))
    deleted = tmp_db.cleanup(ttl_days=90)
    assert deleted == 1
    rows = tmp_db.query()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "new"


def test_disabled_no_record(monkeypatch):
    """When the flag is off, record_response is a no-op (no DB write, no error)."""
    monkeypatch.setenv("HERMES_BILLING_COLLECTOR", "0")
    # Re-evaluate is_collector_enabled under the patched env by calling the real path.
    import agent.billing_collector as real_bc
    assert real_bc.is_collector_enabled() is False
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = real_bc.BillingCollector(path, ttl_days=90)
    monkeypatch.setattr(real_bc, "get_collector", lambda: c)
    agent = _Agent()
    resp = _Response(usage=_Usage(cost=0.5))
    # Should not raise and should not write.
    real_bc.record_response(agent, resp)
    assert c.query() == []
    os.unlink(path)
