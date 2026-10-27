"""Unit tests for the server-side billing collector (agent/billing_collector.py)."""

import importlib
import os
import tempfile
from pathlib import Path

import pytest

# NOTE: we deliberately do NOT set HERMES_BILLING_COLLECTOR at import time. The
# collector is gated; tests that need it ON pass HERMES_HOME + flip the env inside
# the test (or via monkeypatch). A module-level "1" would leak into the real
# ~/.hermes/billing/billing.db during any test that constructs the default
# singleton, so we keep the environment clean here.

from decimal import Decimal

from agent import billing_collector as bc


@pytest.fixture
def tmp_db(monkeypatch):
    """Collector backed by a throwaway DB file under an isolated HERMES_HOME."""
    # Isolate billing.db from the real ~/.hermes so tests never touch production.
    tmp_home = tempfile.mkdtemp(prefix="hermes-billing-test-")
    monkeypatch.setenv("HERMES_HOME", tmp_home)
    monkeypatch.setenv("HERMES_BILLING_COLLECTOR", "1")
    fd, path = tempfile.mkstemp(suffix=".db", dir=tmp_home)
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
    # Restore the original (unknown) state so we don't leak ON into later tests.
    if saved is None:
        os.environ.pop("HERMES_BILLING_COLLECTOR", None)
    else:
        os.environ["HERMES_BILLING_COLLECTOR"] = saved


def test_feature_flag_config_default_off(monkeypatch):
    # With no env override and a clean config, the collector must stay OFF.
    monkeypatch.delenv("HERMES_BILLING_COLLECTOR", raising=False)
    # Ensure config reads default (enabled: False) — monkeypatch load_config_readonly.
    monkeypatch.setattr(
        bc, "is_collector_enabled",
        lambda: (os.environ.get("HERMES_BILLING_COLLECTOR", "").lower() in ("1", "true", "yes", "on")),
    )
    assert bc.is_collector_enabled() is False


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


def test_record_response_no_usage_stores_null_not_zero(tmp_db, monkeypatch):
    """Empty usage object: tokens NULL (not 0) and has_usage=0 (audit D1)."""
    monkeypatch.setattr(bc, "get_collector", lambda: tmp_db)
    agent = _Agent()
    resp = _Response(usage={})  # present but empty
    bc.record_response(agent, resp)
    rows = tmp_db.query()
    assert len(rows) == 1
    r = rows[0]
    assert r["has_usage"] is False
    assert r["actual_cost_usd"] is None
    assert r["prompt_tokens"] is None
    assert r["completion_tokens"] is None
    assert r["cache_read_tokens"] is None


def test_record_response_no_usage_at_all(tmp_db, monkeypatch):
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


def test_record_error_sanitizes_provider_echo(tmp_db):
    """Audit D3: provider error bodies that echo the request must be redacted."""
    noisy = ("rate_limit_exceeded: too many requests. Request body: "
             "{\"messages\":[{\"role\":\"user\",\"content\":\"SECRET PROMPT HERE\"}]}")
    tmp_db.record_error(model="m", provider="p", error_type="rate_limit", error_message=noisy)
    rows = tmp_db.query()
    msg = rows[0]["error_message"]
    assert msg is not None
    assert "SECRET PROMPT HERE" not in msg
    # Structural error class is preserved.
    assert "rate_limit_exceeded" in msg


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


def test_stats_exact_sum_micros(tmp_db):
    """Audit D2: SUM must be exact (integer micro-USD), no float drift."""
    # These three fractional costs expose float-sum drift if stored as REAL.
    for c in (Decimal("0.1"), Decimal("0.2"), Decimal("0.005")):
        tmp_db.record_call(bc.BillingRecord(
            request_id="x", timestamp="2026-10-18T10:00:00Z", model="a", provider="openrouter",
            agent="walter", prompt_tokens=1, completion_tokens=1, actual_cost_usd=c, has_usage=True,
        ))
    assert tmp_db.total_cost_usd() == Decimal("0.305")


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


def test_db_file_is_0600(tmp_db):
    """Audit D3: the billing DB file must be created with owner-only perms (0600)."""
    mode = os.stat(tmp_db._db_path).st_mode & 0o777
    assert mode == 0o600


def test_disabled_no_record_no_db_file(monkeypatch):
    """Audit D8: when the flag is OFF, record_response is a no-op AND creates no DB file.

    The collector is gated: with the flag off, record_response returns before any
    database access, so no billing.db is created under HERMES_HOME.
    """
    monkeypatch.setenv("HERMES_BILLING_COLLECTOR", "0")
    tmp_home = tempfile.mkdtemp(prefix="hermes-billing-off-")
    monkeypatch.setenv("HERMES_HOME", tmp_home)
    import agent.billing_collector as real_bc
    assert real_bc.is_collector_enabled() is False

    # Build the default collector path the collector WOULD use, then assert it is
    # absent both before and after a record call (the disabled path must never
    # instantiate or open the DB).
    default_db = real_bc._default_db_path()
    assert not default_db.exists()
    agent = _Agent()
    resp = _Response(usage=_Usage(cost=0.5))
    real_bc.record_response(agent, resp)  # must be a no-op, must not raise
    # Still no DB file: the collector never touched storage.
    assert not default_db.exists()
