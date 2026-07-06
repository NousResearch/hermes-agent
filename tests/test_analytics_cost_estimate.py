"""cost-estimate analytics: per-tier pricing + projection + endpoint."""
import asyncio
import json
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from agent.analytics import compute_cost_estimate


class _PE:
    input_cost_per_million = Decimal("5.00")
    output_cost_per_million = Decimal("25.00")
    cache_read_cost_per_million = Decimal("0.50")
    cache_write_cost_per_million = Decimal("6.25")


def _groups():
    return [
        {"model": "claude", "billing_provider": "anthropic", "billing_base_url": None,
         "sessions": 2, "input_tokens": 1_000_000, "output_tokens": 200_000,
         "cache_read_tokens": 4_000_000, "cache_write_tokens": 0, "reasoning_tokens": 50_000,
         "estimated_cost_usd": 9.9},
        {"model": "mystery", "billing_provider": "x", "billing_base_url": None,
         "sessions": 1, "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0,
         "cache_write_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.42},
    ]


def test_cost_breakdown_and_projection():
    est = compute_cost_estimate(_groups(), 7 * 86400, lambda m, p, b: _PE() if m == "claude" else None)
    assert est["models"][0]["cost_usd"] == 12.0  # 5 + 5 + 2
    assert est["models"][0]["cost_source"] == "pricing"
    # unpriced model falls back to stored estimate.
    assert est["models"][1]["cost_usd"] == 0.42
    assert est["models"][1]["cost_source"] == "stored_estimate"
    assert est["has_unpriced_models"] is True
    assert est["total_cost_usd"] == 12.42
    assert est["cost_by_tier"] == {"input": 5.0, "output": 5.0, "cache": 2.0}
    assert est["projection"]["daily_usd"] == round(12.42 / 7, 6)
    assert est["projection"]["monthly_usd"] == round(round(12.42 / 7, 6) * 30, 6)


def test_empty_groups():
    est = compute_cost_estimate([], 86400, lambda m, p, b: None)
    assert est["total_cost_usd"] == 0 and est["models"] == []


def test_lookup_exception_falls_back():
    def boom(m, p, b):
        raise RuntimeError("pricing service down")
    est = compute_cost_estimate(_groups()[:1], 86400, boom)
    assert est["models"][0]["cost_source"] == "stored_estimate"
    assert est["models"][0]["cost_usd"] == 9.9


import importlib.util as _ilu
_needs_aiohttp = pytest.mark.skipif(_ilu.find_spec("aiohttp") is None, reason="aiohttp not installed")


@pytest.fixture()
def db():
    d = SessionDB(db_path=Path(tempfile.mkdtemp()) / "cost.db")
    yield d
    d.close()


@_needs_aiohttp
def test_cost_estimate_endpoint_aggregates_sessions(db):
    now = time.time()
    db.create_session(session_id="s1", source="cli", model="m1")
    # populate session-level token columns + cost via update_token_counts.
    db.update_token_counts("s1", input_tokens=1000, output_tokens=200,
                           cache_read_tokens=500, estimated_cost_usd=0.01, model="m1")
    from gateway.platforms.api_server import APIServerAdapter
    a = APIServerAdapter.__new__(APIServerAdapter)
    a._check_auth = lambda request: None
    a._ensure_session_db = lambda: db
    resp = asyncio.run(a._handle_analytics_cost_estimate(SimpleNamespace(query={"window": "24h"})))
    body = json.loads(resp.text)
    assert body["window"] == "24h"
    assert "total_cost_usd" in body and "projection" in body and "models" in body
