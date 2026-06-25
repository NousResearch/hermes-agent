"""token-trends analytics: series, per-request averages, cache-hit rate."""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_token_codec import pack_input_tokens, pack_assistant_tokens
from agent.analytics import compute_token_trends


def test_trends_series_and_cache_hit():
    buckets = [
        {"bucket_start": 0, "requests": 2, "input": 1000, "output": 200, "cache_read": 900, "reasoning": 50},
        {"bucket_start": 60, "requests": 1, "input": 4000, "output": 600, "cache_read": 1000, "reasoning": 0},
    ]
    out = compute_token_trends(buckets)
    assert len(out["series"]) == 2
    assert out["series"][0]["cache_hit_rate"] == 90.0  # 900/1000
    assert out["series"][0]["avg_input_per_request"] == 500.0  # 1000/2
    assert out["totals"]["input"] == 5000 and out["totals"]["output"] == 800
    # overall cache hit = (900+1000)/(1000+4000)
    assert out["cache_hit_rate"] == round(100 * 1900 / 5000, 1)
    assert out["averages_per_request"]["input"] == round(5000 / 3, 1)
    assert "input_distribution" in out["averages_per_request"]


def test_trends_empty():
    out = compute_token_trends([])
    assert out["series"] == [] and out["cache_hit_rate"] is None
    assert out["totals"]["input"] == 0


pytest.importorskip("aiohttp")


@pytest.fixture()
def db():
    d = SessionDB(db_path=Path(tempfile.mkdtemp()) / "trends.db")
    yield d
    d.close()


def _adapter(db):
    from gateway.platforms.api_server import APIServerAdapter
    a = APIServerAdapter.__new__(APIServerAdapter)
    a._check_auth = lambda request: None
    a._ensure_session_db = lambda: db
    return a


def test_token_trends_endpoint(db):
    now = time.time()
    db.create_session(session_id="s1", source="cli")
    db.append_message(session_id="s1", role="tool", content="r",
                      token_count=pack_input_tokens(2000, 1800), timestamp=now - 30)
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(300, 40), timestamp=now - 29)
    req = SimpleNamespace(query={"window": "24h"})
    resp = asyncio.run(_adapter(db)._handle_analytics_token_trends(req))
    body = json.loads(resp.text)
    assert body["window"] == "24h" and body["bucket_seconds"] == 3600
    assert body["totals"]["input"] == 2000 and body["totals"]["output"] == 300
    assert body["cache_hit_rate"] == 90.0  # 1800/2000


def test_token_trends_custom_bucket(db):
    req = SimpleNamespace(query={"window": "1h", "bucket": "30"})
    resp = asyncio.run(_adapter(db)._handle_analytics_token_trends(req))
    body = json.loads(resp.text)
    assert body["bucket_seconds"] == 60  # clamped to min 60


def test_token_trends_bad_bucket(db):
    req = SimpleNamespace(query={"window": "1h", "bucket": "abc"})
    resp = asyncio.run(_adapter(db)._handle_analytics_token_trends(req))
    assert resp.status == 400
