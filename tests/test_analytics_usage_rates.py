"""Time-series aggregation + usage-rates analytics."""
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_token_codec import pack_input_tokens, pack_assistant_tokens
from agent.analytics import compute_usage_rates


@pytest.fixture()
def db():
    d = SessionDB(db_path=Path(tempfile.mkdtemp()) / "rates.db")
    yield d
    d.close()


def _seed_two_minutes(db):
    db.create_session(session_id="s1", source="cli")
    t0 = 1_000_000.0
    db.append_message(session_id="s1", role="tool", content="r",
                      token_count=pack_input_tokens(1000, 800), timestamp=t0 + 1)
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(200, 30), timestamp=t0 + 2)
    db.append_message(session_id="s1", role="tool", content="r",
                      token_count=pack_input_tokens(2000, 1900), timestamp=t0 + 65)
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(400, 0), timestamp=t0 + 66)
    return t0


def test_timeseries_buckets_and_decode(db):
    t0 = _seed_two_minutes(db)
    ts = db.get_message_token_timeseries(t0, t0 + 120, 60)
    assert len(ts) == 2
    assert ts[0]["requests"] == 1 and ts[0]["input"] == 1000 and ts[0]["cache_read"] == 800
    assert ts[1]["input"] == 2000 and ts[1]["output"] == 400
    # bucket boundaries align to multiples of 60.
    assert all(b["bucket_start"] % 60 == 0 for b in ts)


def test_timeseries_rejects_bad_bucket(db):
    with pytest.raises(ValueError):
        db.get_message_token_timeseries(0, 1, 0)


def test_compute_usage_rates_peaks_and_quota_pct():
    minutes = [
        {"requests": 1, "input": 1000, "output": 200},
        {"requests": 3, "input": 5000, "output": 900},
    ]
    quotas = [{"provider": "anthropic", "display": "Anthropic", "rpm": 50,
               "tpm_input": 40000, "tpm_output": 8000, "rpd": None, "tpd": None}]
    out = compute_usage_rates(minutes, {"requests": 4, "input": 6000, "output": 1100}, quotas)
    assert out["rpm"]["peak"] == 3
    assert out["tpm"]["peak"] == 5900  # 5000+900
    assert out["rpd"] == 4 and out["tpd"] == 7100
    pa = out["providers"][0]["pct_of_limit"]
    assert pa["rpm"] == round(100 * 3 / 50, 1)
    assert pa["tpm_input"] == round(100 * 5000 / 40000, 1)


def test_empty_is_zero():
    out = compute_usage_rates([], {}, [])
    assert out["rpm"]["peak"] == 0 and out["tpd"] == 0 and out["providers"] == []


import importlib.util as _ilu
_needs_aiohttp = pytest.mark.skipif(_ilu.find_spec("aiohttp") is None, reason="aiohttp not installed")


def _adapter(db):
    from gateway.platforms.api_server import APIServerAdapter
    a = APIServerAdapter.__new__(APIServerAdapter)
    a._check_auth = lambda request: None
    a._ensure_session_db = lambda: db
    return a


def _seed_recent(db):
    import time
    now = time.time()
    db.create_session(session_id="s1", source="cli")
    db.append_message(session_id="s1", role="tool", content="r",
                      token_count=pack_input_tokens(1000, 800), timestamp=now - 20)
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(200, 30), timestamp=now - 19)


@_needs_aiohttp
def test_usage_rates_endpoint(db):
    _seed_recent(db)
    req = SimpleNamespace(query={"window": "24h"})
    resp = asyncio.run(_adapter(db)._handle_analytics_usage_rates(req))
    body = json.loads(resp.text)
    assert body["window"] == "24h"
    assert body["rpm"]["peak"] == 1
    assert body["rpd"] == 1 and body["tpd"] == 1200
    assert "providers" in body


@_needs_aiohttp
def test_usage_rates_invalid_window(db):
    req = SimpleNamespace(query={"window": "bogus"})
    resp = asyncio.run(_adapter(db)._handle_analytics_usage_rates(req))
    assert resp.status == 400
