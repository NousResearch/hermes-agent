"""Synthetic Responses -> real compression/aux accounting -> a private SessionDB.

The effort regression fixture replaces only HTTP transport and blocks sockets.
Rates below are synthetic arithmetic fixtures, never claims about provider prices.
"""
import asyncio
from decimal import Decimal
import json

import httpx
import pytest

import agent.auxiliary_client as aux
from agent.aux_accounting import reset_accounting_context, set_accounting_context
from agent.usage_pricing import PricingEntry, normalize_usage
from hermes_state import SessionDB
from tests.agent.test_compressor_runtime_handoff import MODEL, URL, compress, response_bytes, wire


def _cache_details_reported(usage) -> bool:
    """Cache-read telemetry is trustworthy iff the adapter carried input details with a
    provider-REPORTED cached_tokens value; unset details, explicit null, or a details dict
    without the key (SDK: cached_tokens=None) must not read as a real zero hit."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return False
    value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return value is not None


@pytest.fixture
def accounting(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "usage.db")
    db.create_session("usage-session", source="cli", model=MODEL)
    # Main-loop totals are deliberately nonzero; auxiliary writes must not add to them.
    db.update_token_counts("usage-session", input_tokens=10, output_tokens=2,
                           model=MODEL, billing_provider="custom", api_call_count=1)
    token = set_accounting_context(db, "usage-session")
    rates = PricingEntry(input_cost_per_million=Decimal("2"),
                         output_cost_per_million=Decimal("10"),
                         cache_read_cost_per_million=Decimal("0.5"),
                         cache_write_cost_per_million=Decimal("2.5"), source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: rates)
    yield db
    reset_accounting_context(token)
    db.close()


def rows(db, task="compression"):
    with db._lock:
        return [dict(r) for r in db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id=? AND task=?",
            ("usage-session", task))]


def replace_usage(hook, usage, terminal=True):
    def response(request, body):
        events = [json.loads(part.split("data: ", 1)[1])
                  for part in response_bytes(body).decode().strip().split("\n\n")]
        for event in events:
            if "response" in event:
                event["response"]["usage"] = usage
        if not terminal:
            events = [event for event in events if event["type"] != "response.completed"]
        data = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=data.encode(), request=request)
    hook[0] = response


CASES = [
    pytest.param({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600}}, 400, 600, 0, True, id="partial"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 1000}}, 0, 1000, 0, True, id="full"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0}}, 1000, 0, 0, True, id="zero"),
    pytest.param({"input_tokens": 0, "input_tokens_details": {"cached_tokens": 600}}, 0, 600, 0, True, id="input-zero"),
    pytest.param({"input_tokens": None, "input_tokens_details": {"cached_tokens": 600}}, 0, 600, 0, True, id="input-null"),
    pytest.param({"input_tokens_details": {"cached_tokens": 600}}, 0, 600, 0, True, id="input-missing"),
    pytest.param({"input_tokens": 1000}, 1000, 0, 0, False, id="details-missing"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": None}, 1000, 0, 0, False, id="details-null"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": {}}, 1000, 0, 0, False, id="cache-missing"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100}}, 300, 600, 100, True, id="cache-write"),
    pytest.param({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600, "cache_creation_tokens": 100}}, 300, 600, 100, True, id="legacy-write"),
]


@pytest.mark.parametrize("fields,fresh,cached,written,known", CASES)
def test_compression_persists_usage(wire, accounting, monkeypatch, fields, fresh, cached, written, known):
    make, requests, _, hook = wire
    seen = []
    original = aux._CodexCompletionsAdapter.create
    def observe(self, **kwargs):
        result = original(self, **kwargs)
        seen.append(result.usage)
        return result
    monkeypatch.setattr(aux._CodexCompletionsAdapter, "create", observe)
    raw = {"output_tokens": 7, "total_tokens": 1007,
           "output_tokens_details": {"reasoning_tokens": 3}, **fields}
    replace_usage(hook, raw)
    agent = make(sid="usage-session")
    compressed, _ = compress(agent)
    assert len(compressed) < 28 and len(requests) == 1
    assert len(seen) == 1 and seen[0].total_tokens == raw["total_tokens"]
    assert _cache_details_reported(seen[0]) == known
    records = rows(accounting)
    assert len(records) == 1
    row = records[0]
    assert (row["input_tokens"], row["cache_read_tokens"], row["cache_write_tokens"],
            row["output_tokens"], row["reasoning_tokens"], row["api_call_count"]) == (fresh, cached, written, 7, 3, 1)
    assert (row["model"], row["billing_provider"], row["billing_base_url"].rstrip("/")) == (MODEL, "custom", URL)
    assert row["estimated_cost_usd"] == pytest.approx((fresh * 2 + cached * .5 + written * 2.5 + 7 * 10) / 1_000_000)
    main = rows(accounting, "")[0]
    assert (main["input_tokens"], main["output_tokens"], main["api_call_count"]) == (10, 2, 1)


@pytest.mark.parametrize("async_mode", [False, True])
def test_aux_call_preserves_totals_presence_and_single_accounting(wire, accounting, async_mode):
    make, requests, _, hook = wire
    raw = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600},
           "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 1007}
    replace_usage(hook, raw)
    agent = make(sid="usage-session")
    args = dict(task="compression", messages=[{"role": "user", "content": "Synthetic."}],
                main_runtime=agent._current_main_runtime())
    result = asyncio.run(aux.async_call_llm(**args)) if async_mode else aux.call_llm(**args)
    assert len(requests) == 1
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (1000, 7, 1007)
    canonical = normalize_usage(result.usage, api_mode="chat_completions", provider="custom")
    assert (canonical.input_tokens, canonical.cache_read_tokens, canonical.output_tokens, canonical.reasoning_tokens) == (400, 600, 7, 3)
    assert _cache_details_reported(result.usage)
    assert rows(accounting)[0]["api_call_count"] == 1
    assert rows(accounting)[0]["estimated_cost_usd"] == pytest.approx(.00117)


def test_pooled_repeated_compression_accumulates_once(wire, accounting):
    make, requests, _, _ = wire
    agent = make(sid="usage-session")
    for pooled in (False, True):
        compressed, _ = compress(agent, pooled=pooled)
        assert len(compressed) < 28
    assert len(requests) == 2
    row = rows(accounting)[0]
    assert (row["input_tokens"], row["cache_read_tokens"], row["output_tokens"],
            row["reasoning_tokens"], row["api_call_count"]) == (800, 1200, 14, 6, 2)
    assert row["estimated_cost_usd"] == pytest.approx(.00234)


@pytest.mark.parametrize("terminal", [False, True])
def test_unknown_usage_never_creates_an_aux_row(wire, accounting, terminal):
    make, requests, _, hook = wire
    replace_usage(hook, None, terminal=terminal)
    agent = make(sid="usage-session")
    args = dict(task="compression", messages=[{"role": "user", "content": "Synthetic."}],
                main_runtime=agent._current_main_runtime())
    # The existing stream consumer permits truncated text, but its usage is unknown.
    assert aux.call_llm(**args).usage is None
    assert len(requests) == 1
    assert rows(accounting) == []


def test_empty_stream_fails_without_invented_usage(wire, accounting):
    make, requests, _, hook = wire
    hook[0] = lambda request, body: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"", request=request)
    agent = make(sid="usage-session")
    with pytest.raises(RuntimeError, match="terminal response"):
        aux.call_llm(task="compression", messages=[{"role": "user", "content": "Synthetic."}],
                     main_runtime=agent._current_main_runtime())
    assert len(requests) == 1 and rows(accounting) == []
