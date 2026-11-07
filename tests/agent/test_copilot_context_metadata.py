"""Advertised per-tier input budgets, tier-scoped limit learning, and the request
extra_body fragment — never a fixed 1M and never a cross-tier cache key."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.copilot_context import (
    context_cache_url, context_tiers, fetch_context_catalog, tier_extra_body,
)
from agent.model_metadata import get_cached_context_length, save_context_length
from agent.turn_overflow import _Recovery, _adopt_provider_context_limit


def test_tier_budgets_come_from_pricing_not_total_window():
    row = {"capabilities": {"limits": {"max_prompt_tokens": 999999}}, "billing": {"tokenPrices": {
        "maxPromptTokens": 272000, "longContext": {"maxPromptTokens": 872000}}}}
    assert context_tiers(row) == {"default": 272000, "long_context": 872000}
    # capabilities.limits.max_prompt_tokens carries the LONG-tier budget in the live catalog
    # (936k/922k/872k observed vs billing's 200k-272k default): it must backfill long_context
    # only, never advertise an oversized default tier.
    assert context_tiers({"capabilities": {"limits": {"max_prompt_tokens": 123000}}}) == {"long_context": 123000}
    assert context_tiers({}) == {}


def test_headless_catalog_uses_sdk_framing_without_prompt_or_shared_config(tmp_path):
    server = tmp_path / "headless.py"
    server.write_text(
        "import sys, json, os\n"
        "assert '--headless' in sys.argv and '--acp' not in sys.argv\n"
        "assert os.environ['COPILOT_SDK_AUTH_TOKEN'] == 'synthetic-token'\n"
        "assert os.path.isdir(os.environ['COPILOT_HOME'])\n"
        "def read():\n"
        "    header = sys.stdin.buffer.readline()\n"
        "    if not header: return None\n"
        "    n = int(header.split(b':')[1]); assert sys.stdin.buffer.readline() == b'\\r\\n'\n"
        "    return json.loads(sys.stdin.buffer.read(n))\n"
        "while r := read():\n"
        "    assert r['method'] in ('connect', 'models.list')\n"
        "    result = {'protocolVersion': 2} if r['method'] == 'connect' else {'models': [\n"
        "        {'id': 'fixture', 'billing': {'tokenPrices': {\n"
        "            'maxPromptTokens': 123000, 'longContext': {'maxPromptTokens': 812000}}}}]}\n"
        "    b = json.dumps({'jsonrpc': '2.0', 'id': r['id'], 'result': result}).encode()\n"
        "    sys.stdout.buffer.write(b'Content-Length: ' + str(len(b)).encode() + b'\\r\\n\\r\\n' + b)\n"
        "    sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    rows = fetch_context_catalog(command=sys.executable, prefix_args=[str(server)],
                                 token="synthetic-token", timeout=5)
    assert rows[0]["id"] == "fixture"


def _tier_agent(tier="long_context", budget=812000, **extra):
    return SimpleNamespace(model="fixture", provider="copilot-acp", base_url="acp://copilot",
        request_overrides={"extra_body": {"copilot_context_tier": tier, "copilot_context_budget": budget}},
        **extra)


def test_overflow_persists_limit_only_for_the_selected_tier(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _tier_agent(api_mode="chat", _buffer_vprint=lambda *a: None,
        context_compressor=SimpleNamespace(update_model=Mock()))
    save_context_length(agent.model, agent.base_url, 123000)
    error = ("Request exceeds the model's prompt budget after reserving output "
             "(803060 tokens; limit 700000). Reduce context.")
    assert _adopt_provider_context_limit(SimpleNamespace(agent=agent), error, 812000) == 700000
    assert get_cached_context_length(agent.model, agent.base_url) == 123000
    assert get_cached_context_length(agent.model, context_cache_url(agent)) == 700000


def test_tier_extra_body_adds_budget_from_cached_catalog(monkeypatch):
    from agent import copilot_context as cc
    calls = []
    monkeypatch.setattr(cc, "_copilot_creds", lambda: {"command": "copilot", "api_key": "t"})
    monkeypatch.setattr(cc, "fetch_context_catalog",
                        lambda **kw: calls.append(kw) or [{"id": "fixture", "billing": {"tokenPrices": {
                            "maxPromptTokens": 272000, "longContext": {"maxPromptTokens": 872000}}}}])
    cc._TIER_BUDGET_CACHE.clear()
    assert cc.tier_extra_body("long_context", "fixture") == {
        "copilot_context_tier": "long_context", "copilot_context_budget": 872000}
    assert cc.tier_extra_body("default", "fixture") == {
        "copilot_context_tier": "default", "copilot_context_budget": 272000}
    assert len(calls) == 1  # second lookup served from the process cache
    assert cc.tier_extra_body(None, "fixture") == {}
    assert cc.tier_extra_body("bogus", "fixture") == {}


def test_tier_extra_body_survives_catalog_failure_with_negative_cache(monkeypatch):
    from agent import copilot_context as cc
    calls = []
    def boom(**kw):
        calls.append(kw)
        raise RuntimeError("catalog down")
    monkeypatch.setattr(cc, "_copilot_creds", lambda: {"command": "copilot", "api_key": "t"})
    monkeypatch.setattr(cc, "fetch_context_catalog", boom)
    cc._TIER_BUDGET_CACHE.clear()
    assert cc.tier_extra_body("long_context", "fixture") == {"copilot_context_tier": "long_context"}
    assert cc.tier_extra_body("long_context", "fixture") == {"copilot_context_tier": "long_context"}
    assert len(calls) == 1  # negative cache: no retry inside the TTL


def test_tier_extra_body_without_credentials_never_spawns(monkeypatch):
    from agent import copilot_context as cc
    monkeypatch.setattr(cc, "_copilot_creds", dict)
    def forbidden(**kw):
        raise AssertionError("must not spawn without credentials")
    monkeypatch.setattr(cc, "fetch_context_catalog", forbidden)
    cc._TIER_BUDGET_CACHE.clear()
    assert cc.tier_extra_body("long_context", "fixture") == {"copilot_context_tier": "long_context"}
