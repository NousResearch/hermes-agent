"""Provider quota reference + /api/analytics/provider-quotas endpoint."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.provider_quotas import (
    get_provider_quota,
    list_provider_quotas,
    normalize_provider,
)


def test_alias_normalization():
    assert normalize_provider("claude") == "anthropic"
    assert normalize_provider("Gemini") == "google"
    assert normalize_provider("AZURE_OPENAI") == "openai"
    assert normalize_provider("nope") is None
    assert normalize_provider(None) is None


def test_get_and_list():
    q = get_provider_quota("anthropic")
    assert q["provider"] == "anthropic"
    assert q["rpm"] == 50 and "source_url" in q and "as_of" in q
    assert get_provider_quota("unknownxyz") is None
    allq = list_provider_quotas()
    assert {p["provider"] for p in allq} >= {"anthropic", "openai", "google"}


import importlib.util as _ilu
_needs_aiohttp = pytest.mark.skipif(_ilu.find_spec("aiohttp") is None, reason="aiohttp not installed")


def _adapter():
    from gateway.platforms.api_server import APIServerAdapter
    a = APIServerAdapter.__new__(APIServerAdapter)
    a._check_auth = lambda request: None
    return a


def _req(provider=None):
    return SimpleNamespace(query=({"provider": provider} if provider else {}))


@_needs_aiohttp
def test_endpoint_all():
    resp = asyncio.run(_adapter()._handle_analytics_provider_quotas(_req()))
    body = json.loads(resp.text)
    assert body["object"] == "list"
    assert any(p["provider"] == "anthropic" for p in body["data"])


@_needs_aiohttp
def test_endpoint_filtered():
    resp = asyncio.run(_adapter()._handle_analytics_provider_quotas(_req("claude")))
    body = json.loads(resp.text)
    assert len(body["data"]) == 1
    assert body["data"][0]["provider"] == "anthropic"
