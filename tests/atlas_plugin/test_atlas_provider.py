"""Tests for the Atlas memory provider (army-of-one Plan 011-C.2).

Mocks httpx so no network/Atlas is required. Verifies ABC conformance,
read formatting, write payload shape, graceful degradation, and the
circuit breaker.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "memory" / "atlas" / "__init__.py"


def _load_provider_module():
    spec = importlib.util.spec_from_file_location("atlas_provider_under_test", _PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_provider_module()


@pytest.fixture
def provider(mod, monkeypatch):
    monkeypatch.setenv("ATLAS_BASE_URL", "http://atlas.test:8000")
    monkeypatch.setenv("ATLAS_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("ATLAS_AGENT_NAME", "hermes")
    p = mod.AtlasMemoryProvider()
    p.initialize(session_id="sess-1", hermes_home="/tmp", platform="cli")
    return p


def test_conforms_to_abc(mod):
    from agent.memory_provider import MemoryProvider
    assert issubclass(mod.AtlasMemoryProvider, MemoryProvider)


def test_name_and_tools(provider):
    assert provider.name == "atlas"
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert names == {"atlas_recall", "atlas_remember"}


def test_is_available_requires_base_url(mod, monkeypatch):
    monkeypatch.delenv("ATLAS_BASE_URL", raising=False)
    assert mod.AtlasMemoryProvider().is_available() is False
    monkeypatch.setenv("ATLAS_BASE_URL", "http://x:8000")
    assert mod.AtlasMemoryProvider().is_available() is True


def test_headers_include_bearer(provider):
    h = provider._headers()
    assert h["Authorization"] == "Bearer test-token"
    assert h["Content-Type"] == "application/json"


def test_format_facts(provider):
    facts = [
        {"key": "pref_response_style", "value": "concise", "life_context": "work"},
        {"key": "x", "value": "x", "life_context": None},  # key==value → no key prefix
        {"key": "empty", "value": "", "life_context": None},  # skipped
    ]
    out = provider._format_facts(facts)
    assert "- pref_response_style: concise [work]" in out
    assert "- x" in out
    assert "empty" not in out


def test_recall_tool_calls_read(provider, mod, monkeypatch):
    import httpx

    class FakeResp:
        def raise_for_status(self): ...
        def json(self):
            return [{"key": "k1", "value": "Blake prefers dark mode", "life_context": "work"}]

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResp()

    monkeypatch.setattr(httpx, "get", fake_get)
    result = json.loads(provider.handle_tool_call("atlas_recall", {}))
    assert "Blake prefers dark mode" in result["result"]
    assert captured["url"].endswith("/v1/memory/hermes/read")
    assert captured["params"]["agent"] == "hermes"


def test_remember_tool_posts_write(provider, mod, monkeypatch):
    import httpx

    class FakeResp:
        def raise_for_status(self): ...

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    result = json.loads(provider.handle_tool_call(
        "atlas_remember", {"content": "Likes espresso", "target": "user", "life_context": "personal"}
    ))
    assert result["result"] == "Fact stored in Atlas."
    assert captured["url"].endswith("/v1/memory/hermes/write")
    assert captured["body"]["content"] == "Likes espresso"
    assert captured["body"]["target"] == "user"
    assert captured["body"]["action"] == "add"
    assert captured["body"]["life_context"] == "personal"
    assert captured["body"]["agent"] == "hermes"


def test_remember_requires_content(provider):
    result = provider.handle_tool_call("atlas_remember", {})
    # tool_error returns a JSON string with an error
    assert "content" in result.lower() or "error" in result.lower()


def test_graceful_degradation_on_http_error(provider, mod, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("atlas down")

    monkeypatch.setattr(httpx, "get", boom)
    # Should not raise — returns an error JSON, records failure
    result = provider.handle_tool_call("atlas_recall", {})
    assert "error" in result.lower() or "failed" in result.lower()


def test_circuit_breaker_opens_after_threshold(provider, mod):
    # Simulate consecutive failures
    for _ in range(mod._BREAKER_THRESHOLD):
        provider._record_failure()
    assert provider._is_breaker_open() is True
    # Recall short-circuits with the breaker message
    result = json.loads(provider.handle_tool_call("atlas_recall", {}))
    assert "unavailable" in result["error"].lower()


def test_sync_turn_is_noop(provider):
    # Atlas does not do turn extraction — must return without error
    assert provider.sync_turn("hi", "hello") is None


def test_unknown_tool(provider):
    result = provider.handle_tool_call("atlas_bogus", {})
    assert "unknown tool" in result.lower() or "error" in result.lower()
