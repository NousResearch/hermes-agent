"""Tests for the Atlas memory provider `atlas_ask` tool (Plan 025-B).

Wraps Atlas `POST /v1/ask` (army-of-one
`backend/src/atlas/api/ask_routes.py` — AskRequest / AskResponse). All
network calls are mocked; no live Atlas required.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "memory" / "atlas" / "__init__.py"


def _load_provider_module():
    spec = importlib.util.spec_from_file_location("atlas_provider_ask_under_test", _PLUGIN)
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
    p.initialize(session_id="sess-ask", hermes_home="/tmp", platform="cli")
    return p


# ---------------------------------------------------------------------------
# Schema / registration
# ---------------------------------------------------------------------------


def test_ask_tool_in_schema_list(provider):
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert "atlas_ask" in names
    assert names == {"atlas_recall", "atlas_remember", "atlas_ask"}


def test_ask_schema_requires_question(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    ask = schemas["atlas_ask"]
    assert ask["parameters"]["required"] == ["question"]
    props = ask["parameters"]["properties"]
    for key in ("question", "life_context", "intent_hint", "max_chunks"):
        assert key in props


def test_ask_schema_steers_to_recall(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    desc = schemas["atlas_ask"]["description"].lower()
    # Tool description must steer toward recall (D4 §Risk 5) and preserve
    # citation markers so Blake can audit.
    assert "history" in desc or "commitment" in desc or "recall" in desc
    assert "cite" in desc


def test_register_exposes_provider(mod):
    captured: dict[str, object] = {}

    class FakeCtx:
        def register_memory_provider(self, p):
            captured["provider"] = p

    mod.register(FakeCtx())
    p = captured["provider"]
    assert p.name == "atlas"
    names = {t["name"] for t in p.get_tool_schemas()}
    assert "atlas_ask" in names


# ---------------------------------------------------------------------------
# Tool call behavior — mocked HTTP
# ---------------------------------------------------------------------------


def test_ask_returns_cited_payload_verbatim(provider, mod, monkeypatch):
    """Atlas's AskResponse is returned verbatim so [cite:...] markers survive."""
    import httpx

    atlas_response = {
        "question": "what's my last Pipedrive activity for Apex Capital?",
        "intent": "lookup",
        "answer": "Last contact was a call with Greg on 2026-05-28 [cite:chunk-abc123].",
        "citations": [
            {"chunk_id": "chunk-abc123", "source_iri": "urn:pipedrive:activity:42",
             "snippet": "Call with Greg re: pipeline review."},
        ],
        "anchors": ["urn:atlas:contact:greg"],
        "temporal": None,
        "confidence": 0.84,
        "latency_ms": 412.0,
        "usd": 0.0021,
    }

    captured: dict[str, object] = {}

    class FakeResp:
        def raise_for_status(self): ...
        def json(self):
            return atlas_response

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    raw = provider.handle_tool_call("atlas_ask", {
        "question": "what's my last Pipedrive activity for Apex Capital?",
    })
    result = json.loads(raw)

    # Verbatim passthrough — citation markers preserved
    assert result == atlas_response
    assert "[cite:chunk-abc123]" in result["answer"]

    # Wire contract
    assert captured["url"].endswith("/v1/ask")
    assert captured["body"]["question"] == "what's my last Pipedrive activity for Apex Capital?"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["timeout"] == mod._ASK_TIMEOUT_SECS
    assert captured["timeout"] >= 10.0


def test_ask_forwards_intent_hint(provider, monkeypatch):
    import httpx

    captured: dict[str, object] = {}

    class FakeResp:
        def raise_for_status(self): ...
        def json(self):
            return {"question": "q", "intent": "lookup", "answer": "a",
                    "citations": [], "anchors": [], "temporal": None,
                    "confidence": 0.0, "latency_ms": 0.0, "usd": 0.0}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider.handle_tool_call("atlas_ask", {
        "question": "q",
        "intent_hint": "lookup",
        "life_context": "work",
        "max_chunks": 7,
    })

    # All three hints fold into intent_hint as a composite signal so
    # Atlas's strict extra="forbid" AskRequest still accepts the payload.
    hint = captured["body"]["intent_hint"]
    assert "lookup" in hint
    assert "life_context:work" in hint
    assert "max_chunks:7" in hint


def test_ask_requires_question(provider):
    result = provider.handle_tool_call("atlas_ask", {})
    assert "question" in result.lower() or "error" in result.lower()
    result = provider.handle_tool_call("atlas_ask", {"question": "   "})
    assert "question" in result.lower() or "error" in result.lower()


def test_ask_500_records_failure_and_returns_error(provider, monkeypatch):
    import httpx

    class FakeResp:
        status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500 Internal Server Error", request=None, response=self,
            )
        def json(self):
            return {}

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)

    before = provider._consecutive_failures
    result = provider.handle_tool_call("atlas_ask", {"question": "q"})
    assert "error" in result.lower() or "failed" in result.lower()
    assert provider._consecutive_failures == before + 1


def test_ask_timeout_engages_breaker(provider, mod, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ReadTimeout("atlas timed out")

    monkeypatch.setattr(httpx, "post", boom)

    # Hammer until the breaker trips
    for _ in range(mod._BREAKER_THRESHOLD):
        result = provider.handle_tool_call("atlas_ask", {"question": "q"})
        assert "error" in result.lower() or "failed" in result.lower()

    assert provider._is_breaker_open() is True

    # Once the breaker is open, the tool short-circuits without calling httpx
    short = json.loads(provider.handle_tool_call("atlas_ask", {"question": "q"}))
    assert "unavailable" in short["error"].lower()


def test_ask_500_eventually_engages_breaker(provider, mod, monkeypatch):
    import httpx

    class FakeResp:
        status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500", request=None, response=self,
            )
        def json(self):
            return {}

    def fake_post(*a, **k):
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    for _ in range(mod._BREAKER_THRESHOLD):
        provider.handle_tool_call("atlas_ask", {"question": "q"})
    assert provider._is_breaker_open() is True
