"""Tests for the Atlas memory provider `atlas_open_contradictions` tool (Plan 025-E).

Wraps Atlas `GET /v1/contradictions?status=open` (army-of-one
`backend/src/atlas/api/contradictions_routes.py`). All network calls are mocked.

The tool exists to let Hermes surface pending memory conflicts to Blake
*before* answering a question that depends on a fact the scanner has
flagged. The advisory `llm_verdict` + `llm_confidence` are carried verbatim
so the chat model can decide whether to ask for a reconciliation.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "memory"
    / "atlas"
    / "__init__.py"
)


def _load_provider_module():
    spec = importlib.util.spec_from_file_location(
        "atlas_provider_open_contradictions_under_test", _PLUGIN
    )
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
    p.initialize(session_id="sess-open-contras", hermes_home="/tmp", platform="cli")
    return p


# ─────────────────────────────────────────────────────────────────────
# Schema / registration
# ─────────────────────────────────────────────────────────────────────


def test_open_contradictions_tool_in_schema_list(provider):
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert "atlas_open_contradictions" in names


def test_open_contradictions_schema_no_required_args(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    tool = schemas["atlas_open_contradictions"]
    assert tool["parameters"]["required"] == []
    props = tool["parameters"]["properties"]
    assert "confidence_min" in props
    assert "limit" in props


def test_open_contradictions_schema_steers_to_reconcile_questions(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    desc = schemas["atlas_open_contradictions"]["description"].lower()
    assert "contradiction" in desc
    # Make sure the description steers toward "pending / unresolved" usage
    assert "open" in desc or "unresolved" in desc or "reconcile" in desc


# ─────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────


_FAKE_ROWS = [
    {
        "id": 1,
        "subject_iri": "urn:work:x",
        "predicate": "https://atlas.blakeaber.dev/ns/pricedAt",
        "object_a": "$29",
        "object_b": "$49",
        "source_a": "urn:src:a",
        "source_b": "urn:src:b",
        "kind": "structural",
        "status": "open",
        "llm_verdict": "contradiction",
        "llm_confidence": 0.85,
        "llm_rationale": "Same product version.",
        "annotated_by": None,
        "annotated_at": None,
        "annotation_rationale": None,
        "detected_at": "2026-05-31T00:00:00Z",
    },
    {
        "id": 2,
        "subject_iri": "urn:work:y",
        "predicate": "https://atlas.blakeaber.dev/ns/role",
        "object_a": "engineer",
        "object_b": "manager",
        "source_a": "urn:src:c",
        "source_b": "urn:src:d",
        "kind": "structural",
        "status": "open",
        "llm_verdict": "ambiguous",
        "llm_confidence": 0.4,
        "llm_rationale": "Could be different time periods.",
        "annotated_by": None,
        "annotated_at": None,
        "annotation_rationale": None,
        "detected_at": "2026-05-31T00:00:00Z",
    },
]


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_open_contradictions_dispatch_filters_by_confidence(provider, monkeypatch):
    captured = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["params"] = kwargs.get("params", {})
        return _Resp(_FAKE_ROWS)

    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get)

    raw = provider.handle_tool_call("atlas_open_contradictions", {})
    result = json.loads(raw)
    # Default floor 0.6 → only the 0.85 row survives.
    assert result["count"] == 1
    assert result["confidence_min"] == 0.6
    assert result["result"][0]["id"] == 1
    # URL + auth wiring
    assert captured["url"].endswith("/v1/contradictions")
    assert captured["params"]["status"] == "open"
    assert captured["headers"].get("Authorization") == "Bearer test-token"


def test_open_contradictions_override_confidence_floor(provider, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: _Resp(_FAKE_ROWS)
    )

    raw = provider.handle_tool_call(
        "atlas_open_contradictions", {"confidence_min": 0.3}
    )
    result = json.loads(raw)
    # Lower floor → both rows survive.
    assert result["count"] == 2
    assert result["confidence_min"] == 0.3


def test_open_contradictions_limit_passed_through(provider, monkeypatch):
    captured = {}

    def _fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params", {})
        return _Resp([])

    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get)
    provider.handle_tool_call("atlas_open_contradictions", {"limit": 10})
    assert captured["params"]["limit"] == 10


def test_open_contradictions_limit_clamps_to_max(provider, monkeypatch):
    captured = {}

    def _fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params", {})
        return _Resp([])

    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get)
    provider.handle_tool_call(
        "atlas_open_contradictions", {"limit": 99999}
    )
    assert captured["params"]["limit"] == 500


def test_open_contradictions_network_error_returns_friendly(provider, monkeypatch):
    import httpx

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    raw = provider.handle_tool_call("atlas_open_contradictions", {})
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "open-contradictions" in parsed["error"].lower()


def test_open_contradictions_handles_non_list_response(provider, monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: _Resp({"oops": "not a list"})
    )
    raw = provider.handle_tool_call("atlas_open_contradictions", {})
    result = json.loads(raw)
    assert result["count"] == 0
    assert result["result"] == []
