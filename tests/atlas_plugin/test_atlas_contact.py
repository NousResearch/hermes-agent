"""Tests for the Atlas memory provider `atlas_contact` tool (Plan 025-C).

Wraps Atlas `GET /v1/contact/{iri:path}/context` (army-of-one
`backend/src/atlas/api/contact_routes.py`). All network calls are mocked.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "memory" / "atlas" / "__init__.py"


def _load_provider_module():
    spec = importlib.util.spec_from_file_location("atlas_provider_contact_under_test", _PLUGIN)
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
    p.initialize(session_id="sess-contact", hermes_home="/tmp", platform="cli")
    return p


# ─────────────────────────────────────────────────────────────────────
# Schema / registration
# ─────────────────────────────────────────────────────────────────────


def test_contact_tool_in_schema_list(provider):
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert "atlas_contact" in names


def test_contact_schema_requires_person_iri(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    contact = schemas["atlas_contact"]
    assert contact["parameters"]["required"] == ["person_iri"]
    props = contact["parameters"]["properties"]
    assert "person_iri" in props
    assert "recency" in props


def test_contact_schema_steers_to_brief_questions(provider):
    schemas = {t["name"]: t for t in provider.get_tool_schemas()}
    desc = schemas["atlas_contact"]["description"].lower()
    # The description must steer the model toward person-context questions
    assert "person" in desc
    assert "commitments" in desc or "preferences" in desc
    # Cold-corpus contract must be visible to the model
    assert "empty" in desc


# ─────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────


def test_contact_tool_call_returns_json_payload(provider, monkeypatch):
    fake_payload = {
        "person_iri": "https://atlas.blakeaber.dev/person/jane",
        "canonical_name": "Jane Doe",
        "org": "acme",
        "recent_interactions": [],
        "warm_intro_paths": [],
        "last_touch": None,
        "events": [{"event_iri": "https://atlas.blakeaber.dev/event/x",
                    "name": "Jane sync", "start": None, "end": None, "role": "attendee"}],
        "commitments": [],
        "preferences": [],
        "contradictions": [],
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fake_payload

    captured = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["params"] = kwargs.get("params", {})
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get)

    result_json = provider.handle_tool_call(
        "atlas_contact",
        {"person_iri": "https://atlas.blakeaber.dev/person/jane"},
    )
    result = json.loads(result_json)
    assert result["canonical_name"] == "Jane Doe"
    assert result["events"][0]["role"] == "attendee"
    # URL is built off ATLAS_BASE_URL + /v1/contact/<iri>/context
    assert "/v1/contact/" in captured["url"]
    assert captured["url"].endswith("/context")
    # Bearer is forwarded
    assert captured["headers"].get("Authorization") == "Bearer test-token"


def test_contact_tool_call_missing_iri_returns_error(provider):
    result = provider.handle_tool_call("atlas_contact", {})
    # tool_error returns a JSON string with an "error" key
    parsed = json.loads(result)
    assert "error" in parsed
    assert "person_iri" in parsed["error"]


def test_contact_tool_call_network_error_returns_friendly(provider, monkeypatch):
    """A failing Atlas call must NOT raise; it must return a tool_error JSON."""
    import httpx

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    result = provider.handle_tool_call(
        "atlas_contact",
        {"person_iri": "https://atlas.blakeaber.dev/person/jane"},
    )
    parsed = json.loads(result)
    assert "error" in parsed
    assert "contact" in parsed["error"].lower() or "failed" in parsed["error"].lower()
