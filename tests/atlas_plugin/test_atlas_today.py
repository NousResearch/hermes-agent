"""Tests for the Atlas memory provider `_today()` helper (Phase 2a).

Wraps Atlas `GET /v1/today` (army-of-one
`backend/src/atlas/api/ask_routes.py:get_today`), the "cockpit" endpoint
that returns ``upcoming_meetings`` + ``overdue_emails`` straight from the
graph. Used by the Hermes ``/daily`` brief to source calendar + inbox.
All network calls are mocked.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "memory" / "atlas" / "__init__.py"


def _load_provider_module():
    spec = importlib.util.spec_from_file_location("atlas_provider_today_under_test", _PLUGIN)
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
    p.initialize(session_id="sess-today", hermes_home="/tmp", platform="cli")
    return p


def test_today_calls_v1_today_with_bearer(provider, monkeypatch):
    fake_payload = {
        "upcoming_meetings": [
            {
                "iri": "https://atlas.blakeaber.dev/event/abc",
                "kind": "calendar_event",
                "canonical_name": "Anthropic intro call",
                "event_time": "2026-06-26T10:00:00+00:00",
                "summary": None,
            }
        ],
        "overdue_emails": [
            {
                "iri": "https://atlas.blakeaber.dev/email/xyz",
                "kind": "email",
                "canonical_name": "Re: term sheet",
                "event_time": "2026-06-26T08:30:00+00:00",
                "summary": None,
            }
        ],
        "pending_drafts": [],
        "agent_runs": [],
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fake_payload

    captured: dict = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get)

    result = provider._today()
    assert result["upcoming_meetings"][0]["canonical_name"] == "Anthropic intro call"
    assert result["overdue_emails"][0]["canonical_name"] == "Re: term sheet"
    # URL is built off ATLAS_BASE_URL + /v1/today
    assert captured["url"].endswith("/v1/today")
    # Bearer is forwarded
    assert captured["headers"].get("Authorization") == "Bearer test-token"


def test_today_raises_on_http_error(provider, monkeypatch):
    """``_today`` propagates transport errors (the daily fetcher catches them)."""
    import httpx

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    with pytest.raises(Exception):
        provider._today()
