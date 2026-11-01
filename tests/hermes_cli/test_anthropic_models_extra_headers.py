"""Anthropic /v1/models discovery must merge the routed provider's ``extra_headers``.

Identity-linked keys that are not scoped to a single workspace get a 400 from every
Anthropic endpoint unless ``anthropic-workspace-id`` rides along. The inference
client builders already merge the route's ``extra_headers``; discovery used to
build its own header dict without them, so the 400 was swallowed (``logger.debug``)
and the picker silently fell back to the static catalog. Covers the curated-id
fixes in the same report: ``claude-fable-5.1`` never existed (the API id is
``claude-fable-5-1``) and ``claude-sonnet-5-5`` / ``claude-haiku-5-5`` were absent.
"""

from __future__ import annotations

import hermes_cli.models as hm
from hermes_cli.models_catalog_static import _PROVIDER_MODELS

WORKSPACE_HEADERS = {"anthropic-workspace-id": "wrkspc_123"}
TEST_KEY = "unit-test-key"  # shapeless fixture, only ever asserted back verbatim


def _capture_get_json(captured: dict):
    def fake_get_json(url, timeout=None, headers=None):
        captured["url"] = url
        captured["headers"] = dict(headers or {})
        return {"data": [{"id": "claude-fable-5-1"}, {"id": "claude-sonnet-5-5"}], "has_more": False}

    return fake_get_json


def _providers_config(base_url: str) -> dict:
    return {"providers": {"claude-api": {"base_url": base_url, "extra_headers": WORKSPACE_HEADERS}}}


def test_route_extra_headers_reach_models_request(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(hm, "_get_json", _capture_get_json(captured))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _providers_config("https://api.anthropic.example"))

    got = hm._fetch_anthropic_models(base_url="https://api.anthropic.example", api_key=TEST_KEY)

    # The picker's tier sort (opus → sonnet → haiku, alphabetical within tier) orders the result.
    assert got == ["claude-sonnet-5-5", "claude-fable-5-1"]
    assert captured["headers"]["anthropic-workspace-id"] == "wrkspc_123"
    # The routed headers ride along; the request's own auth/version headers stay intact.
    assert captured["headers"]["x-api-key"] == TEST_KEY
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


def test_default_endpoint_routes_extra_headers(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(hm, "_get_json", _capture_get_json(captured))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _providers_config("https://api.anthropic.com"))

    got = hm._fetch_anthropic_models(api_key=TEST_KEY)

    assert got is not None
    assert captured["url"].startswith("https://api.anthropic.com/v1/models")
    assert captured["headers"]["anthropic-workspace-id"] == "wrkspc_123"


def test_extra_headers_for_another_route_do_not_leak(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(hm, "_get_json", _capture_get_json(captured))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _providers_config("https://other.example"))

    got = hm._fetch_anthropic_models(base_url="https://api.anthropic.example", api_key=TEST_KEY)

    assert got is not None
    assert "anthropic-workspace-id" not in captured["headers"]


def test_curated_anthropic_catalog_ids():
    catalog = _PROVIDER_MODELS["anthropic"]
    assert "claude-fable-5-1" in catalog
    assert "claude-fable-5.1" not in catalog
    assert "claude-sonnet-5-5" in catalog
    assert "claude-haiku-5-5" in catalog
