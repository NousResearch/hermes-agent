"""Regression tests for the custom-endpoint slow-catalog budget on #134735.

A ``custom:`` provider whose authed ``/v1/models`` takes ~8s TTFB (per-key
entitlement catalog built server-side) deterministically vanished from every
picker surface and failed add-model validation: each entry point held a 5s
budget, the ``provider_models_cache.json`` row was never written, and the
cache-only GUI reads could never see the endpoint. The full discovery path now
uses ``CUSTOM_ENDPOINT_PROBE_TIMEOUT`` (15s); the picker's fast first paint
stays at 1.5s.
"""

import pytest

from hermes_cli.models import CUSTOM_ENDPOINT_PROBE_TIMEOUT
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models_validate import validate_requested_model


@pytest.fixture(autouse=True)
def _no_builtin_catalog_fetches(monkeypatch):
    """Keep the row builder independent of provider credentials and network."""
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.models_local.fetch_ollama_local_models", lambda *_a, **_kw: None)


def _current_custom_rows(monkeypatch, fake_live, **extra):
    monkeypatch.setattr("hermes_cli.model_switch_providers._fetch_picker_live_models", fake_live)
    return list_authenticated_providers(
        current_provider="custom", current_base_url="http://127.0.0.1:9999/v1",
        current_model="kept-model", probe_custom_providers=False,
        probe_current_custom_provider=True, for_picker=True, **extra)


def _custom_row(rows):
    return next(r for r in rows if r["slug"] == "custom")


def test_full_budget_covers_slow_catalogs(monkeypatch):
    """A catalog answering past the old 5s full budget (~8s TTFB) stays discoverable."""

    def _fake_live(_api_key, _api_url, _native_provider, _preserve, headers=None,
                   timeout=5.0, api_mode=None, **_kw):
        return ["slow-catalog-model"] if timeout >= CUSTOM_ENDPOINT_PROBE_TIMEOUT else None

    row = _custom_row(_current_custom_rows(monkeypatch, _fake_live, fast_custom_probe=False))

    assert "slow-catalog-model" in row["models"]


def test_fast_probe_budget_stays_snappy(monkeypatch):
    """The picker's fast first paint keeps its 1.5s budget (slow catalogs warm via the full path)."""
    seen = {}

    def _fake_live(_api_key, _api_url, _native_provider, _preserve, headers=None,
                   timeout=5.0, api_mode=None, **_kw):
        seen["timeout"] = timeout
        return None

    row = _custom_row(_current_custom_rows(monkeypatch, _fake_live, fast_custom_probe=True))

    assert seen["timeout"] == 1.5
    assert row["models"] == ["kept-model"]


def test_add_model_validation_uses_the_full_budget(monkeypatch):
    """``Add custom model`` validation must not inherit probe_api_models' 5s default."""
    seen = {}

    def _fake_probe(_api_key, base_url, timeout=5.0, **_kw):
        seen["timeout"] = timeout
        return {"models": ["slow-catalog-model"], "probed_url": f"{base_url}/models"}

    monkeypatch.setattr("hermes_cli.models.probe_api_models", _fake_probe)
    verdict = validate_requested_model(
        "slow-catalog-model", "custom", api_key="sk-test",
        base_url="https://slow-gateway.example.com/v1")

    assert seen["timeout"] == CUSTOM_ENDPOINT_PROBE_TIMEOUT
    assert verdict["accepted"] is True
