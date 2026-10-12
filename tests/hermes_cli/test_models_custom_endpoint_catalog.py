"""Catalog discovery for user-configured named custom endpoints.

``provider_model_ids('custom:<key>')`` used to fall through to the empty static list for a
configured ``providers.<key>`` endpoint — key, URL and everything else a probe needs sit right
there in config, but no rung looked. Every surface that delegates to it (the WebUI ``/api/models``
picker and ``/api/models/live`` enrichment, slash-command autocomplete) then rendered a one-entry
catalog no matter what the endpoint served. These tests pin the contract: the endpoint's live
``/v1/models`` answer is the catalog, declared ``models`` are the pin (``discover_models: false``)
or the fallback after a failed probe — never a freeze of the live catalog.
"""

from __future__ import annotations

import pytest


_PROVIDER_CONFIG = {
    "model": {"default": "m1", "provider": "custom:litellm"},
    "providers": {
        "litellm": {"api": "https://gw.example.com/v1", "name": "litellm", "key_env": "GW_KEY"}
    },
}

_LEGACY_CONFIG = {
    "model": {"default": "m1", "provider": "custom:my-gw"},
    "custom_providers": [
        {"name": "My GW", "base_url": "https://gw.example.com/v1", "api_key": "${GW_KEY}"}
    ],
}


@pytest.fixture()
def fetched(monkeypatch):
    """Replace the endpoint probe; records calls and serves ``fetched.models``."""
    calls: list[dict] = []
    state: dict = {"models": ["m1", "m2"]}

    def _fake(api_key, base_url, **kwargs):
        calls.append({"api_key": api_key, "base_url": base_url, **kwargs})
        return state["models"]

    monkeypatch.setattr("hermes_cli.models.cached_fetch_api_models", _fake)
    state["calls"] = calls
    return state


def _set_config(monkeypatch, config):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


class TestNamedCustomEndpointCatalog:
    def test_custom_slug_probes_the_configured_endpoint(self, monkeypatch, fetched):
        _set_config(monkeypatch, _PROVIDER_CONFIG)
        monkeypatch.setenv("GW_KEY", "sk-test")
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:litellm") == ["m1", "m2"]
        call = fetched["calls"][0]
        assert call["api_key"] == "sk-test"
        assert call["base_url"] == "https://gw.example.com/v1"
        assert call["api_mode"] is None
        assert call["headers"] is None

    def test_bare_config_key_resolves_to_the_same_entry(self, monkeypatch, fetched):
        """The WebUI picker asks with the raw ``providers:`` key, not the ``custom:`` slug."""
        _set_config(monkeypatch, _PROVIDER_CONFIG)
        monkeypatch.setenv("GW_KEY", "sk-test")
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("litellm") == ["m1", "m2"]
        assert len(fetched["calls"]) == 1

    def test_legacy_custom_providers_row_matched_by_display_name(self, monkeypatch, fetched):
        _set_config(monkeypatch, _LEGACY_CONFIG)
        monkeypatch.setenv("GW_KEY", "sk-env")
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:my-gw") == ["m1", "m2"]
        assert fetched["calls"][0]["api_key"] == "sk-env"
        assert fetched["calls"][0]["base_url"] == "https://gw.example.com/v1"

    def test_discover_models_false_pins_declared_models(self, monkeypatch, fetched):
        config = {
            "providers": {
                "gw": {
                    "api": "https://gw.example.com/v1",
                    "name": "gw",
                    "key_env": "GW_KEY",
                    "discover_models": False,
                    "models": {"pinned-a": {}, "pinned-b": {}},
                }
            }
        }
        _set_config(monkeypatch, config)
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:gw") == ["pinned-a", "pinned-b"]
        assert fetched["calls"] == []

    def test_failed_probe_falls_back_to_declared_models(self, monkeypatch, fetched):
        config = {
            "providers": {
                "gw": {
                    "api": "https://gw.example.com/v1",
                    "name": "gw",
                    "key_env": "GW_KEY",
                    "models": ["declared-a", "declared-b"],
                }
            }
        }
        _set_config(monkeypatch, config)
        fetched["models"] = None
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:gw") == ["declared-a", "declared-b"]

    def test_live_catalog_wins_over_declared_models(self, monkeypatch, fetched):
        """Declared models are a floor, not an allowlist: the endpoint catalog is authoritative."""
        config = {
            "providers": {
                "gw": {
                    "api": "https://gw.example.com/v1",
                    "name": "gw",
                    "key_env": "GW_KEY",
                    "models": {"declared-a": {}},
                }
            }
        }
        _set_config(monkeypatch, config)
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:gw") == ["m1", "m2"]

    def test_unmatched_id_stays_empty_without_a_probe(self, monkeypatch, fetched):
        _set_config(monkeypatch, _PROVIDER_CONFIG)
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:other") == []
        assert fetched["calls"] == []

    def test_endpointless_providers_block_does_not_probe(self, monkeypatch, fetched):
        """A ``providers.<name>`` block without endpoint tunes a built-in; it is not a route."""
        config = {"providers": {"bedrock": {"stale_timeout_seconds": 600}}}
        _set_config(monkeypatch, config)
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:bedrock") == []
        assert fetched["calls"] == []

    def test_no_config_entries_stays_empty(self, monkeypatch, fetched):
        _set_config(monkeypatch, {})
        from hermes_cli.models import provider_model_ids

        assert provider_model_ids("custom:litellm") == []
        assert fetched["calls"] == []
