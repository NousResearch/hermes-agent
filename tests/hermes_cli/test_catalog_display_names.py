"""Catalog labels preserve provider names without changing model discovery."""

from unittest.mock import patch

import pytest

from hermes_cli import models as m
from hermes_cli import models_catalog as catalog


class TestPublishedName:

    @pytest.mark.parametrize("key", ["name", "display_name", "displayName"])
    def test_every_catalog_dialect_is_read(self, key):
        # OpenAI-compatible uses ``name``; Anthropic's /v1/models uses ``display_name``.
        assert catalog._published_name({"id": "vendor/model", key: "Vendor Model"}) == "Vendor Model"

    @pytest.mark.parametrize("fields, expected", [
        ({"name": "Primary", "display_name": "Secondary", "displayName": "Third"}, "Primary"),
        ({"name": "vendor/model", "display_name": "Secondary", "displayName": "Third"}, "Secondary"),
        ({"name": " vendor/model ", "display_name": "vendor/model", "displayName": "Third"}, "Third"),
        ({"name": "vendor/model", "display_name": "vendor/model", "displayName": "vendor/model"}, ""),
        ({"name": "  ", "display_name": None, "displayName": "  模型 • Model  "}, "模型 • Model"),
        ({"name": "vendor/model-v2"}, "vendor/model-v2"),
    ])
    def test_label_priority_and_fallbacks(self, fields, expected):
        assert catalog._published_name({"id": "vendor/model", **fields}) == expected

    @pytest.mark.parametrize("invalid", [True, 42, ["Not a label"], {"text": "Not a label"}])
    def test_non_string_name_does_not_hide_display_name(self, monkeypatch, invalid):
        result = _probe_with(monkeypatch, [{
            "id": "vendor/model", "name": invalid, "display_name": "Vendor Model",
        }])
        assert result["models"] == ["vendor/model"]
        assert result["model_names"] == {"vendor/model": "Vendor Model"}


def _probe_with(monkeypatch, rows):
    monkeypatch.setattr(m, "_get_json", lambda *a, **k: {"data": rows})
    monkeypatch.setattr(m, "_probe_neg_key", lambda base: None)
    return m.probe_api_models("key", "https://gw.example.com/v1", timeout=1)


class TestProbeModelNames:

    def test_names_are_collected_and_ids_still_returned(self, monkeypatch):
        result = _probe_with(monkeypatch, [
            {"id": "antigravity/gemini-3.8-flash-tiered", "name": "Gemini 3.8 Flash Tiered"},
            {"id": "plain/model", "name": "Plain Model"},
        ])
        assert result["models"] == ["antigravity/gemini-3.8-flash-tiered", "plain/model"]
        assert result["model_names"] == {
            "antigravity/gemini-3.8-flash-tiered": "Gemini 3.8 Flash Tiered",
            "plain/model": "Plain Model",
        }

    def test_catalog_without_names_yields_empty_map(self, monkeypatch):
        result = _probe_with(monkeypatch, [{"id": "a/b"}, {"id": "c/d"}])
        assert result["model_names"] == {}
        assert result["models"] == ["a/b", "c/d"]

    def test_id_shaped_name_does_not_hide_display_name(self, monkeypatch):
        # Some catalogs publish the routing id under "name" and the label separately.
        result = _probe_with(monkeypatch, [{
            "id": "vendor/model", "name": "vendor/model", "display_name": "Vendor Model",
        }])
        assert result["models"] == ["vendor/model"]
        assert result["model_names"] == {"vendor/model": "Vendor Model"}

    def test_generation_rows_stay_out_of_both(self, monkeypatch):
        result = _probe_with(monkeypatch, [
            {"id": "codex/gpt-image-2", "name": "GPT Image 2",
             "output_modalities": ["image"], "capabilities": {"type": "image"}},
            {"id": "ok/model", "name": "Okay"},
        ])
        assert result["models"] == ["ok/model"]
        assert "codex/gpt-image-2" not in result["model_names"]


class TestFetchApiModelNames:

    def test_returns_the_probe_map(self, monkeypatch):
        monkeypatch.setattr(m, "probe_api_models",
                            lambda *a, **k: {"models": ["x/y"], "model_names": {"x/y": "X Y"}})
        assert catalog.fetch_api_model_names("key", "https://gw/v1") == {"x/y": "X Y"}

    def test_missing_map_is_empty(self, monkeypatch):
        monkeypatch.setattr(m, "probe_api_models", lambda *a, **k: {"models": ["x/y"]})
        assert catalog.fetch_api_model_names("key", "https://gw/v1") == {}

    def test_probe_failure_is_swallowed(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("endpoint down")

        monkeypatch.setattr(m, "probe_api_models", boom)
        assert catalog.fetch_api_model_names("key", "https://gw/v1") == {}

    def test_forwards_auth_and_mode(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(m, "probe_api_models",
                            lambda key, url, **k: seen.update(key=key, url=url, **k) or {})
        catalog.fetch_api_model_names("sk-1", "https://gw/v1", timeout=3,
                                api_mode="anthropic_messages", headers={"X-Extra": "1"})
        assert seen == {
            "key": "sk-1", "url": "https://gw/v1", "timeout": 3,
            "api_mode": "anthropic_messages", "request_headers": {"X-Extra": "1"},
        }


class TestModelsDevNames:
    """Built-in providers have no custom endpoint to probe, so models.dev supplies their names."""

    def test_names_come_from_the_bundled_catalog(self, monkeypatch):
        def fake(*a, **k):
            return {"openrouter": {"models": {
                "anthropic/claude-opus-5": {"name": "Claude Opus 5"}}}}

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", fake)
        assert catalog.model_display_names() == {"anthropic/claude-opus-5": "Claude Opus 5"}

    def test_never_touches_the_network(self, monkeypatch):
        # A picker render must not block on the registry host; cache/disk only.
        seen = {}
        monkeypatch.setattr("agent.models_dev.fetch_models_dev",
                            lambda *a, **k: seen.update(k) or {})
        catalog.model_display_names()
        assert seen.get("allow_network") is False

    def test_id_shaped_names_are_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "agent.models_dev.fetch_models_dev",
            lambda *a, **k: {"p": {"models": {"a/b": {"name": "a/b"}}}})
        assert catalog.model_display_names() == {}

    def test_missing_catalog_is_empty(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no cache")

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", boom)
        assert catalog.model_display_names() == {}
