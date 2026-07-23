from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import hermes_cli.models as models


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        (" https://API.Anthropic.com/v1/ ", "https://api.anthropic.com"),
        (
            "https://Inference-API.NousResearch.com/v1",
            "https://inference-api.nousresearch.com",
        ),
        ("https://example.com/prefix/v1/", "https://example.com/prefix"),
    ],
)
def test_effective_model_base_url_is_canonical(raw, expected):
    with patch.object(models, "_get_model_config_dict", return_value={"base_url": raw}):
        assert models._effective_model_base_url() == expected


def test_base_url_changes_credential_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch.object(models, "_get_model_config_dict", return_value={}):
        default = models._credential_fingerprint("anthropic")
    with patch.object(
        models,
        "_get_model_config_dict",
        return_value={"base_url": "https://proxy.example/v1"},
    ):
        proxied = models._credential_fingerprint("anthropic")

    assert default != proxied


def test_cache_slot_is_stable_and_segregated_by_base_url():
    with patch.object(models, "_get_model_config_dict", return_value={}):
        assert models._cache_slot_key("anthropic") == "anthropic"

    with patch.object(
        models,
        "_get_model_config_dict",
        return_value={"base_url": "https://PROXY.example/v1/"},
    ):
        first = models._cache_slot_key("anthropic")
    with patch.object(
        models,
        "_get_model_config_dict",
        return_value={"base_url": "https://proxy.example"},
    ):
        equivalent = models._cache_slot_key("anthropic")
    with patch.object(
        models,
        "_get_model_config_dict",
        return_value={"base_url": "https://other.example/v1"},
    ):
        other = models._cache_slot_key("anthropic")

    assert first == equivalent
    assert first != other
    assert first.startswith("anthropic+base_url=")


def test_cached_catalogs_do_not_leak_between_base_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proxy_config = {"base_url": "https://proxy.example/v1"}
    native_config = {}

    with patch.object(models, "provider_model_ids", side_effect=[["proxy-model"], ["native-model"]]) as live:
        with patch.object(models, "_get_model_config_dict", return_value=proxy_config):
            assert models.cached_provider_model_ids("anthropic") == ["proxy-model"]
            assert models.cached_provider_model_ids("anthropic") == ["proxy-model"]
        with patch.object(models, "_get_model_config_dict", return_value=native_config):
            assert models.cached_provider_model_ids("anthropic") == ["native-model"]

    assert live.call_count == 2
    cache_path = tmp_path / "provider_models_cache.json"
    cache = json.loads(cache_path.read_text())
    assert "anthropic" in cache
    assert any(key.startswith("anthropic+base_url=") for key in cache)

    with patch.object(models, "_get_model_config_dict", return_value=proxy_config):
        models.clear_provider_models_cache("anthropic")
    cache = json.loads(cache_path.read_text())
    assert "anthropic" in cache
    assert not any(key.startswith("anthropic+base_url=") for key in cache)
