"""Discovery honors ``model.base_url`` and cache identity tracks the endpoint.

Coatue data-residency report, issues 3 and 4: with
``model.base_url: https://us.api.openai.com/v1`` in config and
``$OPENAI_BASE_URL`` unset, model discovery probed ``api.openai.com`` (the
wrong endpoint) and the catalog-cache fingerprint did not change when the
configured endpoint changed, so ``hermes config set model.base_url ...``
kept serving the stale cached catalog. Regional hosts also bypassed the
curated intersection, flooding the picker with whisper/tts/embedding rows.

Contracts pinned here:
  1. ``_openai_discovery_base_url()`` resolves config ``model.base_url``
     (when the configured provider matches) → ``$OPENAI_BASE_URL`` → default.
  2. ``_credential_fingerprint()`` changes when the effective configured
     endpoint changes.
  3. Official OpenAI hosts (canonical + regional) all get the curated∩live
     intersection; custom proxies keep the verbatim live list.
"""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import pytest

from hermes_cli import models as models_mod


def _cfg(base_url: str | None, provider: str = "openai-api"):
    cfg = {"provider": provider}
    if base_url is not None:
        cfg["base_url"] = base_url
    return cfg


class TestDiscoveryBaseUrl:
    def test_config_base_url_wins_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://us.api.openai.com/v1"),
        ):
            assert (
                models_mod._openai_discovery_base_url("openai-api")
                == "https://us.api.openai.com/v1"
            )

    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://eu.api.openai.com/v1")
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://us.api.openai.com/v1"),
        ):
            assert (
                models_mod._openai_discovery_base_url("openai-api")
                == "https://eu.api.openai.com/v1"
            )

    def test_config_ignored_when_provider_differs(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://proxy.test/v1", provider="openrouter"),
        ):
            assert (
                models_mod._openai_discovery_base_url("openai-api")
                == "https://api.openai.com/v1"
            )

    def test_default_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with mock_patch.object(
            models_mod, "_get_model_config_dict", return_value=_cfg(None)
        ):
            assert (
                models_mod._openai_discovery_base_url("openai-api")
                == "https://api.openai.com/v1"
            )


class TestFingerprintTracksEndpoint:
    def test_config_endpoint_change_changes_fingerprint(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://us.api.openai.com/v1"),
        ):
            fp_us = models_mod._credential_fingerprint("openai-api")
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://eu.api.openai.com/v1"),
        ):
            fp_eu = models_mod._credential_fingerprint("openai-api")
        assert fp_us != fp_eu

    def test_unrelated_provider_fingerprint_stable_across_openai_config(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://us.api.openai.com/v1"),
        ):
            fp_a = models_mod._credential_fingerprint("openrouter")
        with mock_patch.object(
            models_mod,
            "_get_model_config_dict",
            return_value=_cfg("https://eu.api.openai.com/v1"),
        ):
            fp_b = models_mod._credential_fingerprint("openrouter")
        assert fp_a == fp_b


class TestRegionalCatalogFiltering:
    _RAW_DUMP = [
        "gpt-5.6-terra",
        "whisper-1",
        "tts-1",
        "text-embedding-ada-002",
    ]

    @pytest.mark.parametrize(
        "base",
        [
            "https://api.openai.com/v1",
            "https://us.api.openai.com/v1",
            "https://eu.api.openai.com/v1",
        ],
    )
    def test_official_hosts_intersect_with_curated(self, monkeypatch, base):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", base)
        with mock_patch.object(
            models_mod, "fetch_api_models", return_value=list(self._RAW_DUMP)
        ):
            ids = models_mod.provider_model_ids("openai-api", force_refresh=True)
        assert "whisper-1" not in ids
        assert "tts-1" not in ids
        assert "text-embedding-ada-002" not in ids
        assert "gpt-5.6-terra" in ids

    def test_custom_proxy_keeps_live_list_verbatim(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.corp.test/v1")
        with mock_patch.object(
            models_mod, "fetch_api_models", return_value=list(self._RAW_DUMP)
        ):
            ids = models_mod.provider_model_ids("openai-api", force_refresh=True)
        assert ids == self._RAW_DUMP
