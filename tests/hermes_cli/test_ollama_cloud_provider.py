"""Tests for Ollama Cloud provider integration."""

import pytest
from unittest.mock import patch

from hermes_cli.auth import resolve_provider, resolve_api_key_provider_credentials
from agent.models_dev import list_agentic_models, list_provider_models, list_deprecated_model_ids


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestOllamaCloudAliases:

    def test_alias_ollama_underscore(self):
        """ollama_cloud (underscore) is the unambiguous cloud alias."""
        assert resolve_provider("ollama_cloud") == "ollama-cloud"


# ── Auto-detection ──

class TestOllamaCloudAutoDetection:
    def test_auto_detects_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
        assert resolve_provider("auto") == "ollama-cloud"


# ── Credential Resolution ──

class TestOllamaCloudCredentials:
    def test_resolve_with_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
        creds = resolve_api_key_provider_credentials("ollama-cloud")
        assert creds["provider"] == "ollama-cloud"
        assert creds["api_key"] == "ollama-secret"
        assert creds["base_url"] == "https://ollama.com/v1"


    def test_runtime_ollama_cloud(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="ollama-cloud")
        assert result["provider"] == "ollama-cloud"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "ollama-key"
        assert result["base_url"] == "https://ollama.com/v1"


# ── Model Catalog (dynamic — no static list) ──

class TestOllamaCloudModelCatalog:


    def test_provider_model_ids_returns_dynamic_models(self, tmp_path, monkeypatch):
        """provider_model_ids('ollama-cloud') should call fetch_ollama_cloud_models()."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = provider_model_ids("ollama-cloud", force_refresh=True)

        assert len(result) > 0
        assert "qwen3.5:397b" in result


# ── Model Picker (list_authenticated_providers) ──

class TestOllamaCloudModelPicker:
    def test_ollama_cloud_shows_model_count(self, tmp_path, monkeypatch):
        """Ollama Cloud should show non-zero model count in provider picker."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            providers = list_authenticated_providers(current_provider="ollama-cloud")

        ollama = next((p for p in providers if p["slug"] == "ollama-cloud"), None)
        assert ollama is not None, "ollama-cloud should appear when OLLAMA_API_KEY is set"
        assert ollama["total_models"] > 0, "ollama-cloud should show non-zero model count"

    def test_ollama_cloud_not_shown_without_creds(self, monkeypatch):
        """Ollama Cloud should not appear without credentials."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        providers = list_authenticated_providers(current_provider="openrouter")
        ollama = next((p for p in providers if p["slug"] == "ollama-cloud"), None)
        assert ollama is None, "ollama-cloud should not appear without OLLAMA_API_KEY"


# ── Merged Model Discovery ──

class TestOllamaCloudMergedDiscovery:
    def test_merges_live_and_models_dev(self, tmp_path, monkeypatch):
        """Live API models appear first, models.dev additions fill gaps."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "glm-5": {"tool_call": True},
                    "mdev-extra": {"tool_call": True},
                    "nemotron-3-super": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["clef:27b", "glm-5"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        # Live models first, then models.dev additions (deduped)
        assert result[0] == "clef:27b"        # from live API
        assert result[1] == "glm-5"           # from live API (also in models.dev)
        assert "mdev-extra" in result         # from models.dev only
        assert "nemotron-3-super" in result   # from models.dev only
        assert result.count("glm-5") == 1    # no duplicates

    def test_falls_back_to_models_dev_without_api_key(self, tmp_path, monkeypatch):
        """Without API key, only models.dev results are returned."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["glm-5"]

    def test_cache_only_serves_stale_cache_without_rewriting_disk(self, tmp_path, monkeypatch):
        """cache_only (GUI read path) must not persist a live-less list: that stamps it fresh, drops the
        live-only ids, and makes the next probing call serve the trimmed list for an hour."""
        import json
        import time
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        cache = tmp_path / "ollama_cloud_models_cache.json"
        cache.write_text(json.dumps({"models": ["live-only", "shared"], "cached_at": time.time() - 7200}))
        before = (cache.read_text(), cache.stat().st_mtime_ns)

        mock_mdev = {"ollama-cloud": {"models": {"shared": {"tool_call": True}, "mdev-only": {"tool_call": True}}}}
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev), \
             patch("hermes_cli.models.fetch_api_models", side_effect=AssertionError("network probe ran")):
            result = fetch_ollama_cloud_models(cache_only=True)

        assert result == ["live-only", "shared"]
        assert (cache.read_text(), cache.stat().st_mtime_ns) == before

        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev), \
             patch("hermes_cli.models.fetch_api_models", return_value=["live-only", "shared", "new-live"]) as live:
            assert fetch_ollama_cloud_models() == ["live-only", "shared", "new-live", "mdev-only"]
        assert live.called  # the stale cache still triggers a probe on the next non-cache_only call


# ── models.dev Integration ──

class TestOllamaCloudModelsDev:

    def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "ollama-cloud": {
                "models": {
                    "gpt-oss:120b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                    "nemotron-3-nano:30b": {"tool_call": True},
                    "some-embedding:latest": {"tool_call": False},
                    # Ollama 410-retires deprecated ids; must not surface via the catalog.
                    "glm-5.1": {"tool_call": True, "status": "deprecated"},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("ollama-cloud")
        assert "gpt-oss:120b" in result
        assert "glm-5" in result
        assert "nemotron-3-nano:30b" in result
        assert "some-embedding:latest" not in result  # no tool_call
        assert "glm-5.1" not in result                # status=deprecated


# ── Retirement Filtering (models.dev status="deprecated") ──

class TestOllamaCloudRetirementFilter:
    """Ollama Cloud 410-retires whole models; models.dev keeps listing them (tool_call=True,
    status the only flag). Every picker surface that trusts models.dev additions must hide them,
    scoped to the RESOLVED models.dev id (custom ``catalog_provider`` aliases resolve there too).
    Live-verified 2026-10-22: glm-5.1 / minimax-m2.5 / kimi-k2.5 / deepseek-v4-flash(:0731) /
    qwen3.5:397b all HTTP 410 from ollama.com; deepseek-v4-pro (mdev-only bare alias) still 200s,
    so no hard ``live ∩ models.dev`` intersection — status is the filter."""

    def test_deprecated_hidden_but_bare_alias_kept_in_merged_discovery(self, tmp_path, monkeypatch):
        """Merged list drops status=deprecated ids even with tool_call=True; unsuffixed mdev-only
        aliases the live endpoint omits (but still serves) stay in the picker."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "shared": {"tool_call": True},
                    "glm-5.1": {"tool_call": True, "status": "deprecated"},
                    "deepseek-v4-pro": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["clef:27b", "shared"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["clef:27b", "shared", "deepseek-v4-pro"]

    def test_stale_cache_serve_is_filtered(self, tmp_path, monkeypatch):
        """The stale-serve fallback (unbounded age) must not re-offer retired ids from a cache
        written before the retirement was known."""
        import json
        import time
        from hermes_cli.models_local import _load_ollama_cloud_cache

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache = tmp_path / "ollama_cloud_models_cache.json"
        models = ["good-id", "glm-5.1", "deepseek-v4-pro"]

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "good-id": {"tool_call": True},
                    "glm-5.1": {"tool_call": True, "status": "deprecated"},
                    "glm-5.1:cloud": {"tool_call": True, "status": "deprecated"},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            # stale read (unbounded age — the reservoir case)
            cache.write_text(json.dumps({"models": models, "cached_at": time.time() - 7200}))
            assert (_load_ollama_cloud_cache(ignore_ttl=True) or {})["models"] == ["good-id", "deepseek-v4-pro"]
            # fresh read (within the 1h TTL)
            cache.write_text(json.dumps({"models": models, "cached_at": time.time()}))
            assert (_load_ollama_cloud_cache() or {})["models"] == ["good-id", "deepseek-v4-pro"]

    def test_stale_cache_serve_degrades_without_models_dev_data(self, tmp_path, monkeypatch):
        """No models.dev data available → return the cache unchanged, not a blanked list."""
        import json
        import time
        from hermes_cli.models_local import _load_ollama_cloud_cache

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache = tmp_path / "ollama_cloud_models_cache.json"
        cache.write_text(json.dumps({"models": ["good-id", "glm-5.1"], "cached_at": time.time()}))

        with patch("agent.models_dev.fetch_models_dev", return_value={}):
            assert (_load_ollama_cloud_cache(ignore_ttl=True) or {})["models"] == ["good-id", "glm-5.1"]

    def test_other_providers_keep_deprecated_entries(self):
        """Scoping: a provider that serves its deprecated ids (zai) is NOT affected."""
        mock_data = {
            "zai": {"models": {"served-deprecated": {"tool_call": True, "status": "deprecated"}}},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            assert list_agentic_models("zai") == ["served-deprecated"]
            assert list_provider_models("zai") == ["served-deprecated"]

    def test_custom_catalog_provider_alias_also_filtered(self, tmp_path, monkeypatch):
        """A custom provider cataloging ollama-cloud rows resolves through _models_dev_id, so the
        same retirement filter applies — keyed on the resolved id, not the caller's slug."""
        from agent import models_dev
        from hermes_cli import config as config_module

        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "providers:\n  my-ollama:\n    catalog_provider: ollama-cloud\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        config_module._LOAD_CONFIG_CACHE.clear()
        config_module._RAW_CONFIG_CACHE.clear()

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "glm-5": {"tool_call": True},
                    "glm-5.1": {"tool_call": True, "status": "deprecated"},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            assert list_agentic_models("my-ollama") == ["glm-5"]
            assert models_dev.list_deprecated_model_ids("ollama-cloud") == {"glm-5.1"}

    def test_list_deprecated_model_ids_normalizes_status(self, tmp_path, monkeypatch):
        """Status casing/whitespace is not guaranteed upstream; non-dict entries are ignored."""
        mock_data = {
            "ollama-cloud": {
                "models": {
                    "a": {"tool_call": True, "status": "Deprecated "},
                    "b": {"tool_call": True, "status": None},
                    "c": {"tool_call": True},
                    "d": "not-a-dict",
                }
            }
        }
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            assert list_deprecated_model_ids("ollama-cloud") == {"a"}


# ── providers.py New System ──

class TestOllamaCloudProvidersNew:

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider as np
        assert np("ollama") == "custom"  # bare "ollama" = local
        assert np("ollama-cloud") == "ollama-cloud"


# ── Cloud Suffix Stripping ──

class TestOllamaCloudSuffixStripping:
    """models.dev appends :cloud / -cloud suffixes that the live API omits.

    fetch_ollama_cloud_models() must normalise these before the dedup merge so
    users never see broken IDs like 'kimi-k2.6:cloud' in the model picker.
    """


    def test_no_duplicate_when_live_clean_and_mdev_suffixed(self, tmp_path, monkeypatch):
        """Live API returns clean ID; mdev has :cloud variant — result has exactly one entry."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "kimi-k2.6:cloud": {"tool_call": True},
                    "glm-5.1:cloud": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["kimi-k2.6", "glm-5.1"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result.count("kimi-k2.6") == 1
        assert result.count("glm-5.1") == 1
        assert "kimi-k2.6:cloud" not in result
        assert "glm-5.1:cloud" not in result


    def test_strip_suffix_helper(self):
        """Unit test for the _strip_ollama_cloud_suffix helper."""
        from hermes_cli.models_local import _strip_ollama_cloud_suffix

        assert _strip_ollama_cloud_suffix("kimi-k2.6:cloud") == "kimi-k2.6"
        assert _strip_ollama_cloud_suffix("glm-5.1:cloud") == "glm-5.1"
        assert _strip_ollama_cloud_suffix("qwen3-coder:480b-cloud") == "qwen3-coder:480b"
        assert _strip_ollama_cloud_suffix("nemotron-3-nano:30b") == "nemotron-3-nano:30b"
        assert _strip_ollama_cloud_suffix("") == ""
