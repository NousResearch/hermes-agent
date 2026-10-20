"""Tests for Ollama Cloud provider integration."""

import pytest
from unittest.mock import patch

from hermes_cli.auth import resolve_provider, resolve_api_key_provider_credentials
from agent.models_dev import list_agentic_models


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
                    "qwen3.5-next": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5-next"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = provider_model_ids("ollama-cloud", force_refresh=True)

        assert len(result) > 0
        assert "qwen3.5-next" in result


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
                    "qwen3.5-next": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5-next"]), \
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
                    "kimi-k2-flash": {"tool_call": True},
                    "nemotron-3-super": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5-next", "glm-5"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        # Live models first, then models.dev additions (deduped)
        assert result[0] == "qwen3.5-next"   # from live API
        assert result[1] == "glm-5"          # from live API (also in models.dev)
        assert "kimi-k2-flash" in result     # from models.dev only
        assert "nemotron-3-super" in result  # from models.dev only
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
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                    "nemotron-3-nano:30b": {"tool_call": True},
                    "some-embedding:latest": {"tool_call": False},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("ollama-cloud")
        assert "qwen3.5:397b" in result
        assert "glm-5" in result
        assert "nemotron-3-nano:30b" in result
        assert "some-embedding:latest" not in result  # no tool_call


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
                    "glm-5.2:cloud": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["kimi-k2.6", "glm-5.2"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result.count("kimi-k2.6") == 1
        assert result.count("glm-5.2") == 1
        assert "kimi-k2.6:cloud" not in result
        assert "glm-5.2:cloud" not in result


# ── Retired-id Filtering ──

class TestOllamaCloudRetiredModels:
    """ollama.com /v1/models keeps advertising ids the chat endpoint refuses with HTTP 410
    ("was retired on <date>") — verified live 2026-10-19: deepseek-v4-flash[:0731], glm-5.1,
    qwen3.5:397b (2026-09-25), kimi-k2.5, minimax-m2.5 (2026-07-31); clef:27b (400, serves
    /v1/systemone only). Every catalog path must filter them so the picker never offers one."""

    def test_live_and_mdev_merge_drops_retired_ids(self, tmp_path, monkeypatch):
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "kimi-k2.5": {"tool_call": True},           # retired 2026-07-31
                    "deepseek-v4.1-flash": {"tool_call": True},  # alive
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models",
                   return_value=["deepseek-v4-flash", "glm-5.1", "qwen3.5:397b",  # all 410
                                 "glm-5.3-flash"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["glm-5.3-flash", "deepseek-v4.1-flash"]
        # The persisted cache itself must not retain the dead ids either.
        from hermes_cli.models_local import _load_ollama_cloud_cache
        assert _load_ollama_cloud_cache(ignore_ttl=True)["models"] == result

    def test_stale_disk_cache_filtered_on_load(self, tmp_path, monkeypatch):
        """A cache written before the filter existed must not resurrect retired ids."""
        import json as json_mod
        import time as time_mod
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        (tmp_path / "ollama_cloud_models_cache.json").write_text(json_mod.dumps({
            "models": ["minimax-m2.5", "clef:27b", "minimax-m2.7"],
            "cached_at": time_mod.time() - 600,  # fresh inside the 1h TTL
        }))

        with patch("hermes_cli.models.fetch_api_models",
                   side_effect=AssertionError("fresh cache hit must not probe")):
            result = fetch_ollama_cloud_models()

        assert result == ["minimax-m2.7"]

    def test_filter_matches_case_insensitively(self):
        """The filter matches regardless of id case and keeps living ids verbatim."""
        from hermes_cli.models import _filter_retired_ollama_cloud

        assert _filter_retired_ollama_cloud(["GLM-5.1", "GLM-5.2", "DEEPSEEK-V4-FLASH:0731"]) == ["GLM-5.2"]
        assert _filter_retired_ollama_cloud(None) == []
        assert _filter_retired_ollama_cloud([]) == []
