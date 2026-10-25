"""Tests for Google AI Studio (Gemini) provider integration."""

import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider, detect_vendor
from agent.models_dev import list_agentic_models


# ── Provider Registry ──



# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestGeminiAliases:
    def test_explicit_gemini(self):
        assert resolve_provider("gemini") == "gemini"



    def test_normalize_provider(self):
        assert normalize_provider("google") == "gemini"
        assert normalize_provider("gemini") == "gemini"
        assert normalize_provider("google-ai-studio") == "gemini"


# ── Auto-detection ──

class TestGeminiAutoDetection:
    def test_auto_detects_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        assert resolve_provider("auto") == "gemini"


    def test_google_api_key_priority_over_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "primary-key")
        monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "primary-key"
        assert creds["source"] == "GOOGLE_API_KEY"


# ── Credential Resolution ──

class TestGeminiCredentials:
    def test_resolve_with_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["provider"] == "gemini"
        assert creds["api_key"] == "google-secret"
        assert creds["base_url"] == "https://generativelanguage.googleapis.com/v1beta"

    def test_resolve_with_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "gemini-secret"


    def test_runtime_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="gemini")
        assert result["provider"] == "gemini"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "google-key"
        assert result["base_url"] == "https://generativelanguage.googleapis.com/v1beta"


# ── Model Catalog ──



# ── Model Normalization ──

class TestGeminiModelNormalization:


    def test_gemma_vendor_detection(self):
        assert detect_vendor("gemma-4-31b-it") == "google"


    def test_gemma_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemma-4-31b-it", "openrouter")
        assert result == "google/gemma-4-31b-it"


# ── Context Length ──



# ── Agent Init (no SyntaxError) ──

class TestGeminiAgentInit:

    def test_gemini_agent_uses_chat_completions(self, monkeypatch):
        """Gemini still reports chat_completions even though the transport is native."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client:
            mock_client.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="gemini-2.5-flash",
                provider="gemini",
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "gemini"



    def test_gemini_resolve_provider_client_uses_native_client(self, monkeypatch):
        """resolve_provider_client('gemini') should build GeminiNativeClient."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_TEST_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            resolve_provider_client("gemini")
        assert mock_client.called
        mock_openai.assert_not_called()


# ── models.dev Integration ──

class TestGeminiModelsDev:




    def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "google": {
                "models": {
                    "gemini-3-flash-preview": {"tool_call": True},
                    "gemini-2.5-pro": {"tool_call": True},
                    "gemini-embedding-001": {"tool_call": False},
                    "gemini-2.5-flash-preview-tts": {"tool_call": False},
                    "gemini-live-2.5-flash": {"tool_call": True},
                    "gemini-2.5-flash-preview-04-17": {"tool_call": True},
                    "gemma-4-31b-it": {"tool_call": True},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("gemini")
        assert "gemini-3-flash-preview" in result
        assert "gemini-2.5-pro" in result
        assert "gemma-4-31b-it" not in result
        # Filtered out:
        assert "gemini-embedding-001" not in result      # no tool_call
        assert "gemini-2.5-flash-preview-tts" not in result  # no tool_call
        assert "gemini-live-2.5-flash" not in result     # noise: live-
        assert "gemini-2.5-flash-preview-04-17" not in result  # noise: dated preview


class _CatalogResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class TestNativeGeminiProbe:
    """Native AI Studio probing uses ?key= and models[].name.

    The OpenAI-compat surface stays on Bearer + data[].id. The key must not appear
    in the URL returned to callers.
    """

    def test_native_base_url_detected(self):
        from hermes_cli.models import _is_native_gemini_base_url

        assert _is_native_gemini_base_url("https://generativelanguage.googleapis.com/v1beta")
        assert _is_native_gemini_base_url("https://generativelanguage.googleapis.com/v1beta/")

    def test_openai_compat_and_other_hosts_excluded(self):
        from hermes_cli.models import _is_native_gemini_base_url

        assert not _is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert not _is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        assert not _is_native_gemini_base_url("https://api.openai.com/v1")
        assert not _is_native_gemini_base_url(
            "https://aiplatform.googleapis.com/v1beta1/publishers/google"
        )
        assert not _is_native_gemini_base_url("")
        assert not _is_native_gemini_base_url(None)

    def test_probe_uses_query_auth_and_catalog_urlopen(self):
        from hermes_cli import models

        captured = {}

        def _fake_urlopen(req, timeout=5.0, ssl_context=None):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return _CatalogResp(
                b'{"models": [{"name": "models/gemini-2.5-pro"},'
                b' {"name": "models/gemini-2.0-flash"}, {"name": ""}, "skip"]}'
            )

        models._probe_neg_cache.clear()
        with patch.object(models, "_urlopen_model_catalog_request", _fake_urlopen):
            result = models.probe_api_models(
                api_key="sek/ret+=",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )

        assert "key=sek%2Fret%2B%3D" in captured["url"]
        assert "sek/ret+=" not in captured["url"]
        assert "authorization" not in captured["headers"]
        assert result["models"] == ["gemini-2.5-pro", "gemini-2.0-flash"]
        assert result["probed_url"] == "https://generativelanguage.googleapis.com/v1beta/models"
        assert "sek" not in (result["probed_url"] or "")

    def test_probe_failure_returns_none_without_raising(self):
        from hermes_cli import models

        def _boom(req, timeout=5.0, ssl_context=None):
            raise OSError("catalog down")

        models._probe_neg_cache.clear()
        with patch.object(models, "_urlopen_model_catalog_request", _boom):
            result = models.probe_api_models(
                api_key="secret-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
        assert result["models"] is None
        assert "secret-key" not in (result["probed_url"] or "")

    def test_probe_openai_compat_uses_generic_bearer_path(self):
        from hermes_cli import models

        captured = {}

        def _fake_urlopen(req, timeout=5.0, ssl_context=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            return _CatalogResp(b'{"data": [{"id": "models/gemini-2.5-pro"}]}')

        models._probe_neg_cache.clear()
        with patch.object(models, "_urlopen_model_catalog_request", _fake_urlopen):
            result = models.probe_api_models(
                api_key="secret-key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            )

        assert "?key=" not in captured["url"]
        assert captured["headers"].get("Authorization") == "Bearer secret-key"
        assert result["models"] == ["models/gemini-2.5-pro"]

    def test_profile_fetch_models_uses_query_auth(self):
        import providers

        from hermes_cli import models

        profile = providers.get_provider_profile("gemini")
        assert profile is not None
        captured = {}

        def _fake_urlopen(req, timeout=5.0, ssl_context=None):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return _CatalogResp(b'{"models": [{"name": "models/gemini-2.5-pro"}]}')

        with patch.object(models, "_urlopen_model_catalog_request", _fake_urlopen):
            result = profile.fetch_models(
                api_key="secret-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
        assert result == ["gemini-2.5-pro"]
        assert "key=secret-key" in captured["url"]
        assert "authorization" not in captured["headers"]

    def test_profile_openai_compat_keeps_bearer(self, monkeypatch):
        import providers

        profile = providers.get_provider_profile("gemini")
        captured = {}

        def _fake_open(req, timeout=5.0, **kwargs):
            captured["url"] = req.full_url
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return _CatalogResp(b'{"data": [{"id": "gemini-2.5-pro"}]}')

        monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _fake_open)
        result = profile.fetch_models(
            api_key="secret-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        assert "?key=" not in captured["url"]
        assert captured["headers"].get("authorization") == "Bearer secret-key"
        assert result == ["gemini-2.5-pro"]

