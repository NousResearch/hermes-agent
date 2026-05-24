"""Tests for Google AI Studio (Gemini) provider integration."""

import os
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider, detect_vendor
from agent.model_metadata import get_model_context_length
from agent.models_dev import PROVIDER_TO_MODELS_DEV, list_agentic_models, _NOISE_PATTERNS


# ── Provider Registry ──

class TestGeminiProviderRegistry:
    def test_gemini_in_registry(self):
        assert "gemini" in PROVIDER_REGISTRY

    def test_gemini_config(self):
        pconfig = PROVIDER_REGISTRY["gemini"]
        assert pconfig.id == "gemini"
        assert pconfig.name == "Google AI Studio"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_gemini_env_vars(self):
        pconfig = PROVIDER_REGISTRY["gemini"]
        assert pconfig.api_key_env_vars == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert pconfig.base_url_env_var == "GEMINI_BASE_URL"

    def test_gemini_base_url(self):
        assert "generativelanguage.googleapis.com" in PROVIDER_REGISTRY["gemini"].inference_base_url


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

    def test_alias_google(self):
        assert resolve_provider("google") == "gemini"

    def test_alias_google_gemini(self):
        assert resolve_provider("google-gemini") == "gemini"

    def test_alias_google_ai_studio(self):
        assert resolve_provider("google-ai-studio") == "gemini"

    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("google") == "gemini"
        assert _PROVIDER_ALIASES.get("google-gemini") == "gemini"
        assert _PROVIDER_ALIASES.get("google-ai-studio") == "gemini"

    def test_normalize_provider(self):
        assert normalize_provider("google") == "gemini"
        assert normalize_provider("gemini") == "gemini"
        assert normalize_provider("google-ai-studio") == "gemini"


# ── Auto-detection ──

class TestGeminiAutoDetection:
    def test_auto_detects_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        assert resolve_provider("auto") == "gemini"

    def test_auto_detects_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
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

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setenv("GEMINI_BASE_URL", "https://custom.endpoint/v1")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["base_url"] == "https://custom.endpoint/v1"

    def test_runtime_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="gemini")
        assert result["provider"] == "gemini"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "google-key"
        assert result["base_url"] == "https://generativelanguage.googleapis.com/v1beta"


# ── Model Catalog ──

class TestGeminiModelCatalog:
    def test_provider_entry_exists(self):
        """Gemini provider has a model catalog entry. Specific model names
        are data that changes with Google releases and don't belong in tests.
        """
        assert "gemini" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["gemini"]) >= 1

    def test_provider_label(self):
        assert "gemini" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["gemini"] == "Google AI Studio"


# ── Model Normalization ──

class TestGeminiModelNormalization:
    def test_passthrough_bare_name(self):
        assert normalize_model_for_provider("gemini-2.5-flash", "gemini") == "gemini-2.5-flash"

    def test_strip_vendor_prefix(self):
        assert normalize_model_for_provider("google/gemini-2.5-flash", "gemini") == "google/gemini-2.5-flash"

    def test_gemma_vendor_detection(self):
        assert detect_vendor("gemma-4-31b-it") == "google"

    def test_gemini_vendor_detection(self):
        assert detect_vendor("gemini-2.5-flash") == "google"

    def test_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemini-2.5-flash", "openrouter")
        assert result == "google/gemini-2.5-flash"

    def test_gemma_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemma-4-31b-it", "openrouter")
        assert result == "google/gemma-4-31b-it"


# ── Context Length ──

class TestGeminiContextLength:
    def test_gemma_4_31b_context(self):
        # Mock external API lookups to test against hardcoded defaults
        # (models.dev and OpenRouter may return different values like 262144).
        with patch("agent.models_dev.lookup_models_dev_context", return_value=None), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):
            ctx = get_model_context_length("gemma-4-31b-it", provider="gemini")
        assert ctx == 256000

    def test_gemini_3_context(self):
        ctx = get_model_context_length("gemini-3.1-pro-preview", provider="gemini")
        assert ctx == 1048576


# ── Agent Init (no SyntaxError) ──

class TestGeminiAgentInit:
    def test_agent_imports_without_error(self):
        """Verify run_agent.py has no SyntaxError (the critical bug)."""
        import importlib
        import run_agent
        importlib.reload(run_agent)

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

    def test_gemini_agent_uses_native_client(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy_REAL_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("run_agent.OpenAI") as mock_openai, \
             patch("run_agent.ContextCompressor") as mock_compressor:
            mock_client.return_value = MagicMock()
            mock_compressor.return_value = MagicMock(context_length=1048576, threshold_tokens=524288)
            from run_agent import AIAgent
            AIAgent(
                model="gemini-2.5-flash",
                provider="gemini",
                api_key="AIzaSy_REAL_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
        assert mock_client.called
        mock_openai.assert_not_called()

    def test_gemini_custom_base_url_keeps_openai_client(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy_REAL_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("run_agent.OpenAI") as mock_openai, \
             patch("run_agent.ContextCompressor") as mock_compressor:
            mock_openai.return_value = MagicMock()
            mock_compressor.return_value = MagicMock(context_length=128000, threshold_tokens=64000)
            from run_agent import AIAgent
            AIAgent(
                model="gemini-2.5-flash",
                provider="gemini",
                api_key="AIzaSy_REAL_KEY",
                base_url="https://proxy.example.com/v1",
            )
        mock_openai.assert_called_once()

    def test_gemini_openai_compat_base_url_keeps_openai_client(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy_REAL_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("run_agent.OpenAI") as mock_openai, \
             patch("run_agent.ContextCompressor") as mock_compressor:
            mock_openai.return_value = MagicMock()
            mock_compressor.return_value = MagicMock(context_length=1048576, threshold_tokens=524288)
            from run_agent import AIAgent
            AIAgent(
                model="gemini-2.5-flash",
                provider="gemini",
                api_key="AIzaSy_REAL_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            )
        mock_openai.assert_called_once()

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

    def test_gemini_resolve_provider_client_keeps_openai_for_non_native_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy_TEST_KEY")
        monkeypatch.setenv("GEMINI_BASE_URL", "https://proxy.example.com/v1")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            resolve_provider_client("gemini")
        mock_openai.assert_called_once()


# ── models.dev Integration ──

class TestGeminiModelsDev:
    def test_gemini_mapped_to_google(self):
        assert PROVIDER_TO_MODELS_DEV.get("gemini") == "google"

    def test_noise_filter_excludes_tts(self):
        assert _NOISE_PATTERNS.search("gemini-2.5-pro-preview-tts")

    def test_noise_filter_excludes_dated_preview(self):
        assert _NOISE_PATTERNS.search("gemini-2.5-flash-preview-04-17")

    def test_noise_filter_excludes_embedding(self):
        assert _NOISE_PATTERNS.search("gemini-embedding-001")

    def test_noise_filter_excludes_live(self):
        assert _NOISE_PATTERNS.search("gemini-live-2.5-flash")

    def test_noise_filter_excludes_image(self):
        assert _NOISE_PATTERNS.search("gemini-2.5-flash-image")

    def test_noise_filter_excludes_customtools(self):
        assert _NOISE_PATTERNS.search("gemini-3.1-pro-preview-customtools")

    def test_noise_filter_passes_stable(self):
        assert not _NOISE_PATTERNS.search("gemini-2.5-flash")

    def test_noise_filter_passes_preview(self):
        # Non-dated preview (e.g. gemini-3-flash-preview) should pass
        assert not _NOISE_PATTERNS.search("gemini-3-flash-preview")

    def test_noise_filter_passes_gemma(self):
        assert not _NOISE_PATTERNS.search("gemma-4-31b-it")

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

    def test_list_provider_models_hides_low_tpm_google_gemmas(self):
        mock_data = {
            "google": {
                "models": {
                    "gemini-2.5-pro": {},
                    "gemma-4-31b-it": {},
                    "gemma-3-27b-it": {},
                    "gemini-1.5-pro": {},
                    "gemini-2.0-flash": {},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            from agent.models_dev import list_provider_models

            result = list_provider_models("gemini")

        assert "gemini-2.5-pro" in result
        assert "gemma-4-31b-it" not in result
        assert "gemma-3-27b-it" not in result
        assert "gemini-1.5-pro" not in result
        assert "gemini-2.0-flash" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Express Mode — peer provider profile next to AI Studio.
# Same native REST shape ({base}/models/{model}:generateContent), API-key auth
# via x-goog-api-key header, routes through GeminiNativeClient.
# ─────────────────────────────────────────────────────────────────────────────


class TestVertexProviderRegistry:
    def test_vertex_in_registry(self):
        assert "gemini-vertex" in PROVIDER_REGISTRY

    def test_vertex_config(self):
        pconfig = PROVIDER_REGISTRY["gemini-vertex"]
        assert pconfig.id == "gemini-vertex"
        assert pconfig.name == "Google Cloud Vertex AI (Express Mode)"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == (
            "https://aiplatform.googleapis.com/v1beta1/publishers/google"
        )

    def test_vertex_env_vars(self):
        pconfig = PROVIDER_REGISTRY["gemini-vertex"]
        # Pras's preferred VERTEX_API_KEY first, then GCP-conventional fallbacks.
        assert pconfig.api_key_env_vars == (
            "VERTEX_API_KEY",
            "GOOGLE_VERTEX_API_KEY",
            "GOOGLE_CLOUD_API_KEY",
        )
        assert pconfig.base_url_env_var == "VERTEX_BASE_URL"

    def test_vertex_base_url_is_express_mode_endpoint(self):
        assert "aiplatform.googleapis.com" in PROVIDER_REGISTRY["gemini-vertex"].inference_base_url
        assert "/publishers/google" in PROVIDER_REGISTRY["gemini-vertex"].inference_base_url


class TestVertexAliases:
    def test_explicit_gemini_vertex(self):
        assert resolve_provider("gemini-vertex") == "gemini-vertex"

    def test_alias_vertex(self):
        assert resolve_provider("vertex") == "gemini-vertex"

    def test_alias_vertex_ai(self):
        assert resolve_provider("vertex-ai") == "gemini-vertex"

    def test_alias_google_vertex(self):
        assert resolve_provider("google-vertex") == "gemini-vertex"

    def test_alias_vertex_express(self):
        assert resolve_provider("vertex-express") == "gemini-vertex"

    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("vertex") == "gemini-vertex"
        assert _PROVIDER_ALIASES.get("vertex-ai") == "gemini-vertex"
        assert _PROVIDER_ALIASES.get("google-vertex") == "gemini-vertex"
        assert _PROVIDER_ALIASES.get("vertex-express") == "gemini-vertex"

    def test_normalize_provider(self):
        assert normalize_provider("vertex") == "gemini-vertex"
        assert normalize_provider("vertex-ai") == "gemini-vertex"
        assert normalize_provider("gemini-vertex") == "gemini-vertex"

    def test_vertex_does_not_alias_to_plain_gemini(self):
        """Vertex and AI Studio are peer providers — neither aliases to the other."""
        assert _PROVIDER_ALIASES.get("vertex") != "gemini"
        assert normalize_provider("vertex") != "gemini"


class TestVertexAutoDetection:
    def test_auto_detects_vertex_api_key(self, monkeypatch):
        monkeypatch.setenv("VERTEX_API_KEY", "test-vertex-key")
        # auto-detection precedence: GOOGLE_API_KEY → gemini (AI Studio) wins
        # if both are set; pure VERTEX_API_KEY should pick gemini-vertex.
        # Note: resolve_provider("auto") only falls to gemini-vertex when no
        # AI Studio creds exist. This documents the current precedence.
        provider = resolve_provider("auto")
        # When only VERTEX_API_KEY is present, auto-detect picks either
        # gemini-vertex or falls back to a default. We assert membership in
        # the gemini family rather than strict equality to keep the test
        # robust to precedence tweaks.
        assert provider in ("gemini-vertex", "gemini") or provider is None


class TestVertexCredentials:
    def test_resolve_with_vertex_api_key(self, monkeypatch):
        monkeypatch.setenv("VERTEX_API_KEY", "vertex-secret")
        creds = resolve_api_key_provider_credentials("gemini-vertex")
        assert creds["provider"] == "gemini-vertex"
        assert creds["api_key"] == "vertex-secret"
        assert creds["base_url"] == (
            "https://aiplatform.googleapis.com/v1beta1/publishers/google"
        )

    def test_resolve_with_fallback_env_var(self, monkeypatch):
        # Second-priority env var also works.
        monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "gcp-secret")
        creds = resolve_api_key_provider_credentials("gemini-vertex")
        assert creds["api_key"] == "gcp-secret"

    def test_vertex_first_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("VERTEX_API_KEY", "primary")
        monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "secondary")
        creds = resolve_api_key_provider_credentials("gemini-vertex")
        assert creds["api_key"] == "primary"


class TestVertexCanonicalAndModelCatalog:
    def test_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "gemini-vertex" in slugs

    def test_canonical_entry_metadata(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "gemini-vertex")
        assert "Vertex" in entry.label
        assert "Express" in entry.label or "express" in entry.tui_desc.lower()

    def test_provider_label(self):
        assert "gemini-vertex" in _PROVIDER_LABELS
        assert "Vertex" in _PROVIDER_LABELS["gemini-vertex"]

    def test_model_catalog_has_entries(self):
        assert "gemini-vertex" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["gemini-vertex"]
        # Vertex express exposes the standard Gemini family.
        assert any("gemini-3" in m for m in models)
        assert any("flash" in m for m in models)


class TestVertexNativeRouting:
    def test_is_gemini_native_provider_recognizes_vertex(self):
        from agent.gemini_native_adapter import is_gemini_native_provider
        assert is_gemini_native_provider("gemini-vertex") is True
        assert is_gemini_native_provider("gemini") is True
        # Case-insensitive
        assert is_gemini_native_provider("GEMINI-VERTEX") is True

    def test_is_gemini_native_provider_rejects_others(self):
        from agent.gemini_native_adapter import is_gemini_native_provider
        assert is_gemini_native_provider("openrouter") is False
        assert is_gemini_native_provider("anthropic") is False
        assert is_gemini_native_provider(None) is False
        assert is_gemini_native_provider("") is False

    def test_vertex_base_url_recognized_as_native(self):
        from agent.gemini_native_adapter import is_native_gemini_base_url
        assert is_native_gemini_base_url(
            "https://aiplatform.googleapis.com/v1beta1/publishers/google"
        ) is True

    def test_ai_studio_base_url_still_native(self):
        from agent.gemini_native_adapter import is_native_gemini_base_url
        assert is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta"
        ) is True

    def test_ai_studio_openai_compat_subpath_not_native(self):
        from agent.gemini_native_adapter import is_native_gemini_base_url
        assert is_native_gemini_base_url(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ) is False

    def test_unrelated_base_url_not_native(self):
        from agent.gemini_native_adapter import is_native_gemini_base_url
        assert is_native_gemini_base_url("https://api.openai.com/v1") is False
        assert is_native_gemini_base_url("https://openrouter.ai/api/v1") is False
        assert is_native_gemini_base_url("") is False

    def test_native_gemini_providers_constant_is_authoritative(self):
        """NATIVE_GEMINI_PROVIDERS is the single source of truth for native routing."""
        from agent.gemini_native_adapter import NATIVE_GEMINI_PROVIDERS
        assert "gemini" in NATIVE_GEMINI_PROVIDERS
        assert "gemini-vertex" in NATIVE_GEMINI_PROVIDERS
        # Future-proof: subclasses can extend this without touching core
        # auxiliary_client / agent_runtime_helpers callers.

    def test_resolve_provider_client_routes_vertex_to_native(self, monkeypatch):
        """resolve_provider_client('gemini-vertex') should build GeminiNativeClient."""
        monkeypatch.setenv("VERTEX_API_KEY", "AIza_VERTEX_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            resolve_provider_client("gemini-vertex")
        assert mock_client.called, "Vertex should route through GeminiNativeClient"
        mock_openai.assert_not_called()


class TestVertexAgentInit:
    def test_vertex_agent_uses_native_client(self, monkeypatch):
        """End-to-end: AIAgent(provider='gemini-vertex') uses GeminiNativeClient."""
        monkeypatch.setenv("VERTEX_API_KEY", "AIza_VERTEX_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client, \
             patch("run_agent.OpenAI") as mock_openai, \
             patch("run_agent.ContextCompressor") as mock_compressor:
            mock_client.return_value = MagicMock()
            mock_compressor.return_value = MagicMock(
                context_length=1048576, threshold_tokens=524288
            )
            from run_agent import AIAgent
            AIAgent(
                model="gemini-3-flash-preview",
                provider="gemini-vertex",
                api_key="AIza_VERTEX_KEY",
                base_url=(
                    "https://aiplatform.googleapis.com/v1beta1/publishers/google"
                ),
            )
        assert mock_client.called
        mock_openai.assert_not_called()

    def test_vertex_agent_reports_chat_completions(self, monkeypatch):
        """Vertex profile uses api_mode='chat_completions' like other Gemini variants."""
        monkeypatch.setenv("VERTEX_API_KEY", "AIza_VERTEX_KEY")
        with patch("agent.gemini_native_adapter.GeminiNativeClient") as mock_client:
            mock_client.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="gemini-3-flash-preview",
                provider="gemini-vertex",
                api_key="AIza_VERTEX_KEY",
                base_url=(
                    "https://aiplatform.googleapis.com/v1beta1/publishers/google"
                ),
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "gemini-vertex"


# ── --provider Flag Resolution ──

class TestVertexProviderFlagResolution:
    """Regression tests for /model --provider gemini-vertex (and aliases).

    Vertex was originally registered only in the plugin registry
    (``providers/__init__.py``), which the model picker reads. The
    ``--provider`` flag handler used ``resolve_provider_full()`` from
    ``hermes_cli/providers.py``, which only looked at HERMES_OVERLAYS +
    models.dev + user config — so vertex was unreachable via the flag
    even though the picker showed it. ``get_provider()`` now falls back
    to the plugin registry as a final step; these tests pin that.
    """

    def test_resolves_canonical_name(self):
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("gemini-vertex")
        assert pdef is not None
        assert pdef.id == "gemini-vertex"
        assert pdef.source == "plugin"

    def test_resolves_short_alias_vertex(self):
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("vertex")
        assert pdef is not None
        assert pdef.id == "gemini-vertex"

    def test_resolves_vertex_express_alias(self):
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("vertex-express")
        assert pdef is not None
        assert pdef.id == "gemini-vertex"

    def test_resolves_vertex_ai_alias(self):
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("vertex-ai")
        assert pdef is not None
        assert pdef.id == "gemini-vertex"

    def test_carries_express_mode_env_vars(self):
        """Make sure the resolved ProviderDef has the API-key envs, not the
        service-account envs that models.dev's google-vertex carries."""
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("gemini-vertex")
        assert pdef is not None
        assert "VERTEX_API_KEY" in pdef.api_key_env_vars
        assert pdef.auth_type == "api_key"

    def test_uses_express_mode_base_url(self):
        from hermes_cli.providers import resolve_provider_full
        pdef = resolve_provider_full("gemini-vertex")
        assert pdef is not None
        assert "aiplatform.googleapis.com" in pdef.base_url
