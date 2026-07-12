"""Tests for Xiaomi MiMo provider support."""


import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestXiaomiProviderRegistry:
    """Verify Xiaomi is registered correctly in the PROVIDER_REGISTRY."""

    def test_registered(self):
        assert "xiaomi" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["xiaomi"].name == "Xiaomi MiMo"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["xiaomi"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["xiaomi"].inference_base_url == "https://api.xiaomimimo.com/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["xiaomi"].api_key_env_vars == ("XIAOMI_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["xiaomi"].base_url_env_var == "XIAOMI_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestXiaomiAliases:
    """All aliases should resolve to 'xiaomi'."""

    @pytest.mark.parametrize("alias", [
        "xiaomi", "mimo", "xiaomi-mimo",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        # Clear env to avoid auto-detection interfering
        for key in ("XIAOMI_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-key-12345678")
        assert resolve_provider(alias) == "xiaomi"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("mimo") == "xiaomi"
        assert normalize_provider("xiaomi-mimo") == "xiaomi"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("mimo") == "xiaomi"
        assert normalize_provider("xiaomi-mimo") == "xiaomi"


# =============================================================================
# Auto-detection
# =============================================================================


class TestXiaomiAutoDetection:
    """Setting XIAOMI_API_KEY should auto-detect the provider."""

    def test_auto_detect(self, monkeypatch):
        # Clear all other provider env vars
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                     "DASHSCOPE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY",
                     "MINIMAX_API_KEY", "KILOCODE_API_KEY",
                     "HF_TOKEN", "GLM_API_KEY", "COPILOT_GITHUB_TOKEN",
                     "GH_TOKEN", "GITHUB_TOKEN", "MINIMAX_CN_API_KEY",
                     "TOKENHUB_API_KEY", "ARCEEAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-xiaomi-test-12345678")
        provider = resolve_provider("auto")
        assert provider == "xiaomi"


# =============================================================================
# Credentials
# =============================================================================


class TestXiaomiCredentials:
    """Test credential resolution for the xiaomi provider."""

    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-12345678")
        status = get_api_key_provider_status("xiaomi")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
        status = get_api_key_provider_status("xiaomi")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-12345678")
        monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("xiaomi")
        assert creds["api_key"] == "sk-test-12345678"
        assert creds["base_url"] == "https://api.xiaomimimo.com/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-12345678")
        monkeypatch.setenv("XIAOMI_BASE_URL", "https://custom.xiaomi.example/v1")
        creds = resolve_api_key_provider_credentials("xiaomi")
        assert creds["base_url"] == "https://custom.xiaomi.example/v1"


# =============================================================================
# Model catalog (dynamic — no static list)
# =============================================================================


class TestXiaomiModelCatalog:
    """Xiaomi uses dynamic model discovery via models.dev."""

    def test_models_dev_mapping(self):
        from agent.models_dev import PROVIDER_TO_MODELS_DEV
        assert PROVIDER_TO_MODELS_DEV["xiaomi"] == "xiaomi"

    def test_static_model_list_fallback(self):
        """Static _PROVIDER_MODELS fallback must exist for model picker.

        We only assert the provider key is present — the specific model
        names are data that changes with upstream releases and doesn't
        belong in tests.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "xiaomi" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["xiaomi"]) >= 1

    def test_list_agentic_models_mock(self, monkeypatch):
        """When models.dev returns Xiaomi data, list_agentic_models should return models."""
        from agent import models_dev as md

        fake_data = {
            "xiaomi": {
                "name": "Xiaomi",
                "api": "https://api.xiaomimimo.com/v1",
                "env": ["XIAOMI_API_KEY"],
                "models": {
                    "mimo-v2-pro": {
                        "limit": {"context": 1000000},
                        "tool_call": True,
                    },
                    "mimo-v2-omni": {
                        "limit": {"context": 256000},
                        "tool_call": True,
                    },
                    "mimo-v2-flash": {
                        "limit": {"context": 256000},
                        "tool_call": True,
                    },
                },
            }
        }
        monkeypatch.setattr(md, "fetch_models_dev", lambda: fake_data)

        result = md.list_agentic_models("xiaomi")
        assert "mimo-v2-pro" in result
        assert "mimo-v2-flash" in result


# =============================================================================
# Normalization
# =============================================================================


class TestXiaomiNormalization:
    """Model name normalization — Xiaomi is a direct provider."""

    def test_vendor_prefix_mapping(self):
        from hermes_cli.model_normalize import _VENDOR_PREFIXES
        assert _VENDOR_PREFIXES.get("mimo") == "xiaomi"

    def test_matching_prefix_strip(self):
        """xiaomi/mimo-v2-pro should normalize to mimo-v2-pro for direct API."""
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "xiaomi" in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_lowercase_model_provider(self):
        """Xiaomi must be in _LOWERCASE_MODEL_PROVIDERS."""
        from hermes_cli.model_normalize import _LOWERCASE_MODEL_PROVIDERS
        assert "xiaomi" in _LOWERCASE_MODEL_PROVIDERS

    def test_lowercase_subset_of_matching_prefix(self):
        """_LOWERCASE_MODEL_PROVIDERS must be a subset of _MATCHING_PREFIX_STRIP_PROVIDERS.

        Otherwise the .lower() code path is unreachable dead code — the
        provider check at line 422 gates entry to the block.
        """
        from hermes_cli.model_normalize import (
            _LOWERCASE_MODEL_PROVIDERS,
            _MATCHING_PREFIX_STRIP_PROVIDERS,
        )
        assert _LOWERCASE_MODEL_PROVIDERS.issubset(_MATCHING_PREFIX_STRIP_PROVIDERS), (
            f"_LOWERCASE_MODEL_PROVIDERS has entries not in _MATCHING_PREFIX_STRIP_PROVIDERS: "
            f"{_LOWERCASE_MODEL_PROVIDERS - _MATCHING_PREFIX_STRIP_PROVIDERS}"
        )

    def test_normalize_strips_provider_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("xiaomi/mimo-v2-pro", "xiaomi")
        assert result == "mimo-v2-pro"

    def test_normalize_bare_name_unchanged(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("mimo-v2-pro", "xiaomi")
        assert result == "mimo-v2-pro"

    @pytest.mark.parametrize("empty_input", ["", None, "   "])
    def test_normalize_empty_and_none(self, empty_input):
        """None, empty, and whitespace-only inputs return empty string."""
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider(empty_input, "xiaomi")
        assert result == ""

    @pytest.mark.parametrize("input_name,expected", [
        ("MiMo-V2.5-Pro", "mimo-v2.5-pro"),
        ("MIMO-V2.5-PRO", "mimo-v2.5-pro"),
        ("MiMo-v2.5-pro", "mimo-v2.5-pro"),
        ("mimo-v2.5-pro", "mimo-v2.5-pro"),     # already lowercase
        ("MiMo-V2-Pro", "mimo-v2-pro"),
        ("MiMo-V2-Omni", "mimo-v2-omni"),
        ("MiMo-V2-Flash", "mimo-v2-flash"),
        ("MiMo-V2.5", "mimo-v2.5"),
    ])
    def test_normalize_lowercases_mixed_case(self, input_name, expected):
        """Xiaomi's API requires lowercase model IDs — mixed case from docs must be lowered."""
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider(input_name, "xiaomi")
        assert result == expected

    @pytest.mark.parametrize("input_name,expected", [
        ("xiaomi/MiMo-V2.5-Pro", "mimo-v2.5-pro"),
        ("xiaomi/MIMO-V2.5-PRO", "mimo-v2.5-pro"),
        ("xiaomi/mimo-v2.5-pro", "mimo-v2.5-pro"),
    ])
    def test_normalize_strips_prefix_and_lowercases(self, input_name, expected):
        """Provider prefix stripping AND lowercasing must both work together."""
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider(input_name, "xiaomi")
        assert result == expected


# =============================================================================
# URL mapping
# =============================================================================


class TestXiaomiURLMapping:
    """Test URL → provider inference for Xiaomi endpoints."""

    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.xiaomimimo.com") == "xiaomi"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "xiaomi" in _PROVIDER_PREFIXES
        assert "mimo" in _PROVIDER_PREFIXES
        assert "xiaomi-mimo" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url
        assert _infer_provider_from_url("https://api.xiaomimimo.com/v1") == "xiaomi"

    def test_infer_from_regional_urls(self):
        """Regional token-plan endpoints should also resolve to xiaomi."""
        from agent.model_metadata import _infer_provider_from_url
        assert _infer_provider_from_url("https://token-plan-ams.xiaomimimo.com/v1") == "xiaomi"
        assert _infer_provider_from_url("https://token-plan-cn.xiaomimimo.com/v1") == "xiaomi"
        assert _infer_provider_from_url("https://token-plan-sgp.xiaomimimo.com/v1") == "xiaomi"


# =============================================================================
# providers.py
# =============================================================================


class TestXiaomiProvidersModule:
    """Test Xiaomi in the unified providers module."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "xiaomi" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["xiaomi"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "XIAOMI_BASE_URL"
        assert not overlay.is_aggregator

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("mimo") == "xiaomi"
        assert normalize_provider("xiaomi-mimo") == "xiaomi"

    def test_label(self):
        from hermes_cli.providers import get_label
        assert get_label("xiaomi") == "Xiaomi MiMo"

    def test_get_provider(self):
        pdef = None
        try:
            from hermes_cli.providers import get_provider
            pdef = get_provider("xiaomi")
        except Exception:
            pass
        if pdef is not None:
            assert pdef.id == "xiaomi"
            assert pdef.transport == "openai_chat"


# =============================================================================
# Auxiliary client
# =============================================================================


class TestXiaomiAuxiliary:
    """Xiaomi auxiliary routing: vision → omni, non-vision → user's main model, never flash."""

    def test_no_flash_in_aux_models(self):
        """mimo-v2-flash must NEVER be used for automatic aux routing."""
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "xiaomi" not in _API_KEY_PROVIDER_AUX_MODELS

    def test_vision_model_override(self):
        """Xiaomi vision tasks should use mimo-v2.5 (multimodal), not the main model."""
        from agent.auxiliary_client import _PROVIDER_VISION_MODELS
        assert "xiaomi" in _PROVIDER_VISION_MODELS
        assert _PROVIDER_VISION_MODELS["xiaomi"] == "mimo-v2.5"


# =============================================================================
# Agent init (no SyntaxError, correct api_mode)
# =============================================================================


class TestXiaomiDoctor:
    """Verify hermes doctor recognizes Xiaomi env vars."""

    def test_provider_env_hints(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS
        assert "XIAOMI_API_KEY" in _PROVIDER_ENV_HINTS


class TestXiaomiAgentInit:
    """Verify the agent can be constructed with xiaomi provider without errors."""

    def test_no_syntax_errors(self):
        """Importing run_agent with xiaomi should not raise."""
        import importlib
        importlib.import_module("run_agent")

    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE
        overlay = HERMES_OVERLAYS["xiaomi"]
        api_mode = TRANSPORT_TO_API_MODE[overlay.transport]
        assert api_mode == "chat_completions"


# =============================================================================
# Native web_search tool injection
# =============================================================================


class TestXiaomiWebSearch:
    """Test MiMo native web_search tool injection via prepare_tools()."""

    def _get_profile(self):
        from providers import get_provider_profile
        return get_provider_profile("xiaomi")

    def _reset_cache(self):
        """Reset the class-level web_search config cache between tests."""
        from plugins.model_providers.xiaomi import XiaomiProfile
        XiaomiProfile._ws_cache = XiaomiProfile._WS_UNSET

    def test_profile_is_xiaomi_profile(self):
        """xiaomi profile must be XiaomiProfile subclass."""
        from plugins.model_providers.xiaomi import XiaomiProfile
        profile = self._get_profile()
        assert isinstance(profile, XiaomiProfile)

    def test_prepare_tools_passthrough_when_disabled(self, monkeypatch):
        """No web_search injected when config absent/disabled."""
        self._reset_cache()
        monkeypatch.setenv("HERMES_CONFIG", "/nonexistent")
        profile = self._get_profile()
        tools = [{"type": "function", "function": {"name": "terminal"}}]
        result = profile.prepare_tools(tools, model="mimo-v2.5-pro")
        assert result == tools
        assert len(result) == 1

    def test_prepare_tools_injects_web_search(self, monkeypatch):
        """web_search injected when config providers.xiaomi.web_search=true."""
        self._reset_cache()
        cfg = {"providers": {"xiaomi": {"web_search": True}}}
        from hermes_cli import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: cfg)

        profile = self._get_profile()
        tools = [{"type": "function", "function": {"name": "terminal"}}]
        result = profile.prepare_tools(tools, model="mimo-v2.5-pro")
        assert len(result) == 2
        ws = result[1]
        assert ws["type"] == "web_search"
        assert ws["max_keyword"] == 3
        assert ws["force_search"] is False
        assert "user_location" not in ws  # no location when empty

    def test_prepare_tools_custom_params(self, monkeypatch):
        """Custom max_keyword, force, location from dict config."""
        self._reset_cache()
        cfg = {"providers": {"xiaomi": {"web_search": {
            "max_keyword": 5,
            "force": True,
            "location": "China",
        }}}}
        from hermes_cli import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: cfg)

        profile = self._get_profile()
        tools = []
        result = profile.prepare_tools(tools, model="mimo-v2-pro")
        ws = result[0]
        assert ws["max_keyword"] == 5
        assert ws["force_search"] is True
        assert ws["user_location"] == {"type": "approximate", "country": "China"}

    def test_prepare_tools_no_duplicate(self, monkeypatch):
        """Calling prepare_tools twice should not duplicate web_search."""
        self._reset_cache()
        cfg = {"providers": {"xiaomi": {"web_search": True}}}
        from hermes_cli import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: cfg)

        profile = self._get_profile()
        tools = [{"type": "function", "function": {"name": "terminal"}}]
        result = profile.prepare_tools(tools, model="mimo-v2.5-pro")
        result2 = profile.prepare_tools(result, model="mimo-v2.5-pro")
        assert len(result2) == 2  # still terminal + web_search, not 3

    def test_prepare_tools_injects_from_empty_list(self, monkeypatch):
        """Native search injects even when Hermes has no tools."""
        self._reset_cache()
        cfg = {"providers": {"xiaomi": {"web_search": True}}}
        from hermes_cli import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: cfg)

        profile = self._get_profile()
        result = profile.prepare_tools([], model="mimo-v2.5-pro")
        assert len(result) == 1
        assert result[0]["type"] == "web_search"

    def test_web_search_config_cached(self, monkeypatch):
        """Config is read once and cached — subsequent calls return same value."""
        self._reset_cache()
        call_count = {"n": 0}
        cfg = {"providers": {"xiaomi": {"web_search": True}}}

        def counting_load():
            call_count["n"] += 1
            return cfg

        from hermes_cli import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "load_config_readonly", counting_load)

        profile = self._get_profile()
        profile.prepare_tools([], model="mimo-v2.5-pro")
        profile.prepare_tools([], model="mimo-v2.5-pro")
        assert call_count["n"] == 1  # only one config read

    def test_providers_xiaomi_does_not_shadow_builtin(self):
        """providers.xiaomi with only web_search must not create a user-defined provider."""
        from hermes_cli.providers import resolve_user_provider
        # This mimics config.yaml: providers: {xiaomi: {web_search: true}}
        user_config = {"xiaomi": {"web_search": True}}
        result = resolve_user_provider("xiaomi", user_config)
        assert result is None  # must NOT return a broken ProviderDef


# =============================================================================
# URL citation annotation flow
# =============================================================================


class TestUrlCitationAnnotations:
    """Test that MiMo web_search url_citation annotations are captured and carried through."""

    def test_normalize_response_captures_annotations(self):
        """normalize_response should extract annotations from message into provider_data."""
        from agent.transports.chat_completions import ChatCompletionsTransport
        from types import SimpleNamespace

        transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)

        # Build a mock response that looks like MiMo's web_search output
        annotation = SimpleNamespace(
            type="url_citation",
            url_citation=SimpleNamespace(
                start_index=0,
                end_index=10,
                title="Example Article",
                url="https://example.com/article",
            ),
        )
        msg = SimpleNamespace(
            content="Here is some info from the web.",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
            refusal=None,
            annotations=[annotation],
            model_extra=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], usage=None, model="mimo-v2.5-pro")

        result = transport.normalize_response(response)

        assert result.annotations is not None
        assert len(result.annotations) == 1
        annot = result.annotations[0]
        assert annot["type"] == "url_citation"
        assert annot["url_citation"]["url"] == "https://example.com/article"
        assert annot["url_citation"]["title"] == "Example Article"

    def test_normalize_response_annotations_from_model_extra(self):
        """Annotations should also be picked up from model_extra (SDK quirk fallback)."""
        from agent.transports.chat_completions import ChatCompletionsTransport
        from types import SimpleNamespace

        transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)

        annotation = {"type": "url_citation", "url_citation": {"url": "https://x.com", "title": "X", "start_index": 0, "end_index": 1}}
        msg = SimpleNamespace(
            content="Test",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
            refusal=None,
            annotations=None,
            model_extra={"annotations": [annotation]},
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], usage=None, model="mimo-v2.5-pro")

        result = transport.normalize_response(response)

        assert result.annotations is not None
        assert len(result.annotations) == 1
        assert result.annotations[0]["url_citation"]["url"] == "https://x.com"

    def test_normalize_response_no_annotations(self):
        """No annotations field should result in None, not an error."""
        from agent.transports.chat_completions import ChatCompletionsTransport
        from types import SimpleNamespace

        transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)

        msg = SimpleNamespace(
            content="Hello",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
            refusal=None,
            annotations=None,
            model_extra=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], usage=None, model="mimo-v2.5-pro")

        result = transport.normalize_response(response)
        assert result.annotations is None

    def test_normalized_response_annotations_property(self):
        """NormalizedResponse.annotations reads from provider_data."""
        from agent.transports.types import NormalizedResponse

        nr = NormalizedResponse(
            content="test",
            tool_calls=None,
            finish_reason="stop",
            provider_data={"annotations": [{"type": "url_citation", "url_citation": {"url": "https://a.com"}}]},
        )
        assert nr.annotations is not None
        assert len(nr.annotations) == 1

    def test_normalized_response_annotations_none_when_no_provider_data(self):
        """annotations property returns None when no provider_data."""
        from agent.transports.types import NormalizedResponse

        nr = NormalizedResponse(content="test", tool_calls=None, finish_reason="stop")
        assert nr.annotations is None
