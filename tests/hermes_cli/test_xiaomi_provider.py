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
# Model setup flow regression coverage
# =============================================================================


def _run_xiaomi_model_flow(
    monkeypatch,
    inputs,
    config_model,
    *,
    api_key="tp-test",
    selected="mimo-v2.5-pro",
):
    """Run the interactive Xiaomi setup flow with patched I/O and config."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import config as config_mod
    from hermes_cli import main as main_mod
    from hermes_cli import model_setup_flows as flows_mod
    from agent import models_dev as models_dev_mod

    saved = {}
    cfg = {"model": dict(config_model)}

    monkeypatch.setattr(main_mod, "_prompt_api_key", lambda *a, **k: (api_key, False))
    monkeypatch.setattr(auth_mod, "_prompt_model_selection", lambda *a, **k: selected)
    monkeypatch.setattr(auth_mod, "_save_model_choice", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "deactivate_provider", lambda *a, **k: None)
    monkeypatch.setattr(models_dev_mod, "list_agentic_models", lambda provider: [selected])

    def fake_get_env_value(name):
        if name == "XIAOMI_API_KEY":
            return api_key
        if name == "XIAOMI_BASE_URL":
            return ""
        return ""

    monkeypatch.setattr(config_mod, "get_env_value", fake_get_env_value)
    monkeypatch.setattr(config_mod, "save_env_value", lambda *a, **k: None)
    monkeypatch.setattr(config_mod, "remove_env_value", lambda *a, **k: None)
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)

    def fake_save_config(new_cfg):
        saved.clear()
        saved.update(new_cfg)

    monkeypatch.setattr(config_mod, "save_config", fake_save_config)

    iterator = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(iterator))

    flows_mod._model_flow_xiaomi({}, current_model="")
    return saved["model"]


class TestXiaomiModelFlow:
    def test_token_plan_existing_cn_endpoint_is_default_and_preserved(self, monkeypatch):
        model = _run_xiaomi_model_flow(
            monkeypatch,
            inputs=["", "", ""],
            config_model={
                "provider": "xiaomi",
                "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                "api_key": "stale-inline-key",
                "api_mode": "anthropic_messages",
            },
        )

        assert model["provider"] == "xiaomi"
        assert model["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
        assert "api_key" not in model
        assert "api_mode" not in model

    def test_token_plan_existing_eu_endpoint_is_default_and_preserved(self, monkeypatch):
        model = _run_xiaomi_model_flow(
            monkeypatch,
            inputs=["", "", ""],
            config_model={
                "provider": "xiaomi",
                "base_url": "https://token-plan-ams.xiaomimimo.com/v1",
            },
        )

        assert model["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"

    def test_token_plan_custom_endpoint_is_preserved_unless_replaced(self, monkeypatch):
        model = _run_xiaomi_model_flow(
            monkeypatch,
            inputs=["", "", ""],
            config_model={
                "provider": "xiaomi",
                "base_url": "https://token-plan-private.xiaomimimo.com/v1",
            },
        )

        assert model["base_url"] == "https://token-plan-private.xiaomimimo.com/v1"

    def test_token_plan_region_selection_can_replace_custom_endpoint(self, monkeypatch):
        model = _run_xiaomi_model_flow(
            monkeypatch,
            inputs=["", "3", ""],
            config_model={
                "provider": "xiaomi",
                "base_url": "https://token-plan-private.xiaomimimo.com/v1",
            },
        )

        assert model["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"

    def test_manual_base_url_override_is_saved(self, monkeypatch):
        model = _run_xiaomi_model_flow(
            monkeypatch,
            inputs=["2", "2", "https://token-plan-custom.xiaomimimo.com/v1"],
            config_model={"provider": "xiaomi", "base_url": "https://api.xiaomimimo.com/v1"},
        )

        assert model["base_url"] == "https://token-plan-custom.xiaomimimo.com/v1"
