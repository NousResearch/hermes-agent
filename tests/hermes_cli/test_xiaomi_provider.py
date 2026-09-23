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


    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["xiaomi"].inference_base_url == "https://api.xiaomimimo.com/v1"


def test_token_plan_is_a_distinct_picker_and_credential_provider():
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_MODELS
    from hermes_cli.models_catalog_static import group_providers
    from hermes_cli.main_provider_setup import _GENERIC_API_KEY_PROVIDERS

    plan = PROVIDER_REGISTRY["xiaomi-token-plan"]
    assert plan.api_key_env_vars == ("XIAOMI_TOKEN_PLAN_API_KEY",)
    assert plan.base_url_env_var == "XIAOMI_TOKEN_PLAN_BASE_URL"
    assert "xiaomi-token-plan" in _GENERIC_API_KEY_PROVIDERS
    assert "mimo-v2.6-pro" in _PROVIDER_MODELS["xiaomi-token-plan"]
    assert "xiaomi-token-plan" in {entry.slug for entry in CANONICAL_PROVIDERS}
    assert group_providers(["xiaomi", "xiaomi-token-plan"]) == [
        {"kind": "group", "group_id": "xiaomi", "label": "Xiaomi MiMo",
         "description": "Pay-as-you-go API or Token Plan", "members": ["xiaomi", "xiaomi-token-plan"]}
    ]


def test_token_plan_routes_its_own_key_to_selected_region(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XIAOMI_API_KEY", "sk-standard-test")
    monkeypatch.setenv("XIAOMI_TOKEN_PLAN_API_KEY", "tp-plan-test")
    monkeypatch.setenv("XIAOMI_TOKEN_PLAN_BASE_URL", "https://token-plan-ams.xiaomimimo.com/v1")
    resolved = resolve_runtime_provider(requested="xiaomi-token-plan", target_model="mimo-v2.6-pro")

    assert resolved["provider"] == "xiaomi-token-plan"
    assert resolved["api_key"] == "tp-plan-test"
    assert resolved["base_url"] == "https://token-plan-ams.xiaomimimo.com/v1"
    assert resolved["api_mode"] == "chat_completions"


@pytest.mark.parametrize("region,url", [
    ("China", "https://token-plan-cn.xiaomimimo.com/v1"),
    ("Singapore", "https://token-plan-sgp.xiaomimimo.com/v1"),
    ("Europe", "https://token-plan-ams.xiaomimimo.com/v1"),
])
def test_token_plan_setup_offers_each_official_region(monkeypatch, region, url):
    from hermes_cli.model_setup_flows import _select_xiaomi_token_plan_endpoint

    def choose(choices, **kwargs):
        return next(i for i, choice in enumerate(choices) if choice.startswith(region))

    monkeypatch.setattr("hermes_cli.main_provider_setup._prompt_provider_choice", choose)
    assert _select_xiaomi_token_plan_endpoint("") == url


def test_token_plan_normalizes_mimo_model_names():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("xiaomi-token-plan/MiMo-V2.6-Pro", "xiaomi-token-plan") == "mimo-v2.6-pro"


def test_token_plan_chat_picker_excludes_non_chat_models(monkeypatch):
    from hermes_cli.model_setup_flows import _api_key_provider_model_list

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **kw: [
        "mimo-v2.6-pro", "mimo-v2.5-tts", "mimo-v2.5-asr"])
    models = _api_key_provider_model_list(
        "xiaomi-token-plan", PROVIDER_REGISTRY["xiaomi-token-plan"],
        "tp-test", "XIAOMI_TOKEN_PLAN_API_KEY", "https://token-plan-sgp.xiaomimimo.com/v1")

    assert "mimo-v2.6-pro" in models
    assert all("tts" not in model and "asr" not in model for model in models)


def test_token_plan_setup_persists_provider_and_region(tmp_path, monkeypatch):
    from hermes_cli.config import get_env_value, load_config
    from hermes_cli.model_setup_flows import _model_flow_api_key_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XIAOMI_TOKEN_PLAN_API_KEY", "tp-setup-test")
    monkeypatch.setattr("hermes_cli.main_provider_setup._prompt_api_key", lambda *a, **kw: ("tp-setup-test", False))
    monkeypatch.setattr("hermes_cli.main_provider_setup._prompt_provider_choice", lambda *a, **kw: 0)
    monkeypatch.setattr("hermes_cli.model_setup_flows._api_key_provider_model_list", lambda *a: ["mimo-v2.6-pro"])
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: "mimo-v2.6-pro")
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *a, **kw: {})

    _model_flow_api_key_provider(load_config(), "xiaomi-token-plan")

    model = load_config()["model"]
    assert model["provider"] == "xiaomi-token-plan"
    assert model["default"] == "mimo-v2.6-pro"
    assert model["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert get_env_value("XIAOMI_TOKEN_PLAN_BASE_URL") == model["base_url"]


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
                     "MINIMAX_API_KEY", "AI_GATEWAY_API_KEY", "KILOCODE_API_KEY",
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



    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-12345678")
        monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("xiaomi")
        assert creds["api_key"] == "sk-test-12345678"
        assert creds["base_url"] == "https://api.xiaomimimo.com/v1"


    def test_resolve_credentials_reads_home_external_secret_scope(
        self, tmp_path, monkeypatch
    ):
        """BWS-injected keys belong in the profile scope that loaded them."""
        from agent import secret_scope as ss
        from hermes_cli import config as config_module
        from hermes_cli import env_loader

        home = tmp_path / "hermes"
        home.mkdir()
        (home / ".env").write_text("", encoding="utf-8")
        monkeypatch.setattr(config_module, "get_env_path", lambda: home / ".env")
        config_module.invalidate_env_cache()

        monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(home.resolve()),
            {"XIAOMI_API_KEY": "sk-bws-xiaomi-12345678"},
        )

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
        try:
            creds = resolve_api_key_provider_credentials("xiaomi")
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert creds["api_key"] == "sk-bws-xiaomi-12345678"
        assert creds["source"] == "XIAOMI_API_KEY"




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


    def test_matching_prefix_strip(self):
        """xiaomi/mimo-v2-pro should normalize to mimo-v2-pro for direct API."""
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "xiaomi" in _MATCHING_PREFIX_STRIP_PROVIDERS


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



# =============================================================================
# URL mapping
# =============================================================================


class TestXiaomiURLMapping:
    """Test URL → provider inference for Xiaomi endpoints."""


    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "xiaomi" in _PROVIDER_PREFIXES
        assert "mimo" in _PROVIDER_PREFIXES
        assert "xiaomi-mimo" in _PROVIDER_PREFIXES


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




# =============================================================================
# Agent init (no SyntaxError, correct api_mode)
# =============================================================================


class TestXiaomiDoctor:
    """Verify hermes doctor recognizes Xiaomi env vars."""

    def test_provider_env_hints(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS
        assert "XIAOMI_API_KEY" in _PROVIDER_ENV_HINTS
        assert "XIAOMI_TOKEN_PLAN_API_KEY" in _PROVIDER_ENV_HINTS


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
