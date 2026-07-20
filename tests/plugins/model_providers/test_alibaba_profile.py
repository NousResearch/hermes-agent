"""Behavior contract for the Alibaba/Qwen Cloud PAYG provider profile.

Qwen Cloud PAYG and Alibaba Cloud Token Plan are separate billing lanes.  The
profile tests intentionally exercise the registered profile, runtime resolver,
and API-key resolver rather than a source snapshot so aliases and credentials
cannot silently drift across lanes.
"""

from __future__ import annotations

import pytest


PAYG_ALIASES = (
    "dashscope",
    "alibaba-cloud",
    "qwen-dashscope",
    "qwen-cloud",
    "qwencloud",
)
PAYG_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
PAYG_FALLBACK_MODELS = (
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen3-coder-plus",
    "qwen3-coder-flash",
    "qwen3-coder-next",
)


@pytest.fixture
def alibaba_profile():
    """Resolve the PAYG profile through the real provider discovery path."""
    import model_tools  # noqa: F401  -- triggers provider plugin discovery
    import providers

    profile = providers.get_provider_profile("alibaba")
    assert profile is not None, "alibaba PAYG provider profile must be registered"
    return profile


class TestAlibabaPaygIdentity:
    def test_canonical_identity_and_public_naming(self, alibaba_profile):
        assert alibaba_profile.name == "alibaba"
        assert alibaba_profile.auth_type == "api_key"
        assert alibaba_profile.api_mode == "chat_completions"

        # Keep the established public label while making the profile's billing
        # lane unambiguous in picker/profile metadata.
        display = alibaba_profile.display_name.lower()
        assert "qwen" in display and "cloud" in display
        assert "payg" in display or "pay-as-you-go" in display

        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.models import CANONICAL_PROVIDERS

        assert PROVIDER_REGISTRY["alibaba"].name == "Qwen Cloud"
        assert next(p for p in CANONICAL_PROVIDERS if p.slug == "alibaba").label == "Qwen Cloud"

    @pytest.mark.parametrize("alias", PAYG_ALIASES)
    def test_alias_resolves_to_payg_canonical(self, alibaba_profile, alias):
        import providers
        from hermes_cli.auth import resolve_provider

        assert providers.get_provider_profile(alias) is alibaba_profile
        assert resolve_provider(alias) == "alibaba"

    def test_aliases_are_declared_on_profile(self, alibaba_profile):
        assert set(PAYG_ALIASES).issubset(set(alibaba_profile.aliases))

    def test_endpoint_and_auth_environment_contract(self, alibaba_profile):
        assert alibaba_profile.base_url == PAYG_BASE_URL
        assert alibaba_profile.env_vars[0] == "DASHSCOPE_API_KEY"
        assert "DASHSCOPE_BASE_URL" in alibaba_profile.env_vars
        assert alibaba_profile.env_vars.index("DASHSCOPE_API_KEY") < alibaba_profile.env_vars.index(
            "DASHSCOPE_BASE_URL"
        )

        from hermes_cli.auth import PROVIDER_REGISTRY

        config = PROVIDER_REGISTRY["alibaba"]
        assert config.auth_type == "api_key"
        assert config.api_key_env_vars == ("DASHSCOPE_API_KEY",)
        assert config.base_url_env_var == "DASHSCOPE_BASE_URL"
        assert config.inference_base_url == PAYG_BASE_URL

    def test_fallbacks_are_agentic_and_auxiliary_model_is_cheap(self, alibaba_profile):
        """Offline fallbacks stay in the agent/tool-capable model lane.

        ProviderProfile is the curated fallback boundary; it must not be filled
        with embedding, reranking, speech, or preview-only models that cannot
        serve the normal Hermes tool loop.  The auxiliary choice should be a
        cheap model from that same curated set.
        """
        fallback_models = tuple(alibaba_profile.fallback_models)
        assert fallback_models
        assert all(isinstance(model, str) and model.strip() for model in fallback_models)
        assert "qwen3.8-max-preview" not in {model.lower() for model in fallback_models}

        auxiliary = alibaba_profile.default_aux_model
        assert auxiliary in fallback_models
        assert any(
            marker in auxiliary.lower() for marker in ("flash", "mini", "lite")
        ), f"expected a cheap auxiliary model, got {auxiliary!r}"

        non_agentic_markers = ("embed", "rerank", "whisper", "tts", "image", "video")
        assert not any(
            any(marker in model.lower() for marker in non_agentic_markers)
            for model in fallback_models
        )

    def test_fallback_catalog_is_the_curated_payg_catalog(self, alibaba_profile):
        assert tuple(alibaba_profile.fallback_models) == PAYG_FALLBACK_MODELS
        assert alibaba_profile.default_aux_model == "qwen3.6-flash"

    @pytest.mark.parametrize(
        ("reasoning_config", "expected"),
        [
            (None, {}),
            ({"enabled": True, "effort": "high"}, {"enable_thinking": True}),
            ({"enabled": False}, {"enable_thinking": False}),
        ],
    )
    def test_thinking_wire_shape(self, alibaba_profile, reasoning_config, expected):
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config,
            model="qwen3.7-plus",
        )
        assert extra_body == expected
        assert top_level == {}

    def test_non_hybrid_models_do_not_receive_qwen_thinking_fields(
        self, alibaba_profile
    ):
        assert alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3-coder-next",
        ) == ({}, {})


class TestAlibabaPaygCredentials:
    def test_runtime_uses_payg_credentials_when_both_billing_lanes_exist(
        self, monkeypatch, alibaba_profile
    ):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "payg-key-canary")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://payg.example.test/v1")
        monkeypatch.setenv("QWEN_TOKEN_PLAN_API_KEY", "token-plan-key-canary")
        monkeypatch.setenv("BAILIAN_TOKEN_PLAN_API_KEY", "bailian-key-canary")

        from hermes_cli.auth import resolve_api_key_provider_credentials
        from hermes_cli.runtime_provider import resolve_runtime_provider

        credentials = resolve_api_key_provider_credentials("alibaba")
        assert credentials["api_key"] == "payg-key-canary"
        assert credentials["source"] == "DASHSCOPE_API_KEY"
        assert credentials["base_url"] == "https://payg.example.test/v1"

        runtime = resolve_runtime_provider(requested="alibaba")
        assert runtime["provider"] == "alibaba"
        assert runtime["api_key"] == "payg-key-canary"
        assert runtime["base_url"] == "https://payg.example.test/v1"
        assert runtime["api_mode"] == "chat_completions"

    def test_payg_does_not_cross_fallback_to_token_plan_key(self, monkeypatch, alibaba_profile):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
        monkeypatch.setenv("QWEN_TOKEN_PLAN_API_KEY", "token-plan-key-canary")
        monkeypatch.setenv("BAILIAN_TOKEN_PLAN_API_KEY", "bailian-key-canary")

        from hermes_cli.auth import resolve_api_key_provider_credentials

        credentials = resolve_api_key_provider_credentials("alibaba")
        assert credentials["api_key"] == ""
        assert credentials["source"] == "default"
        assert credentials["base_url"] == PAYG_BASE_URL
