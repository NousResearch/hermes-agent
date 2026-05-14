"""Tests for the provider module registry and profiles."""

import pytest
from providers import get_provider_profile, _REGISTRY
from providers.base import ProviderProfile, OMIT_TEMPERATURE


class TestRegistry:
    def test_discovery_populates_registry(self):
        p = get_provider_profile("nvidia")
        assert p is not None
        assert p.name == "nvidia"

    def test_alias_lookup(self):
        assert get_provider_profile("kimi").name == "kimi-coding"
        assert get_provider_profile("moonshot").name == "kimi-coding"
        assert get_provider_profile("kimi-coding-cn").name == "kimi-coding-cn"
        assert get_provider_profile("or").name == "openrouter"
        assert get_provider_profile("nous-portal").name == "nous"
        assert get_provider_profile("qwen").name == "qwen-oauth"
        assert get_provider_profile("qwen-portal").name == "qwen-oauth"

    def test_unknown_provider_returns_none(self):
        assert get_provider_profile("nonexistent-provider") is None

    def test_all_providers_have_name(self):
        get_provider_profile("nvidia")  # trigger discovery
        for name, profile in _REGISTRY.items():
            assert profile.name == name


class TestNvidiaProfile:
    def test_max_tokens(self):
        p = get_provider_profile("nvidia")
        assert p.default_max_tokens == 16384

    def test_no_special_temperature(self):
        p = get_provider_profile("nvidia")
        assert p.fixed_temperature is None

    def test_base_url(self):
        p = get_provider_profile("nvidia")
        assert "nvidia.com" in p.base_url


class TestKimiProfile:
    def test_temperature_omit(self):
        p = get_provider_profile("kimi")
        assert p.fixed_temperature is OMIT_TEMPERATURE

    def test_max_tokens(self):
        p = get_provider_profile("kimi")
        assert p.default_max_tokens == 32000

    def test_cn_separate_profile(self):
        p = get_provider_profile("kimi-coding-cn")
        assert p.name == "kimi-coding-cn"
        assert p.env_vars == ("KIMI_CN_API_KEY",)
        assert "moonshot.cn" in p.base_url

    def test_cn_not_alias_of_kimi(self):
        kimi = get_provider_profile("kimi-coding")
        cn = get_provider_profile("kimi-coding-cn")
        assert kimi is not cn
        assert kimi.base_url != cn.base_url

    def test_thinking_enabled(self):
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"})
        assert eb["thinking"] == {"type": "enabled"}
        assert tl["reasoning_effort"] == "high"

    def test_thinking_disabled(self):
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": False})
        assert eb["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in tl

    def test_reasoning_effort_default(self):
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": True})
        assert tl["reasoning_effort"] == "medium"

    def test_no_config_defaults(self):
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config=None)
        assert eb["thinking"] == {"type": "enabled"}
        assert tl["reasoning_effort"] == "medium"


class TestOpenRouterProfile:
    def test_extra_body_with_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(provider_preferences={"allow": ["anthropic"]})
        assert body["provider"] == {"allow": ["anthropic"]}

    def test_extra_body_no_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body()
        assert body == {}

    def test_reasoning_full_config(self):
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": True, "effort": "high"}

    def test_reasoning_disabled_still_passes(self):
        """OpenRouter passes disabled reasoning through (unlike Nous)."""
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": False}

    def test_default_reasoning(self):
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(supports_reasoning=True)
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}


class TestNousProfile:
    def test_tags(self):
        p = get_provider_profile("nous")
        body = p.build_extra_body()
        assert body["tags"] == ["product=hermes-agent"]

    def test_auth_type(self):
        p = get_provider_profile("nous")
        assert p.auth_type == "oauth_device_code"

    def test_reasoning_enabled(self):
        p = get_provider_profile("nous")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_reasoning_omitted_when_disabled(self):
        p = get_provider_profile("nous")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
        )
        assert "reasoning" not in eb


class TestQwenProfile:
    def test_max_tokens(self):
        p = get_provider_profile("qwen-oauth")
        assert p.default_max_tokens == 65536

    def test_auth_type(self):
        p = get_provider_profile("qwen-oauth")
        assert p.auth_type == "oauth_external"

    def test_extra_body_vl(self):
        p = get_provider_profile("qwen-oauth")
        body = p.build_extra_body()
        assert body["vl_high_resolution_images"] is True

    def test_prepare_messages_normalizes_content(self):
        p = get_provider_profile("qwen-oauth")
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = p.prepare_messages(msgs)
        # System message: content normalized to list, cache_control on last part
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert result[0]["content"][-1]["text"] == "Be helpful"
        # User message: content normalized to list
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["text"] == "hello"

    def test_metadata_top_level(self):
        p = get_provider_profile("qwen-oauth")
        meta = {"sessionId": "s123", "promptId": "p456"}
        eb, tl = p.build_api_kwargs_extras(qwen_session_metadata=meta)
        assert tl["metadata"] == meta
        assert "metadata" not in eb


class TestBaseProfile:
    def test_prepare_messages_passthrough(self):
        p = ProviderProfile(name="test")
        msgs = [{"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) is msgs

    def test_build_extra_body_empty(self):
        p = ProviderProfile(name="test")
        assert p.build_extra_body() == {}

    def test_build_api_kwargs_extras_empty(self):
        p = ProviderProfile(name="test")
        eb, tl = p.build_api_kwargs_extras()
        assert eb == {}
        assert tl == {}
