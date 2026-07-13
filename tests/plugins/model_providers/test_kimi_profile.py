"""Unit tests for the Kimi/Moonshot provider profile's reasoning wiring.

Moonshot's OpenAI-compat endpoint (``api.moonshot.ai/v1``) treats
``extra_body.thinking`` and a top-level ``reasoning_effort`` as mutually
exclusive. The profile must send at most one of them — never both — so a
request can't trip "cannot specify both 'thinking' and 'reasoning_effort'".

This mirrors the kimi-k2 handling already shipped for the opencode-go relay
(see ``tests/plugins/model_providers/test_opencode_go_profile.py``).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def kimi_profile():
    """Resolve the registered Kimi profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Kimi profile. Going through ``get_provider_profile`` keeps the test honest:
    if the registered class is ever swapped for a plain ``ProviderProfile`` the
    assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("kimi-coding")
    assert profile is not None, "kimi-coding provider profile must be registered"
    return profile


class TestKimiReasoningWireShape:
    """``build_api_kwargs_extras`` never emits thinking + reasoning_effort together."""

    def test_no_config_enables_thinking_without_effort(self, kimi_profile):
        """No reasoning_config → thinking on, server picks the depth.

        Regression guard: this path previously also sent
        ``reasoning_effort="medium"``, pairing thinking + effort on every
        default call.
        """
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(reasoning_config=None)
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_explicit_effort_sends_effort_only(self, kimi_profile, effort):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top_level == {"reasoning_effort": effort}
        assert "thinking" not in extra_body

    def test_enabled_without_effort_falls_back_to_thinking(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["", "garbage", "xhigh", "max"])
    def test_unrecognized_effort_falls_back_to_thinking(self, kimi_profile, effort):
        """Unknown/strong efforts aren't in Moonshot's low|medium|high set, so
        we drop to the thinking toggle rather than sending an invalid effort."""
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_disabled_sends_thinking_disabled_only(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_disabled_ignores_effort(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize(
        "reasoning_config",
        [
            None,
            {"enabled": True},
            {"enabled": True, "effort": "high"},
            {"enabled": True, "effort": "garbage"},
            {"enabled": False},
            {"enabled": False, "effort": "low"},
        ],
    )
    def test_never_emits_both(self, kimi_profile, reasoning_config):
        """The core invariant: thinking and reasoning_effort are never both set."""
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config
        )
        assert not ("thinking" in extra_body and "reasoning_effort" in top_level)


class TestKimiFullKwargsIntegration:
    """The transport's full kwargs carry at most one reasoning knob."""

    def _build(self, kimi_profile, reasoning_config):
        from agent.transports.chat_completions import ChatCompletionsTransport

        return ChatCompletionsTransport().build_kwargs(
            model="kimi-k2-turbo-preview",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=kimi_profile,
            reasoning_config=reasoning_config,
            base_url="https://api.moonshot.ai/v1",
            provider_name="kimi-coding",
        )

    def test_explicit_effort_omits_thinking(self, kimi_profile):
        kwargs = self._build(kimi_profile, {"enabled": True, "effort": "high"})
        assert kwargs["reasoning_effort"] == "high"
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_no_config_omits_effort(self, kimi_profile):
        kwargs = self._build(kimi_profile, None)
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


class TestKimiCnAliasResolution:
    """Regression tests for kimi-coding-cn alias fix (issue #17739).

    kimi-coding-cn must NOT alias to kimi-for-coding -- it has its own
    ProviderConfig with KIMI_CN_API_KEY and a different base URL.
    kimi-cn and moonshot-cn must resolve to kimi-coding-cn (not kimi-for-coding).
    """

    def test_kimi_coding_cn_resolves_to_itself_not_kimi_for_coding(self):
        """resolve_provider_full('kimi-coding-cn') must return a ProviderDef
        whose id is 'kimi-coding-cn', not 'kimi-for-coding'."""
        from hermes_cli import providers
        result = providers.resolve_provider_full("kimi-coding-cn")
        assert result is not None, "kimi-coding-cn must resolve to a provider"
        assert result.id == "kimi-coding-cn", (
            f"kimi-coding-cn must resolve to its own ProviderDef, not '{result.id}'. "
            "This alias was incorrectly mapped to kimi-for-coding (issue #17739), "
            "which caused KIMI_CN_API_KEY and the China endpoint to be ignored."
        )

    def test_kimi_cn_alias_resolves_to_kimi_coding_cn(self):
        """Short alias 'kimi-cn' must resolve to kimi-coding-cn (not kimi-for-coding)."""
        from hermes_cli import providers
        result = providers.resolve_provider_full("kimi-cn")
        assert result is not None, "kimi-cn must resolve to a provider"
        assert result.id == "kimi-coding-cn", (
            f"kimi-cn must resolve to kimi-coding-cn, got '{result.id}'"
        )

    def test_moonshot_cn_alias_resolves_to_kimi_coding_cn(self):
        """Short alias 'moonshot-cn' must resolve to kimi-coding-cn."""
        from hermes_cli import providers
        result = providers.resolve_provider_full("moonshot-cn")
        assert result is not None, "moonshot-cn must resolve to a provider"
        assert result.id == "kimi-coding-cn", (
            f"moonshot-cn must resolve to kimi-coding-cn, got '{result.id}'"
        )

    def test_kimi_coding_cn_has_dedicated_api_key_env(self):
        """kimi-coding-cn ProviderConfig must use KIMI_CN_API_KEY, not KIMI_API_KEY."""
        from hermes_cli import auth
        config = auth.PROVIDER_REGISTRY.get("kimi-coding-cn")
        assert config is not None, "kimi-coding-cn must be in PROVIDER_REGISTRY"
        assert "KIMI_CN_API_KEY" in config.api_key_env_vars, (
            "kimi-coding-cn must use KIMI_CN_API_KEY to avoid picking up "
            "the global kimi-for-coding credential"
        )
        assert "KIMI_API_KEY" not in config.api_key_env_vars, (
            "kimi-coding-cn must NOT use KIMI_API_KEY (that is for kimi-for-coding)"
        )
