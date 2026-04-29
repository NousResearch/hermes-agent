"""Tests for AIAgent._anthropic_prompt_cache_policy().

The policy returns ``(should_cache, use_native_layout)`` for five endpoint
classes. The test matrix pins the decision for each so a regression (e.g.
silently dropping caching on third-party Anthropic gateways, or applying
the native layout on OpenRouter) surfaces loudly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent(
    *,
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
    api_mode: str = "chat_completions",
    model: str = "anthropic/claude-sonnet-4.6",
) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.base_url = base_url
    agent.api_mode = api_mode
    agent.model = model
    agent._base_url_lower = (base_url or "").lower()
    agent.client = MagicMock()
    agent.quiet_mode = True
    return agent


class TestNativeAnthropic:
    def test_claude_on_native_anthropic_caches_with_native_layout(self):
        agent = _make_agent(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-6",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_api_anthropic_host_detected_even_when_provider_label_differs(self):
        # Some pool configurations label native Anthropic as "anthropic-direct"
        # or similar; falling back to hostname keeps caching on.
        agent = _make_agent(
            provider="anthropic-direct",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-opus-4.6",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)


class TestOpenRouter:
    def test_claude_on_openrouter_caches_with_envelope_layout(self):
        agent = _make_agent(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="anthropic/claude-sonnet-4.6",
        )
        should, native = agent._anthropic_prompt_cache_policy()
        assert should is True
        assert native is False  # OpenRouter uses envelope layout

    def test_non_claude_on_openrouter_does_not_cache(self):
        agent = _make_agent(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="openai/gpt-5.4",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)


class TestThirdPartyAnthropicGateway:
    """Third-party gateways speaking the Anthropic protocol (MiniMax, Zhipu GLM, LiteLLM)."""

    def test_minimax_claude_via_anthropic_messages(self):
        agent = _make_agent(
            provider="custom",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-6",
        )
        should, native = agent._anthropic_prompt_cache_policy()
        assert should is True, "Third-party Anthropic gateway with Claude must cache"
        assert native is True, "Third-party Anthropic gateway uses native cache_control layout"

    def test_third_party_anthropic_non_claude_unknown_provider_does_not_cache(self):
        # A provider exposing e.g. GLM via anthropic_messages transport from
        # a host we don't recognize — we don't know whether it supports
        # cache_control, so stay conservative.
        agent = _make_agent(
            provider="custom",
            base_url="https://some-unknown-gateway.example.com/anthropic",
            api_mode="anthropic_messages",
            model="glm-4.5",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)


class TestMiniMaxAnthropicWire:
    """MiniMax's own model family on its Anthropic-compatible endpoint.

    MiniMax documents cache_control support on ``/anthropic`` (0.1× read
    pricing, 5-minute TTL). Issue #17332: the blanket ``is_claude`` gate on
    the third-party-gateway branch left MiniMax-M2.7 etc. paying full input
    cost every turn. The capability-registry approach (#17339) generalises
    the original fix from #17333 by reading ``ProviderConfig.extra`` rather
    than hardcoding a provider/host check, but the user-visible behaviour
    these tests pin down is unchanged.
    """

    def test_minimax_m27_on_provider_minimax_caches_native_layout(self):
        agent = _make_agent(
            provider="minimax",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="minimax-m2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_minimax_m25_on_provider_minimax_cn_caches_native_layout(self):
        agent = _make_agent(
            provider="minimax-cn",
            base_url="https://api.minimaxi.com/anthropic",
            api_mode="anthropic_messages",
            model="minimax-m2.5",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_custom_provider_pointed_at_minimax_host_caches(self):
        # User wires a custom provider manually at MiniMax's Anthropic URL;
        # host match alone should be sufficient to enable caching.
        agent = _make_agent(
            provider="custom",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="minimax-m2.1",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_minimax_host_china_endpoint_caches(self):
        agent = _make_agent(
            provider="custom",
            base_url="https://api.minimaxi.com/anthropic",
            api_mode="anthropic_messages",
            model="minimax-m2.1",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_minimax_provider_on_openai_wire_does_not_cache(self):
        # chat_completions transport — MiniMax's cache_control support is
        # documented only for the /anthropic endpoint. Stay off.
        agent = _make_agent(
            provider="minimax",
            base_url="https://api.minimax.io/v1",
            api_mode="chat_completions",
            model="minimax-m2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)


class TestOpenAIWireFormatOnCustomProvider:
    """A custom provider using chat_completions (OpenAI wire) should NOT get caching."""

    def test_custom_openai_wire_does_not_cache_even_with_claude_name(self):
        # This is the blocklist risk #9621 failed to avoid: sending
        # cache_control fields in OpenAI-wire JSON can trip strict providers
        # that reject unknown keys.  Stay off unless the transport is
        # explicitly anthropic_messages or the aggregator is OpenRouter.
        agent = _make_agent(
            provider="custom",
            base_url="https://api.fireworks.ai/inference/v1",
            api_mode="chat_completions",
            model="claude-sonnet-4",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)


class TestQwenAlibabaFamily:
    """Qwen on OpenCode/OpenCode-Go/Alibaba — needs cache_control even on OpenAI-wire.

    Upstream pi-mono #3392 / #3393 documented that these providers serve
    zero cache hits without Anthropic-style markers. Regression reported
    by community user (Qwen3.6 on opencode-go burning through
    subscription with no cache). Envelope layout, not native, because the
    wire format is OpenAI chat.completions.
    """

    def test_qwen_on_opencode_go_caches_with_envelope_layout(self):
        agent = _make_agent(
            provider="opencode-go",
            base_url="https://opencode.ai/v1",
            api_mode="chat_completions",
            model="qwen3.6-plus",
        )
        should, native = agent._anthropic_prompt_cache_policy()
        assert should is True, "Qwen on opencode-go must cache"
        assert native is False, "opencode-go is OpenAI-wire; envelope layout"

    def test_qwen35_plus_on_opencode_go(self):
        agent = _make_agent(
            provider="opencode-go",
            base_url="https://opencode.ai/v1",
            api_mode="chat_completions",
            model="qwen3.5-plus",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, False)

    def test_qwen_on_opencode_zen_caches(self):
        agent = _make_agent(
            provider="opencode",
            base_url="https://opencode.ai/v1",
            api_mode="chat_completions",
            model="qwen3-coder-plus",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, False)

    def test_qwen_on_direct_alibaba_caches(self):
        agent = _make_agent(
            provider="alibaba",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_mode="chat_completions",
            model="qwen3-coder",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, False)

    def test_non_qwen_on_opencode_go_does_not_cache(self):
        # GLM / Kimi on opencode-go don't need markers (they have automatic
        # server-side caching or none at all).
        agent = _make_agent(
            provider="opencode-go",
            base_url="https://opencode.ai/v1",
            api_mode="chat_completions",
            model="glm-5",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)

    def test_kimi_on_opencode_go_does_not_cache(self):
        agent = _make_agent(
            provider="opencode-go",
            base_url="https://opencode.ai/v1",
            api_mode="chat_completions",
            model="kimi-k2.5",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)

    def test_qwen_on_openrouter_not_affected(self):
        # Qwen via OpenRouter falls through — OpenRouter has its own
        # upstream caching arrangement for Qwen (provider-dependent).
        agent = _make_agent(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="qwen/qwen3-coder",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)


class TestExplicitOverrides:
    """Policy accepts keyword overrides for switch_model / fallback activation."""

    def test_overrides_take_precedence_over_self(self):
        agent = _make_agent(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="openai/gpt-5.4",
        )
        # Simulate switch_model evaluating cache policy for a Claude target
        # before self.model is mutated.
        should, native = agent._anthropic_prompt_cache_policy(
            model="anthropic/claude-sonnet-4.6",
        )
        assert (should, native) == (True, False)

    def test_fallback_target_evaluated_independently(self):
        # Starting on native Anthropic but falling back to OpenRouter.
        agent = _make_agent(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-opus-4.6",
        )
        should, native = agent._anthropic_prompt_cache_policy(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model="anthropic/claude-sonnet-4.6",
        )
        assert (should, native) == (True, False)


# ── #17332: capability-registry-driven cache decisions ────────────────

class TestCapabilityRegistryAnthropicCache:
    """Provider-registry / hostname-driven enablement (#17332).

    The legacy gate enabled caching on third-party Anthropic-protocol
    gateways only for Claude-named models, which silently disabled
    caching for providers that serve their own non-Claude families
    (MiniMax M2.x). The capability registry replaces brand-name guessing
    with an explicit ``ProviderConfig.extra['anthropic_cache']`` flag
    plus a hostname allowlist, with a user-facing
    ``agent.anthropic_cache_hosts`` config opt-in for unlisted
    endpoints.
    """

    def test_minimax_built_in_provider_caches_with_native_layout(self):
        agent = _make_agent(
            provider="minimax",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_minimax_cn_built_in_provider_caches_with_native_layout(self):
        agent = _make_agent(
            provider="minimax-cn",
            base_url="https://api.minimaxi.com/anthropic",
            api_mode="anthropic_messages",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_custom_provider_pointing_at_minimax_host_caches(self):
        # Operator configured ``provider: custom`` with MiniMax's URL —
        # registry-declared host triggers caching even though the
        # provider id is generic.
        agent = _make_agent(
            provider="custom",
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_custom_provider_pointing_at_minimax_cn_host_caches(self):
        agent = _make_agent(
            provider="custom",
            base_url="https://api.minimaxi.com/anthropic",
            api_mode="anthropic_messages",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_user_configured_host_opts_in_unknown_gateway(self):
        # An operator running a private gateway can opt-in via
        # ``agent.anthropic_cache_hosts`` without an upstream patch.
        agent = _make_agent(
            provider="custom",
            base_url="https://my-internal.gateway.local/anthropic",
            api_mode="anthropic_messages",
            model="our-internal-llm",
        )
        agent._anthropic_cache_user_hosts_cached = (
            "my-internal.gateway.local",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_user_config_normalizes_case_and_whitespace(self):
        agent = _make_agent(
            provider="custom",
            base_url="https://Custom.Example.LOCAL/anthropic",
            api_mode="anthropic_messages",
            model="x",
        )
        # Helper normalizes; the read in __init__ also normalizes — pin both.
        agent._anthropic_cache_user_hosts_cached = ("custom.example.local",)
        assert agent._anthropic_prompt_cache_policy() == (True, True)

    def test_capability_branch_requires_anthropic_messages_transport(self):
        # MiniMax host but on chat_completions (OpenAI wire) — must NOT
        # send cache_control or strict OpenAI-wire providers will 400.
        agent = _make_agent(
            provider="minimax",
            base_url="https://api.minimax.io/anthropic",
            api_mode="chat_completions",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)

    def test_unlisted_anthropic_protocol_gateway_stays_off(self):
        agent = _make_agent(
            provider="custom",
            base_url="https://example-private.invalid/anthropic",
            api_mode="anthropic_messages",
            model="some-internal-llm",
        )
        assert agent._anthropic_prompt_cache_policy() == (False, False)

    def test_provider_id_lookup_is_case_insensitive(self):
        agent = _make_agent(
            provider="MiniMax",  # capitalized
            base_url="https://api.minimax.io/anthropic",
            api_mode="anthropic_messages",
            model="MiniMax-M2.7",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, True)


class TestCapabilityHelperUnit:
    """Direct tests for the helper module."""

    def test_minimax_provider_id_returns_true(self):
        from agent.anthropic_cache_capability import (
            provider_supports_anthropic_cache,
        )
        assert provider_supports_anthropic_cache("minimax", None) is True

    def test_unknown_provider_returns_false(self):
        from agent.anthropic_cache_capability import (
            provider_supports_anthropic_cache,
        )
        assert provider_supports_anthropic_cache("openai", None) is False

    def test_host_lookup_without_provider(self):
        from agent.anthropic_cache_capability import (
            provider_supports_anthropic_cache,
        )
        assert (
            provider_supports_anthropic_cache(
                None, "https://api.minimax.io/anthropic"
            )
            is True
        )

    def test_user_configured_host_added(self):
        from agent.anthropic_cache_capability import (
            provider_supports_anthropic_cache,
        )
        assert (
            provider_supports_anthropic_cache(
                "custom",
                "https://opted-in.example/anthropic",
                user_configured_hosts=("opted-in.example",),
            )
            is True
        )

    def test_empty_inputs_return_false(self):
        from agent.anthropic_cache_capability import (
            provider_supports_anthropic_cache,
        )
        assert provider_supports_anthropic_cache(None, None) is False
        assert provider_supports_anthropic_cache("", "") is False
