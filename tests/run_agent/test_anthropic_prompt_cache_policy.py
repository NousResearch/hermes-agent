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
    cost every turn. Allowlist MiniMax explicitly via provider id or host.
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
            model="minimax-m2.7",
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

    def test_qwen_on_nous_portal_caches_with_envelope_layout(self):
        # Nous Portal Qwen takes the same envelope-layout cache_control
        # path as Portal Claude. Without this, Portal-routed qwen3.6-plus
        # falls through to the alibaba-family check (which only matches
        # provider=opencode/alibaba) and serves 0% cache hits.
        agent = _make_agent(
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
            model="qwen3.6-plus",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, False)

    def test_qwen_vendored_slug_on_nous_portal_caches(self):
        # Same path but with the vendored slug form Portal sometimes uses.
        agent = _make_agent(
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
            model="qwen/qwen3.6-plus",
        )
        assert agent._anthropic_prompt_cache_policy() == (True, False)

    def test_non_qwen_non_claude_on_nous_portal_does_not_cache(self):
        # Portal scope is narrow: Claude OR Qwen only. Other models
        # routed through Portal keep their existing fall-through behavior.
        agent = _make_agent(
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
            model="openai/gpt-5.4",
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


# ─────────────────────────────────────────────────────────────────────
# Long-lived prefix cache policy (cross-session 1h tier)
# ─────────────────────────────────────────────────────────────────────

