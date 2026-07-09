"""Regression tests for #61296: switch_model() must not trust a stale non-empty
base_url when the caller is switching to a different provider.

The empty-string guard (#52727) prevents an Ollama localhost URL from leaking
into a cloud provider that returns "" for base_url. But it does not catch a
truthy *stale* URL: when ``switch_model(new_provider='minimax-cn',
base_url='https://ollama.com/v1', ...)`` lands, the truthy check passes and
the agent ends up with the new provider's API key paired against the old
provider's endpoint — silent credential mis-routing.

The fix detects ``base_url == old_base_url`` AND a provider change, then drops
the stale URL so the new provider's ``set_next_api_call_base_url()`` resolves
from registry.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import switch_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_agent(
    current_provider,
    current_model,
    current_base_url,
    current_pool=None,
):
    """Bare agent object with the minimum attributes switch_model touches.

    ``current_pool`` defaults to None so the helper mirrors a real boot
    state; the same-provider tests construct an agent WITH a pool to
    pin down the no-reload contract from #52727.
    """
    agent = MagicMock(name=f"Agent[{current_provider}]")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = current_base_url
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {
        "api_key": "***",
        "base_url": current_base_url,
    }
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._cached_system_prompt = "cached-system-prompt"
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._credential_pool = current_pool
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    return agent


def _make_pool(provider):
    pool = MagicMock(name=f"Pool[{provider}]")
    pool.provider = provider
    return pool


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------


class TestSwitchModelStaleBaseUrl:
    """Issue #61296: switch_model must reject a non-empty stale base_url when
    the provider is changing. Pre-fix the truthy `if base_url:` guard trusted
    it and produced cross-wired agents (new provider key → old provider URL).
    """

    def test_old_provider_url_rejected_on_provider_change(self):
        """Caller repeats the OLD provider's URL alongside a NEW provider.

        Before the fix: agent.base_url == 'https://ollama.com/v1' after the
        switch — agent now serves minimax-cn traffic through ollama.com.
        After the fix: agent.base_url is reset to '' so the new
        provider's registry provides a fresh endpoint.
        """
        agent = _make_agent(
            current_provider="ollama-cloud",
            current_model="qwen3:32b",
            current_base_url="https://ollama.com/v1",
        )

        with patch(
            "agent.credential_pool.load_pool",
            return_value=MagicMock(provider="minimax-cn"),
        ):
            switch_model(
                agent,
                new_model="MiniMax-M3",
                new_provider="minimax-cn",
                api_key="minimax-secret-key",
                base_url="https://ollama.com/v1",  # stale, truthy
                api_mode="chat_completions",
            )

        assert agent.provider == "minimax-cn"
        # The whole point: base_url must NOT be the old provider's URL.
        assert agent.base_url != "https://ollama.com/v1", (
            f"switch_model cross-wired agent: provider={agent.provider!r} "
            f"but base_url={agent.base_url!r} is still the old provider's "
            "endpoint (#61296)"
        )
        # And the new provider's key was applied.
        assert agent.api_key == "minimax-secret-key"

    def test_same_provider_same_url_kept(self):
        """A same-provider re-selection with the same URL is NOT stale.

        The caller is re-affirming the current URL, possibly with a
        different model. Nothing has gone stale. The fix's guard must
        NOT fire and base_url must stay verbatim.
        """
        agent = _make_agent(
            current_provider="groq",
            current_model="llama-3.3-70b",
            current_base_url="https://api.groq.com/openai/v1",
            current_pool=_make_pool("groq"),
        )

        with patch("agent.credential_pool.load_pool") as load_pool_mock:
            switch_model(
                agent,
                new_model="llama-3.1-8b",
                new_provider="groq",  # SAME
                api_key="groq-key-new",
                base_url="https://api.groq.com/openai/v1",  # SAME
                api_mode="chat_completions",
            )

        assert agent.base_url == "https://api.groq.com/openai/v1"
        # Same-provider switches do not reload the pool (#52727).
        load_pool_mock.assert_not_called()

    def test_same_provider_different_url_accepted(self):
        """Caller switches within the same provider to a different endpoint.

        E.g. user re-points groq at a regional proxy or self-hosted mirror.
        base_url differs from the old URL → not the stale pattern.
        """
        agent = _make_agent(
            current_provider="groq",
            current_model="llama-3.3-70b",
            current_base_url="https://api.groq.com/openai/v1",
            current_pool=_make_pool("groq"),
        )

        with patch("agent.credential_pool.load_pool") as load_pool_mock:
            switch_model(
                agent,
                new_model="llama-3.3-70b",
                new_provider="groq",
                api_key="groq-key-new",
                base_url="https://groq-proxy.internal/v1",  # different
                api_mode="chat_completions",
            )

        assert agent.base_url == "https://groq-proxy.internal/v1"
        load_pool_mock.assert_not_called()

    def test_different_provider_different_url_accepted(self):
        """Caller passes a brand new URL alongside a brand new provider.

        Healthy case — neither side matches the old agent state. Not stale.
        """
        agent = _make_agent(
            current_provider="ollama-cloud",
            current_model="qwen3:32b",
            current_base_url="https://ollama.com/v1",
        )

        with patch(
            "agent.credential_pool.load_pool",
            return_value=MagicMock(provider="minimax-cn"),
        ):
            switch_model(
                agent,
                new_model="MiniMax-M3",
                new_provider="minimax-cn",
                api_key="minimax-secret-key",
                base_url="https://api.minimax.com/v1",  # different from old
                api_mode="chat_completions",
            )

        assert agent.base_url == "https://api.minimax.com/v1"
        assert agent.provider == "minimax-cn"

    def test_agent_without_existing_base_url_not_triggered(self):
        """An agent with no previous base_url never matches the stale-guard.

        Covers boot-time switches where the agent started blank. Almost
        all of those also change providers, but the guard must be a no-op
        rather than spuriously reset a URL the caller just typed.
        """
        agent = _make_agent(
            current_provider="",
            current_model="",
            current_base_url="",
        )

        with patch(
            "agent.credential_pool.load_pool",
            return_value=MagicMock(provider="minimax-cn"),
        ):
            switch_model(
                agent,
                new_model="MiniMax-M3",
                new_provider="minimax-cn",
                api_key="minimax-secret-key",
                base_url="https://api.minimax.com/v1",
                api_mode="chat_completions",
            )

        # Caller-provided URL is kept verbatim on the boot path.
        assert agent.base_url == "https://api.minimax.com/v1"
