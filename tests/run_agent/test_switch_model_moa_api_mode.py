"""Regression tests for #54259: switch_model() must reset api_mode to
``chat_completions`` when switching to the MoA virtual provider.

The MoA facade is exposed via ``MoAClient.chat.completions.create()`` and
carries the non-HTTP ``base_url`` "moa://local". When a session was
previously on a provider with a different ``api_mode`` (most commonly
``codex_responses`` from a codex slot), ``switch_model`` left
``agent.api_mode`` unchanged, and the conversation loop's primary call
went through ``client.responses.create()`` (or the OpenAI SDK bound to
``moa://local``) — returning 404 and triggering a fallback to a
reference model. The init path in ``agent_init.py`` already forces
``api_mode = "chat_completions"`` at line 724; this test pins the
matching contract on the runtime switch path.
"""

from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import switch_model


def _make_agent(current_provider, current_model, current_api_mode):
    """Bare agent object with the minimum attributes switch_model touches."""
    agent = MagicMock(name=f"Agent[{current_provider}]")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = f"https://{current_provider}.example/v1"
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = current_api_mode
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {
        "api_key": "***",
        "base_url": f"https://{current_provider}.example/v1",
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
    agent._credential_pool = None
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    return agent


class TestSwitchModelMoAApiMode:
    """Issue #54259: switching to MoA must reset api_mode to chat_completions."""

    def test_switch_to_moa_from_codex_resets_api_mode(self):
        """Session running on codex_responses → /moa must end in chat_completions.

        The original bug: api_mode stayed ``codex_responses``, the primary
        call was routed through ``client.responses.create()`` against
        ``moa://local``, the SDK returned HTTP 404, and the conversation
        loop fell back to a reference model. ``switch_model`` must
        overwrite ``api_mode`` to ``chat_completions`` when the new
        provider is MoA.
        """
        agent = _make_agent("openai-codex", "gpt-5.5", "codex_responses")

        with patch("agent.moa_loop.MoAClient") as moa_client_cls:
            switch_model(
                agent,
                new_model="frontier",
                new_provider="moa",
                api_key="",
                base_url="",
                api_mode="codex_responses",  # caller still thinks codex
            )

        assert agent.provider == "moa"
        # The MoA virtual runtime, not the old codex base_url.
        assert agent.base_url == "moa://local"
        assert agent.api_key == "moa-virtual-provider"
        # The fix: api_mode must be chat_completions so the conversation
        # loop's ``elif agent.provider == "moa"`` branch matches.
        assert agent.api_mode == "chat_completions", (
            f"switch_model did not reset api_mode for moa (got "
            f"{agent.api_mode!r}); primary call would route to "
            f"client.responses.create() and 404 on moa://local"
        )
        moa_client_cls.assert_called_once_with("frontier")

    def test_switch_to_moa_from_chat_completions_keeps_api_mode(self):
        """Switching to MoA from a chat_completions session must not regress."""
        agent = _make_agent("openrouter", "anthropic/claude-opus-4.6", "chat_completions")

        with patch("agent.moa_loop.MoAClient") as moa_client_cls:
            switch_model(
                agent,
                new_model="review",
                new_provider="moa",
                api_key="",
                base_url="",
                api_mode="chat_completions",
            )

        assert agent.provider == "moa"
        assert agent.base_url == "moa://local"
        assert agent.api_mode == "chat_completions"
        moa_client_cls.assert_called_once_with("review")

    def test_switch_to_moa_invalidates_transport_cache(self):
        """The transport cache (keyed on api_mode) must be cleared on switch."""
        agent = _make_agent("openai-codex", "gpt-5.5", "codex_responses")
        agent._transport_cache = {"old-key": object()}

        with patch("agent.moa_loop.MoAClient"):
            switch_model(
                agent,
                new_model="frontier",
                new_provider="moa",
                api_key="",
                base_url="",
                api_mode="codex_responses",
            )

        assert agent.api_mode == "chat_completions"
        # ``switch_model`` clears the transport cache when api_mode flips
        # from chat_completions to codex_responses (see line 1594-1595).
        # The MoA branch above does the flip in the other direction; the
        # init path clears it directly. Either way the cache must NOT
        # retain stale entries from the previous provider.
        assert agent._transport_cache == {}
