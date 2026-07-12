"""Regression tests for the ``switch_model`` Anthropic-on-Vertex construction site.

``agent/agent_runtime_helpers.py::switch_model`` handles the mid-session
``/model`` swap. Its ``anthropic_messages`` branch used to unconditionally
call ``build_anthropic_client(...)`` — which is the wrong client for
Anthropic Claude on Vertex, because that path talks to the native
Anthropic endpoint with a static API key. A Gemini-on-Vertex session
that switched to ``anthropic/claude-opus-4-8`` mid-conversation would
build the native client with no API key on the deployment, hitting a
401 (or worse, silently succeeding against a stale-cached one).

The fix (reviewed on the upstream PR that added the Anthropic-on-Vertex
provider) adds a ``new_provider == "vertex"`` branch that builds an
``AnthropicVertex`` client instead, resolving project/region freshly
via ``get_anthropic_vertex_config()`` so a session that started on a
non-Vertex provider can still swap to Vertex Claude cleanly.

These tests exercise the switch on a live agent facade and pin the
invariants:

* ``build_anthropic_vertex_client`` is the client factory called on
  the switch — NOT ``build_anthropic_client``.
* Project/region are populated from ``get_anthropic_vertex_config()``
  and stashed on the agent (``_vertex_project_id`` /
  ``_vertex_region``) for subsequent rebuild sites.
* Non-Vertex ``anthropic_messages`` providers (native Anthropic,
  MiniMax, ...) still take the ``build_anthropic_client`` path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import switch_model


def _make_agent(current_provider: str, current_model: str, current_api_mode: str = "chat_completions"):
    """Bare agent facade with just the attributes ``switch_model`` reads/writes.

    Mirrors the fixture shape used in
    ``tests/run_agent/test_switch_model_pool_reload_52727.py``; kept
    local so a future refactor of the pool-reload fixture doesn't
    accidentally re-shape this test's expectations.
    """
    agent = MagicMock(name=f"Agent[{current_provider}]")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = f"https://{current_provider}.example/v1"
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = current_api_mode
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {}
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._vertex_project_id = None
    agent._vertex_region = None
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


class TestSwitchModelVertexAnthropic:
    """The Vertex Claude construction path — the review-caught gap."""

    def test_gemini_on_vertex_to_claude_on_vertex_builds_anthropic_vertex(self):
        """The interesting real-world flow: session is happily talking to
        Gemini-on-Vertex, user runs ``/model anthropic/claude-opus-4-8``,
        rebuild constructs an AnthropicVertex client with the resolved
        project/region.  Pre-fix: this called build_anthropic_client and
        broke the session.
        """
        agent = _make_agent(
            current_provider="vertex",
            current_model="google/gemini-3.1-pro-preview",
            current_api_mode="chat_completions",
        )
        vertex_client_sentinel = MagicMock(name="AnthropicVertexClient")
        native_client_sentinel = MagicMock(name="NativeAnthropicClient")

        with (
            patch(
                "agent.anthropic_vertex_adapter.get_anthropic_vertex_config",
                return_value=("khala-498208", "global"),
            ),
            patch(
                "agent.anthropic_vertex_adapter.build_anthropic_vertex_client",
                return_value=vertex_client_sentinel,
            ) as build_vertex_mock,
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=native_client_sentinel,
            ) as build_native_mock,
            patch("agent.credential_pool.load_pool", return_value=None),
        ):
            switch_model(
                agent,
                new_model="anthropic/claude-opus-4-8",
                new_provider="vertex",
                api_key="",
                base_url="",
                api_mode="anthropic_messages",
            )

        # Vertex client factory fires; native factory does not.
        assert build_vertex_mock.called, (
            "AnthropicVertex must be constructed on a vertex + anthropic_messages "
            "switch — build_anthropic_vertex_client was not called."
        )
        assert not build_native_mock.called, (
            "build_anthropic_client MUST NOT fire on the vertex path — "
            "it would send Anthropic-native traffic to the wrong endpoint "
            "with the wrong auth."
        )
        call_kwargs = build_vertex_mock.call_args
        pos = call_kwargs.args
        # build_anthropic_vertex_client is called positionally: (project_id, region)
        # plus a timeout kwarg.
        assert pos[0] == "khala-498208"
        assert pos[1] == "global"
        assert "timeout" in call_kwargs.kwargs

        # State reflects the new client + resolved config.
        assert agent._anthropic_client is vertex_client_sentinel
        assert agent._vertex_project_id == "khala-498208"
        assert agent._vertex_region == "global"
        # AnthropicVertex handles bearer minting internally; the api_key
        # slot carries a non-empty placeholder so downstream "auth
        # resolved" checks pass without paperwork-shuffling.
        assert agent.api_key == "vertex-adc"
        assert agent._anthropic_api_key == "vertex-adc"
        # Not a native-Anthropic OAuth session.
        assert agent._is_anthropic_oauth is False
        # OpenAI-shaped client is cleared — this session is now on the
        # Anthropic Messages path.
        assert agent.client is None
        # And the agent's provider/model/api_mode reflect the new state.
        assert agent.provider == "vertex"
        assert agent.model == "anthropic/claude-opus-4-8"
        assert agent.api_mode == "anthropic_messages"

    def test_none_region_from_config_falls_back_to_global(self):
        """If get_anthropic_vertex_config returns region=None (rare —
        credentials with no region and no VERTEX_REGION env), the switch
        must not leave the client with region=None. It falls back to
        'global' matching the runtime-provider default."""
        agent = _make_agent("openrouter", "anthropic/claude-opus-4-8")
        with (
            patch(
                "agent.anthropic_vertex_adapter.get_anthropic_vertex_config",
                return_value=("khala-498208", None),
            ),
            patch(
                "agent.anthropic_vertex_adapter.build_anthropic_vertex_client",
                return_value=MagicMock(),
            ) as build_vertex_mock,
            patch("agent.credential_pool.load_pool", return_value=None),
        ):
            switch_model(
                agent,
                new_model="anthropic/claude-opus-4-8",
                new_provider="vertex",
                api_mode="anthropic_messages",
            )

        assert agent._vertex_region == "global"
        # And the factory saw the fallback too.
        assert build_vertex_mock.call_args.args[1] == "global"

    def test_native_anthropic_switch_still_uses_build_anthropic_client(self):
        """Regression guard: the fix must not accidentally affect switches
        onto the native Anthropic provider or other non-Vertex
        anthropic_messages providers (MiniMax, Alibaba, ...). Those still
        go through build_anthropic_client."""
        agent = _make_agent("openrouter", "anthropic/claude-opus-4-8")
        with (
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(name="NativeAnthropicClient"),
            ) as build_native_mock,
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="sk-ant-token-here",
            ),
            patch(
                "agent.anthropic_vertex_adapter.build_anthropic_vertex_client",
                return_value=MagicMock(name="AnthropicVertexClient"),
            ) as build_vertex_mock,
            patch("agent.credential_pool.load_pool", return_value=None),
        ):
            switch_model(
                agent,
                new_model="claude-opus-4-8",
                new_provider="anthropic",
                api_key="",
                base_url="https://api.anthropic.com",
                api_mode="anthropic_messages",
            )

        assert build_native_mock.called
        assert not build_vertex_mock.called
