"""Tests for ``_is_anthropic_oauth`` guard against third-party Anthropic-compatible providers.

The invariant: ``self._is_anthropic_oauth`` must only ever be True when
``self.provider == 'anthropic'`` (native Anthropic).  Third-party providers
that speak the Anthropic protocol (MiniMax, Zhipu GLM, Alibaba DashScope,
Kimi, LiteLLM proxies, etc.) must never trip OAuth code paths — doing so
injects Claude-Code identity headers and system prompts that cause
401/403 from those endpoints.

This test class covers all FIVE sites that assign ``_is_anthropic_oauth``:

1. ``AIAgent.__init__``                              (line ~1022)
2. ``AIAgent.switch_model``                          (line ~1832)
3. ``AIAgent._try_refresh_anthropic_client_credentials`` (line ~5335)
4. ``AIAgent._swap_credential``                      (line ~5378)
5. ``AIAgent._try_activate_fallback``                (line ~6536)
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

# A plausible-looking OAuth token (``sk-ant-`` without the ``-api`` suffix).
_OAUTH_LIKE_TOKEN = "sk-ant-oauth-example-1234567890abcdef"

@pytest.fixture
def agent():
    """Minimal AIAgent construction, skipping tool discovery."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a

class TestOAuthFlagOnRefresh:
    """Site 3 — _try_refresh_anthropic_client_credentials."""

    def test_third_party_provider_refresh_is_noop(self, agent):
        """Refresh path returns False immediately when provider != anthropic — the
        OAuth flag can never be mutated for third-party providers. Double-defended
        by the per-assignment guard at line ~5393 so future refactors can't
        reintroduce the bug."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "minimax"          # ← third-party
        agent._anthropic_api_key = "***"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        # The function short-circuits on non-anthropic providers.
        assert result is False
        # And the flag is untouched regardless.
        assert agent._is_anthropic_oauth is False

    @pytest.mark.parametrize("base_url", [
        "https://llmbox.bytedance.net",
        "http://127.0.0.1:8080/anthropic.com",  # substring spoof: the host is still foreign
        "https://llmbox.bytedance.net/anthropic",  # accepted proxy shape, but holds a custom key
    ])
    def test_third_party_endpoint_skips_refresh(self, agent, base_url):
        """provider == 'anthropic' on a third-party endpoint must not refresh: the refresh
        would swap in native Anthropic credentials the endpoint was never given."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "custom-api-key"
        agent._anthropic_base_url = base_url
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is False
        assert agent._anthropic_api_key == "custom-api-key"
        assert agent._is_anthropic_oauth is False

    @pytest.mark.parametrize("base_url", [
        "https://api.claude.com",
        "https://llm.corp.example/anthropic",
    ])
    def test_accepted_native_proxy_keeps_rotating_anthropic_token(self, agent, base_url):
        """Hosts the resolver accepts as native Anthropic already hold the Anthropic token, so
        blocking the refresh would strand an expiring OAuth token (401 with no recovery)."""
        old, new = "sk-ant-oat01-old-token-aaaaaaaa", "sk-ant-oat01-new-token-bbbbbbbb"
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = old
        agent._anthropic_base_url = base_url
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True
        agent._primary_runtime = {"anthropic_api_key": old, "is_anthropic_oauth": False}

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=new),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._anthropic_api_key == new
        # Fallback restore rebuilds from the key + flag pair, so the flag moves with the key.
        assert agent._primary_runtime == {"anthropic_api_key": new, "is_anthropic_oauth": agent._is_anthropic_oauth}

    def test_compression_before_any_request_sends_the_refreshed_token(self, agent):
        """Claude Code revokes the old token on refresh. Manual /compress and turn-start compaction
        run before any main request, so compress_context must refresh first and move every holder
        (the compressor forwards its OWN ``api_key``), or the summary 401s for the session's life."""
        from agent.conversation_compression import compress_context

        old, new = "sk-ant-oat01-old", "sk-ant-oat01-new"
        seen, announced = [], []
        agent.api_mode, agent.provider = "anthropic_messages", "anthropic"
        agent._anthropic_base_url = "https://api.anthropic.com"
        agent._anthropic_client = MagicMock()
        agent.api_key = agent._anthropic_api_key = agent.context_compressor.api_key = old
        agent._primary_runtime = {"api_key": old, "anthropic_api_key": old, "compressor_api_key": old}
        agent._compression_feasibility_checked = True
        cc = agent.context_compressor

        def fake_compress(_messages, **_kwargs):
            seen.append(cc.api_key)
            return [{"role": "user", "content": "[summary]"}, {"role": "assistant", "content": "tail"}]

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=new),
            patch.object(AIAgent, "_build_direct_anthropic_client", return_value=MagicMock()),
            patch.object(cc, "compress", side_effect=fake_compress),
            patch.object(agent, "_emit_status", side_effect=lambda _s: announced.append(agent._anthropic_api_key)),
        ):
            compress_context(agent, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                             "system", approx_tokens=100_000, force=True)

        assert seen == [new]
        assert announced[:1] == [old]  # the Desktop announce lands before the (possibly blocking) refresh
        assert agent._anthropic_api_key == agent.api_key == new
        assert agent._primary_runtime == {"api_key": new, "anthropic_api_key": new, "compressor_api_key": new,
                                          "is_anthropic_oauth": agent._is_anthropic_oauth}

    def test_auxiliary_main_route_uses_refreshed_token(self, agent):
        """Production order: the turn publishes its aux runtime BEFORE the first request triggers
        the silent refresh, so same-turn `auto` aux calls must still pick up the new token."""
        from agent import auxiliary_client as aux
        from agent.auxiliary_key_rotation import rotate_runtime_main_api_key
        from agent.chat_completion_helpers import _context_thread_target
        from agent.turn_context import _publish_runtime_main

        old, new = "sk-ant...aaaa", "sk-ant...bbbb"
        agent.api_mode, agent.provider, agent.model = "anthropic_messages", "anthropic", "claude-opus-4-6"
        agent.base_url = agent._anthropic_base_url = "https://api.anthropic.com"
        agent.api_key = agent._anthropic_api_key = old
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True
        seen = {}

        def fake_resolve(provider, model, explicit_api_key=None, **kwargs):
            seen["api_key"] = explicit_api_key
            return MagicMock(), model

        refreshed = []
        try:
            _publish_runtime_main(agent)
            with (
                patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=new),
                patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            ):
                # The refresh runs in the request worker's copied Context; the turn thread reads after.
                worker = threading.Thread(target=_context_thread_target(
                    lambda: refreshed.append(agent._try_refresh_anthropic_client_credentials())))
                worker.start()
                worker.join()
            assert refreshed == [True]
            runtime = aux._normalize_main_runtime(None)
            assert runtime.get("api_key") == new
            with (
                patch.object(aux, "resolve_provider_client", side_effect=fake_resolve),
                patch.object(aux, "_is_provider_unhealthy", return_value=False),
            ):
                aux._try_main_provider_route(
                    "anthropic", agent.model, runtime.get("base_url", ""), runtime.get("api_key"), "anthropic_messages",
                )
            assert seen["api_key"] == new
            # A scoped runtime rotates in place; the legacy mirrors are never republished by a rotation.
            with aux.scoped_runtime_main({"provider": "anthropic", "api_key": new, "model": "m"}):
                rotate_runtime_main_api_key(new, "sk-ant...cccc")
                assert aux._RUNTIME_MAIN_CONTEXT.get()["api_key"] == "sk-ant...cccc"
            assert (aux._RUNTIME_MAIN_MODEL, aux._RUNTIME_MAIN_API_KEY) == (agent.model, old)
            assert aux._compat_runtime_main() is None  # unchanged mirrors never become a runtime input
        finally:
            aux.clear_runtime_main()


class TestOAuthFlagOnCredentialSwap:
    """Site 4 — _swap_credential (credential pool rotation)."""

    def test_pool_swap_on_third_party_never_flips_oauth(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.provider = "glm"              # ← Zhipu GLM via /anthropic
        agent._anthropic_api_key = "old-key"
        agent._anthropic_base_url = "https://open.bigmodel.cn/api/anthropic"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        entry = MagicMock()
        entry.runtime_api_key = _OAUTH_LIKE_TOKEN
        entry.runtime_base_url = "https://open.bigmodel.cn/api/anthropic"

        with patch("agent.anthropic_adapter.build_anthropic_client",
                   return_value=MagicMock()):
            agent._swap_credential(entry)

        assert agent._is_anthropic_oauth is False

class TestOAuthFlagOnConstruction:
    """Site 1 — AIAgent.__init__ on a third-party anthropic_messages provider."""

    def test_minimax_init_does_not_flip_oauth(self):
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            # Simulate a stale ANTHROPIC_TOKEN in the env — the init code
            # MUST NOT fall back to it when provider != anthropic.
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
        ):
            agent = AIAgent(
                api_key="minimax-key-1234",
                base_url="https://api.minimax.io/anthropic",
                provider="minimax",
                api_mode="anthropic_messages",
                model="claude-sonnet-4-6",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        # The effective key should be the explicit minimax-key, not the
        # stale Anthropic OAuth token, and the OAuth flag must be False.
        assert agent._anthropic_api_key == "minimax-key-1234"
        assert agent._is_anthropic_oauth is False
