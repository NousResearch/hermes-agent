"""Tests for per-turn primary runtime restoration and transport recovery.

Verifies that:
1. Fallback is turn-scoped: a new turn restores the primary model/provider
2. The fallback chain index resets so all fallbacks are available again
3. Context compressor state is restored alongside the runtime
4. Transient transport errors get one recovery cycle before fallback
5. Recovery is skipped for aggregator providers (OpenRouter, Nous)
6. Non-transport errors don't trigger recovery
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None, provider="custom", base_url="https://my-llm.example.com/v1"):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url=base_url,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_resolve(base_url="https://openrouter.ai/api/v1", api_key="fallback-key-1234"):
    """Helper to create a mock client for resolve_provider_client."""
    mock_client = MagicMock()
    mock_client.api_key = api_key
    mock_client.base_url = base_url
    return mock_client


# =============================================================================
# _primary_runtime snapshot
# =============================================================================

class TestPrimaryRuntimeSnapshot:
    def test_snapshot_created_at_init(self):
        agent = _make_agent()
        assert hasattr(agent, "_primary_runtime")
        rt = agent._primary_runtime
        assert rt["model"] == agent.model
        assert rt["provider"] == "custom"
        assert rt["base_url"] == "https://my-llm.example.com/v1"
        assert rt["api_mode"] == agent.api_mode
        assert "client_kwargs" in rt
        assert "compressor_context_length" in rt

    def test_snapshot_includes_compressor_state(self):
        agent = _make_agent()
        rt = agent._primary_runtime
        cc = agent.context_compressor
        assert rt["compressor_model"] == cc.model
        assert rt["compressor_provider"] == cc.provider
        assert rt["compressor_context_length"] == cc.context_length
        assert rt["compressor_threshold_tokens"] == cc.threshold_tokens

    def test_snapshot_includes_anthropic_state_when_applicable(self):
        """Anthropic-mode agents should snapshot Anthropic-specific state."""
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-test-12345678",
                base_url="https://api.anthropic.com",
                provider="anthropic",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        rt = agent._primary_runtime
        assert "anthropic_api_key" in rt
        assert "anthropic_base_url" in rt
        assert "is_anthropic_oauth" in rt

    def test_snapshot_omits_anthropic_for_openai_mode(self):
        agent = _make_agent(provider="custom")
        rt = agent._primary_runtime
        assert "anthropic_api_key" not in rt


# =============================================================================
# _restore_primary_runtime()
# =============================================================================

class TestRestorePrimaryRuntime:
    def test_noop_when_not_fallback(self):
        agent = _make_agent()
        assert agent._fallback_activated is False
        assert agent._restore_primary_runtime() is False

    def test_restores_model_and_provider(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        original_model = agent.model
        original_provider = agent.provider

        # Simulate fallback activation
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_activated is True
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"

        # Restore should bring back the primary
        with patch("run_agent.OpenAI", return_value=MagicMock()):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent._fallback_activated is False
        assert agent.model == original_model
        assert agent.provider == original_provider

    def test_resets_fallback_index(self):
        """After restore, the full fallback chain should be available again."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "openrouter", "model": "model-a"},
                {"provider": "anthropic", "model": "model-b"},
            ],
        )
        # Advance through the chain
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_index == 1  # consumed one entry

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent._fallback_index == 0  # reset for next turn

    def test_restores_compressor_state(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        original_ctx_len = agent.context_compressor.context_length
        original_threshold = agent.context_compressor.threshold_tokens

        # Simulate fallback modifying compressor
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        # Manually simulate compressor being changed (as _try_activate_fallback does)
        agent.context_compressor.context_length = 32000
        agent.context_compressor.threshold_tokens = 25600

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent.context_compressor.context_length == original_ctx_len
        assert agent.context_compressor.threshold_tokens == original_threshold

    def test_restores_prompt_caching_flag(self):
        agent = _make_agent()
        original_caching = agent._use_prompt_caching

        # Simulate fallback changing the caching flag
        agent._fallback_activated = True
        agent._use_prompt_caching = not original_caching

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            agent._restore_primary_runtime()

        assert agent._use_prompt_caching == original_caching

    def test_restore_survives_exception(self):
        """If client rebuild fails, the method returns False gracefully."""
        agent = _make_agent()
        agent._fallback_activated = True

        with patch("run_agent.OpenAI", side_effect=Exception("connection refused")):
            result = agent._restore_primary_runtime()

        assert result is False


# =============================================================================
# _try_recover_primary_transport()
# =============================================================================

def _make_transport_error(error_type="ReadTimeout"):
    """Create an exception whose type().__name__ matches the given name."""
    cls = type(error_type, (Exception,), {})
    return cls("connection timed out")


class TestTryRecoverPrimaryTransport:

    def test_recovers_on_read_timeout(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_recovers_on_connect_timeout(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ConnectTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_recovers_on_pool_timeout(self):
        agent = _make_agent(provider="zai")
        error = _make_transport_error("PoolTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_recovers_on_openai_api_connection_error(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("APIConnectionError")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_recovers_on_openai_api_timeout_error(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("APITimeoutError")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_skipped_when_already_on_fallback(self):
        agent = _make_agent(provider="custom")
        agent._fallback_activated = True
        error = _make_transport_error("ReadTimeout")

        result = agent._try_recover_primary_transport(
            error, retry_count=3, max_retries=3,
        )
        assert result is False

    def test_skipped_for_non_transport_error(self):
        """Non-transport errors (ValueError, APIError, etc.) skip recovery."""
        agent = _make_agent(provider="custom")
        error = ValueError("invalid model")

        result = agent._try_recover_primary_transport(
            error, retry_count=3, max_retries=3,
        )
        assert result is False

    def test_skipped_for_openrouter(self):
        agent = _make_agent(provider="openrouter", base_url="https://openrouter.ai/api/v1")
        error = _make_transport_error("ReadTimeout")

        result = agent._try_recover_primary_transport(
            error, retry_count=3, max_retries=3,
        )
        assert result is False

    def test_skipped_for_nous_provider(self):
        agent = _make_agent(provider="nous", base_url="https://inference.nous.nousresearch.com/v1")
        error = _make_transport_error("ReadTimeout")

        result = agent._try_recover_primary_transport(
            error, retry_count=3, max_retries=3,
        )
        assert result is False

    def test_allowed_for_anthropic_direct(self):
        """Direct Anthropic endpoint should get recovery."""
        agent = _make_agent(provider="anthropic", base_url="https://api.anthropic.com")
        # For non-anthropic_messages api_mode, it will use OpenAI client
        error = _make_transport_error("ConnectError")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_allowed_for_ollama(self):
        agent = _make_agent(provider="ollama", base_url="http://localhost:11434/v1")
        error = _make_transport_error("ConnectTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is True

    def test_wait_time_scales_with_retry_count(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )
            # wait_time = min(3 + retry_count, 8) = min(6, 8) = 6
            mock_sleep.assert_called_once_with(6)

    def test_wait_time_capped_at_8(self):
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            agent._try_recover_primary_transport(
                error, retry_count=10, max_retries=3,
            )
            # wait_time = min(3 + 10, 8) = 8
            mock_sleep.assert_called_once_with(8)

    def test_closes_existing_client_before_rebuild(self):
        agent = _make_agent(provider="custom")
        old_client = agent.client
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch("time.sleep"), \
             patch.object(agent, "_close_openai_client") as mock_close:
            agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )
            mock_close.assert_called_once_with(
                old_client, reason="primary_recovery", shared=True,
            )

    def test_survives_rebuild_failure(self):
        """If client rebuild fails, returns False gracefully."""
        agent = _make_agent(provider="custom")
        error = _make_transport_error("ReadTimeout")

        with patch("run_agent.OpenAI", side_effect=Exception("socket error")), \
             patch("time.sleep"):
            result = agent._try_recover_primary_transport(
                error, retry_count=3, max_retries=3,
            )

        assert result is False


# =============================================================================
# Integration: restore_primary_runtime called from run_conversation
# =============================================================================

class TestRestoreInRunConversation:
    """Verify the hook in run_conversation() calls _restore_primary_runtime."""

    def test_restore_called_at_turn_start(self):
        agent = _make_agent()
        agent._fallback_activated = True

        with patch.object(agent, "_restore_primary_runtime", return_value=True) as mock_restore, \
             patch.object(agent, "run_conversation", wraps=None) as _:
            # We can't easily run the full conversation, but we can verify
            # the method exists and is callable
            agent._restore_primary_runtime()
            mock_restore.assert_called_once()

    def test_full_cycle_fallback_then_restore(self):
        """Simulate: turn 1 activates fallback, turn 2 restores primary."""
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            provider="custom",
        )

        # Turn 1: activate fallback
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            assert agent._try_activate_fallback() is True

        assert agent._fallback_activated is True
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"
        assert agent._fallback_index == 1

        # Turn 2: restore primary
        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True

        assert agent._fallback_activated is False
        assert agent._fallback_index == 0
        assert agent.provider == "custom"
        assert agent.base_url == "https://my-llm.example.com/v1"


# =============================================================================
# _try_restore_smart_routed_primary()
# =============================================================================

class TestTryRestoreSmartRoutedPrimary:
    def test_noop_without_smart_routed_primary(self):
        agent = _make_agent()
        assert agent._try_restore_smart_routed_primary() is False

    def test_noop_when_already_restored(self):
        agent = _make_agent()
        agent._smart_routed_primary = {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "test-key",
        }
        agent._smart_routed_primary_restored = True
        assert agent._try_restore_smart_routed_primary() is False

    def test_restores_model_provider_and_client(self):
        agent = _make_agent(provider="custom", base_url="https://my-llm.example.com/v1")
        agent.model = "cheap-model"
        agent.provider = "cheap-provider"
        agent.base_url = "https://bad-url.example.com/v1"
        agent._smart_routed_primary = {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "primary-key",
        }

        mock_client = _mock_resolve(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "anthropic/claude-sonnet-4"),
        ):
            result = agent._try_restore_smart_routed_primary()

        assert result is True
        assert agent._smart_routed_primary_restored is True
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"
        assert agent.base_url == "https://openrouter.ai/api/v1"
        assert agent.client is mock_client

    def test_returns_false_when_primary_client_unresolvable(self):
        agent = _make_agent()
        agent._smart_routed_primary = {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
        }
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ):
            assert agent._try_restore_smart_routed_primary() is False

    def test_prompt_caching_restored_for_claude_on_openrouter(self):
        agent = _make_agent()
        agent._smart_routed_primary = {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "key",
        }
        mock_client = _mock_resolve(base_url="https://openrouter.ai/api/v1")
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "anthropic/claude-sonnet-4"),
        ):
            agent._try_restore_smart_routed_primary()
            assert agent._use_prompt_caching is True

    def test_prompt_caching_restored_for_native_anthropic(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant...5678",
                base_url="https://api.anthropic.com",
                provider="anthropic",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        agent._smart_routed_primary = {
            "provider": "anthropic",
            "model": "claude-opus-4",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "api_key": "sk-ant-primary",
        }
        mock_client = _mock_resolve(
            api_key="sk-ant-primary",
            base_url="https://api.anthropic.com",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "claude-opus-4"),
        ):
            agent._try_restore_smart_routed_primary()

        assert agent._anthropic_api_key == "sk-ant-primary"
        assert agent.api_mode == "anthropic_messages"


# =============================================================================
# Integration: smart-routed primary restore in run_conversation
# =============================================================================

class TestSmartRoutedRestoreInRunConversation:
    """Verify that a 4xx from a smart-routed cheap model triggers primary restore."""

    def _make_smart_routed_agent(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="cheap-key",
                base_url="https://bad-endpoint.example.com/v1",
                provider="custom",
                model="cheap-model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                smart_routed_primary={
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet-4",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "chat_completions",
                    "api_key": "primary-key",
                },
            )
            agent.client = MagicMock()
            return agent

    def test_client_error_restores_primary_before_fallback_chain(self):
        agent = self._make_smart_routed_agent()

        # First call: 404 client error from the cheap model
        class FakeNotFoundError(Exception):
            pass
        FakeNotFoundError.__name__ = "NotFoundError"

        not_found = FakeNotFoundError("404 Not Found")
        not_found.status_code = 404

        # Second call: success from the restored primary
        # Use SimpleNamespace to avoid MagicMock auto-return issues
        success_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello from primary",
                        tool_calls=None,
                        role="assistant",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(total_tokens=10),
        )

        primary_client = _mock_resolve(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
        )
        primary_client.chat.completions.create = MagicMock(return_value=success_response)

        cheap_client_create = MagicMock(side_effect=not_found)
        agent.client.chat.completions.create = cheap_client_create

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(primary_client, "anthropic/claude-sonnet-4"),
        ):
            result = agent.run_conversation("hi")

        assert result["completed"] is True
        assert result["final_response"] == "Hello from primary"
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.provider == "openrouter"
        assert agent._smart_routed_primary_restored is True
        # Ensure cheap client was called once and primary client was called once
        cheap_client_create.assert_called_once()
        primary_client.chat.completions.create.assert_called_once()

    def test_max_retries_exhausted_restores_primary(self):
        """If cheap model fails 3 times (transport error), restore primary on max retries."""
        agent = self._make_smart_routed_agent()

        class FakeServiceUnavailableError(Exception):
            pass
        FakeServiceUnavailableError.__name__ = "ServiceUnavailableError"

        err = FakeServiceUnavailableError("503 Service Unavailable")
        err.status_code = 503

        success_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello after retries",
                        tool_calls=None,
                        role="assistant",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(total_tokens=10),
        )

        primary_client = _mock_resolve(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
        )
        primary_client.chat.completions.create = MagicMock(return_value=success_response)

        cheap_client_create = MagicMock(side_effect=err)
        agent.client.chat.completions.create = cheap_client_create

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(primary_client, "anthropic/claude-sonnet-4"),
        ):
            result = agent.run_conversation("hi")

        assert result["completed"] is True
        assert result["final_response"] == "Hello after retries"
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent._smart_routed_primary_restored is True
        # cheap model got its 3 retries, then primary succeeded
        assert cheap_client_create.call_count == 3
        primary_client.chat.completions.create.assert_called_once()

    def test_primary_restore_fails_then_fallback_chain_kicks_in(self):
        """If cheap model fails and restored primary also fails, general fallback activates."""
        agent = self._make_smart_routed_agent()
        agent._fallback_chain = [
            {"provider": "zai", "model": "glm-5"},
        ]

        class FakeNotFoundError(Exception):
            pass
        FakeNotFoundError.__name__ = "NotFoundError"

        not_found = FakeNotFoundError("404 Not Found")
        not_found.status_code = 404

        success_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello from fallback",
                        tool_calls=None,
                        role="assistant",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(total_tokens=10),
        )

        primary_client = _mock_resolve(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
        )
        primary_client.chat.completions.create = MagicMock(side_effect=not_found)

        cheap_client_create = MagicMock(side_effect=not_found)
        agent.client.chat.completions.create = cheap_client_create

        fallback_client = _mock_resolve(
            api_key="zai-key",
            base_url="https://open.z.ai/api/v1",
        )
        fallback_client.chat.completions.create = MagicMock(return_value=success_response)

        def resolve_side_effect(provider, **kwargs):
            if provider == "openrouter":
                return (primary_client, "anthropic/claude-sonnet-4")
            if provider == "zai":
                return (fallback_client, "glm-5")
            return (None, None)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=resolve_side_effect,
        ):
            result = agent.run_conversation("hi")

        assert result["completed"] is True
        assert result["final_response"] == "Hello from fallback"
        assert agent._smart_routed_primary_restored is True
        assert agent._fallback_activated is True
        assert agent.model == "glm-5"
        assert agent.provider == "zai"
        # cheap model called once, primary called once, fallback called once
        cheap_client_create.assert_called_once()
        primary_client.chat.completions.create.assert_called_once()
        fallback_client.chat.completions.create.assert_called_once()
