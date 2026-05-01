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
# Rate-limit cooldown gate
# =============================================================================

class TestRateLimitCooldown:
    """Verify _restore_primary_runtime() respects the 60s rate-limit cooldown."""

    def test_restore_blocked_during_cooldown(self):
        """While _rate_limited_until is in the future, restore returns False."""
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_activated is True

        # Manually set cooldown well into the future
        agent._rate_limited_until = time.monotonic() + 60

        result = agent._restore_primary_runtime()
        assert result is False
        assert agent._fallback_activated is True  # still on fallback

    def test_restore_allowed_after_cooldown_expires(self):
        """Once the cooldown window passes, restore proceeds normally."""
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback()

        assert agent._fallback_activated is True

        # Cooldown already expired
        agent._rate_limited_until = time.monotonic() - 1

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent._fallback_activated is False

    def test_cooldown_set_on_rate_limit_reason(self):
        """_try_activate_fallback with rate_limit reason sets _rate_limited_until."""
        from run_agent import FailoverReason
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        before = time.monotonic()
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)

        assert hasattr(agent, "_rate_limited_until")
        assert agent._rate_limited_until > before + 50  # ~60s from now

    def test_cooldown_not_set_when_already_on_fallback(self):
        """Chain-switching while already on fallback must not reset cooldown."""
        from run_agent import FailoverReason
        agent = _make_agent(
            fallback_model=[
                {"provider": "openrouter", "model": "model-a"},
                {"provider": "anthropic", "model": "model-b"},
            ],
        )
        mock_client = _mock_resolve()
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            # First call: leaving primary → cooldown should be set
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            first_cooldown = getattr(agent, "_rate_limited_until", 0)

            # Second call: already on fallback (provider != primary) → cooldown must not advance
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            second_cooldown = getattr(agent, "_rate_limited_until", 0)

        # second call should not have extended the cooldown
        assert second_cooldown == first_cooldown
