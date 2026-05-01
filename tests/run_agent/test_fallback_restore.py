"""Tests for fallback provider restore behavior."""
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(quiet_mode=True):
    """Create a minimal AIAgent using the 'custom' provider pattern.

    Passes base_url and provider='custom' so that AIAgent.__init__ can
    resolve a client without a real API key, following the established
    pattern in test_primary_runtime_restore.py.
    """
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=quiet_mode,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
    return agent


# Minimal _primary_runtime snapshot for successful restore tests.
_VALID_PRIMARY_RUNTIME = {
    "model": "test-model",
    "provider": "test-provider",
    "base_url": "https://api.test.com",
    "api_mode": "openai_chat",
    "api_key": "test-key-1234567890",
    "client_kwargs": {},
    "use_prompt_caching": False,
    "compressor_model": None,
    "compressor_context_length": None,
    "compressor_base_url": None,
    "compressor_api_key": None,
    "compressor_provider": None,
}


class TestFallbackRestore:
    """Fallback restore should properly track state and notify on persistent failure."""

    def test_fallback_activated_cleared_on_successful_restore(self):
        """_fallback_activated should be False after successful restore."""
        agent = _make_agent()

        # Simulate fallback activation
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._primary_runtime = dict(_VALID_PRIMARY_RUNTIME)

        # Mock client creation and context compressor
        with (
            patch.object(agent, "_create_openai_client", return_value=MagicMock()),
            patch.object(agent.context_compressor, "update_model"),
        ):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent._fallback_activated is False
        assert agent._fallback_index == 0
        assert agent._fallback_restore_fail_count == 0

    def test_fallback_activated_remains_true_on_failed_restore(self):
        """When restore fails, _fallback_activated should remain True."""
        agent = _make_agent()

        # Simulate fallback activation with a broken primary_runtime
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._primary_runtime = {"model": None}  # Will cause KeyError in restore

        result = agent._restore_primary_runtime()

        assert result is False
        assert agent._fallback_activated is True
        assert agent._fallback_restore_fail_count >= 1

    def test_restore_fail_count_increments_on_consecutive_failures(self):
        """Consecutive restore failures should increment the counter."""
        agent = _make_agent()

        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._primary_runtime = {"model": None}  # Broken

        # First failure
        agent._restore_primary_runtime()
        count_after_first = agent._fallback_restore_fail_count

        # Second failure
        agent._fallback_activated = True  # Still in fallback
        agent._restore_primary_runtime()
        count_after_second = agent._fallback_restore_fail_count

        assert count_after_second == count_after_first + 1

    def test_restore_fail_count_resets_on_success(self):
        """Successful restore should reset the failure counter."""
        agent = _make_agent()

        # Simulate prior failures
        agent._fallback_restore_fail_count = 5

        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._primary_runtime = dict(_VALID_PRIMARY_RUNTIME)

        with (
            patch.object(agent, "_create_openai_client", return_value=MagicMock()),
            patch.object(agent.context_compressor, "update_model"),
        ):
            agent._restore_primary_runtime()

        assert agent._fallback_restore_fail_count == 0

    def test_notification_fires_exactly_once_at_threshold(self):
        """Warning notification should fire exactly once when fail count reaches 3, not on every subsequent failure."""
        agent = _make_agent(quiet_mode=False)

        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._primary_runtime = {"model": None}  # Broken

        emit_calls = []
        with patch.object(agent, "_emit_status", side_effect=lambda msg: emit_calls.append(msg)):
            # Failures 1 and 2 — no notification yet
            agent._restore_primary_runtime()
            agent._fallback_activated = True
            agent._restore_primary_runtime()
            assert len(emit_calls) == 0, "No notification expected before threshold"

            # Failure 3 — notification fires exactly once
            agent._fallback_activated = True
            agent._restore_primary_runtime()
            assert len(emit_calls) == 1, "Notification should fire exactly once at threshold"

            # Failures 4 and 5 — no additional notifications
            agent._fallback_activated = True
            agent._restore_primary_runtime()
            agent._fallback_activated = True
            agent._restore_primary_runtime()
            assert len(emit_calls) == 1, "Notification must not repeat after threshold"

    def test_switch_model_resets_restore_fail_count(self):
        """switch_model should reset _fallback_restore_fail_count along with other fallback state."""
        agent = _make_agent()

        # Simulate accumulated restore failures
        agent._fallback_restore_fail_count = 5
        agent._fallback_activated = True
        agent._fallback_index = 2

        with (
            patch.object(agent, "_create_openai_client", return_value=MagicMock()),
            patch.object(agent.context_compressor, "update_model"),
        ):
            agent.switch_model(
                new_model="new-model",
                new_provider="custom",
                api_key="test-key-1234567890",
                base_url="https://api.test.com",
            )

        assert agent._fallback_restore_fail_count == 0
        assert agent._fallback_activated is False
        assert agent._fallback_index == 0
