"""Tests that switch_model preserves config_context_length."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store config_context_length for later use in switch_model
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_preserves_config_context_length(mock_ctx_len):
    """When switching models, config_context_length should be passed to get_model_context_length."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify get_model_context_length was called with config_context_length
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 32_768

    # Verify compressor was updated
    assert agent.context_compressor.model == "new-model"


def test_switch_model_without_config_context_length():
    """When switching models without config override, config_context_length should be None."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx_len:
        # Switch model
        agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

        # Verify get_model_context_length was called with None
        mock_ctx_len.assert_called_once()
        call_kwargs = mock_ctx_len.call_args.kwargs
        assert call_kwargs.get("config_context_length") is None


# ---------------------------------------------------------------------------
# _check_minimum_context_length — minimum-floor reject and config override
# ---------------------------------------------------------------------------

def _make_agent_for_min_check(*, ctx_length: int, config_override) -> AIAgent:
    """Build an AIAgent stub that exposes only what _check_minimum_context_length reads."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "fake-local-model"
    agent.context_compressor = SimpleNamespace(context_length=ctx_length)
    agent._config_context_length = config_override
    return agent


def test_check_minimum_context_length_rejects_low_context_without_override():
    """A model below MINIMUM_CONTEXT_LENGTH with no config override raises."""
    low_ctx = MINIMUM_CONTEXT_LENGTH - 1
    agent = _make_agent_for_min_check(ctx_length=low_ctx, config_override=None)
    with pytest.raises(ValueError, match=r"below the minimum"):
        agent._check_minimum_context_length()


def test_check_minimum_context_length_accepts_low_context_with_override():
    """When the user explicitly sets model.context_length, the floor is bypassed.

    The error message itself promises 'set model.context_length in config.yaml
    to override' — without this bypass the override silently fails for any
    value below the floor (the exact case it exists to handle, e.g.
    hermes-brain:qwen3-14b-ctx32k at 32768 tokens).
    """
    low_ctx = 32_768
    agent = _make_agent_for_min_check(ctx_length=low_ctx, config_override=low_ctx)
    # Must not raise.
    agent._check_minimum_context_length()


def test_check_minimum_context_length_accepts_above_floor_without_override():
    """Models at or above the minimum pass without an override."""
    agent = _make_agent_for_min_check(
        ctx_length=MINIMUM_CONTEXT_LENGTH, config_override=None
    )
    agent._check_minimum_context_length()


def test_check_minimum_context_length_accepts_zero_context_length():
    """A zero/missing context_length (compressor not yet initialized) is a no-op.

    The check guards on ``ctx and ctx < MINIMUM_CONTEXT_LENGTH`` — a 0 short-
    circuits because the value is meaningless, not because it's "fine".
    """
    agent = _make_agent_for_min_check(ctx_length=0, config_override=None)
    agent._check_minimum_context_length()
