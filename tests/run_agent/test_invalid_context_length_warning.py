"""Tests that invalid context_length values in config produce visible warnings."""

import pytest
from unittest.mock import patch


def _build_agent(model_cfg, custom_providers=None, model="anthropic/claude-opus-4.6",
                 mock_context_length=128_000):
    """Build an AIAgent with the given model config."""
    cfg = {"model": model_cfg}
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers

    base_url = model_cfg.get("base_url", "")

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=mock_context_length),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model=model,
            api_key="test-key-1234567890",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def test_valid_integer_context_length_no_warning():
    """Plain integer context_length should work silently."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({"default": "gpt5.4", "provider": "custom",
                              "base_url": "http://localhost:4000/v1",
                              "context_length": 256000})
    assert agent._config_context_length == 256000
    for c in mock_logger.warning.call_args_list:
        assert "Invalid" not in str(c)


def test_string_k_suffix_context_length_warns():
    """context_length: '256K' should warn the user clearly."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({"default": "gpt5.4", "provider": "custom",
                              "base_url": "http://localhost:4000/v1",
                              "context_length": "256K"})
    assert agent._config_context_length is None
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "Invalid" in str(c) and "256K" in str(c)]
    assert len(warning_calls) == 1
    assert "plain integer" in str(warning_calls[0])


def test_string_numeric_context_length_works():
    """context_length: '256000' (string) should parse fine via int()."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({"default": "gpt5.4", "provider": "custom",
                              "base_url": "http://localhost:4000/v1",
                              "context_length": "256000"})
    assert agent._config_context_length == 256000
    for c in mock_logger.warning.call_args_list:
        assert "Invalid" not in str(c)


def test_custom_providers_invalid_context_length_warns():
    """Invalid context_length in custom_providers should warn."""
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"context_length": "256K"}
            },
        }
    ]
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent(
            {"default": "gpt5.4", "provider": "custom",
             "base_url": "http://localhost:4000/v1"},
            custom_providers=custom_providers,
            model="gpt5.4",
        )
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "Invalid" in str(c) and "256K" in str(c)]
    assert len(warning_calls) == 1
    assert "custom_providers" in str(warning_calls[0])


def test_custom_providers_valid_context_length():
    """Valid integer in custom_providers should work silently."""
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"context_length": 256000}
            },
        }
    ]
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent(
            {"default": "gpt5.4", "provider": "custom",
             "base_url": "http://localhost:4000/v1"},
            custom_providers=custom_providers,
            model="gpt5.4",
        )
    for c in mock_logger.warning.call_args_list:
        assert "Invalid" not in str(c)


def test_explicit_context_length_override_skips_minimum_check():
    """When user explicitly sets context_length < 64K, the minimum check
    should be skipped. Regression test for #8430.
    """
    # Mock model metadata to return a small context (below 64K minimum)
    # but user explicitly overrides via config — should NOT raise ValueError
    agent = _build_agent(
        {
            "default": "qwen3-235b",
            "provider": "custom",
            "base_url": "http://localhost:4000/v1",
            "context_length": 32768,
        },
        mock_context_length=32768,  # Simulate model with small context
    )
    # Should have stored the explicit value
    assert agent._config_context_length == 32768
    # Agent built successfully — no ValueError raised
    # (without the fix, this would raise ValueError about context below minimum)


def test_no_explicit_context_length_raises_on_small_model():
    """When user does NOT set context_length and model context is below 64K,
    ValueError should be raised. This is the negative case.
    """
    with pytest.raises(ValueError, match="context window.*below the minimum"):
        _build_agent(
            {
                "default": "small-model",
                "provider": "custom",
                "base_url": "http://localhost:4000/v1",
                # No context_length override — should trigger the check
            },
            mock_context_length=32_000,  # Model with small context
        )
