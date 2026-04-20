"""Tests that invalid context_length values in config produce visible warnings."""

from unittest.mock import patch, MagicMock, call


def _build_agent(model_cfg, custom_providers=None, model="anthropic/claude-opus-4.6"):
    """Build an AIAgent with the given model config."""
    cfg = {"model": model_cfg}
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers

    base_url = model_cfg.get("base_url", "")

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
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
    # No warning about invalid context_length
    for c in mock_logger.warning.call_args_list:
        assert "Invalid" not in str(c)


def test_string_k_suffix_context_length_warns():
    """context_length: '256K' should warn the user clearly."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({"default": "gpt5.4", "provider": "custom",
                              "base_url": "http://localhost:4000/v1",
                              "context_length": "256K"})
    assert agent._config_context_length is None
    # Should have warned
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
