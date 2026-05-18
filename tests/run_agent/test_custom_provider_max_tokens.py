"""Tests that max_tokens is read from custom_providers per-model config.

Regression test for #28046: prior to the fix, max_tokens in
custom_providers[].models.<model> was ignored, so API calls fell back to
the 4096 default even when the user configured a higher value.
"""

from unittest.mock import patch


def _build_agent(model_cfg, custom_providers=None, model="gpt5.4"):
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


def test_custom_providers_max_tokens_applied():
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"max_tokens": 32000},
            },
        }
    ]
    agent = _build_agent(
        {"default": "gpt5.4", "provider": "custom",
         "base_url": "http://localhost:4000/v1"},
        custom_providers=custom_providers,
        model="gpt5.4",
    )
    assert agent.max_tokens == 32000


def test_custom_providers_max_tokens_string_numeric_parses():
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"max_tokens": "16000"},
            },
        }
    ]
    agent = _build_agent(
        {"default": "gpt5.4", "provider": "custom",
         "base_url": "http://localhost:4000/v1"},
        custom_providers=custom_providers,
        model="gpt5.4",
    )
    assert agent.max_tokens == 16000


def test_custom_providers_invalid_max_tokens_warns_and_stays_none():
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"max_tokens": "32K"},
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
    assert agent.max_tokens is None
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "Invalid max_tokens" in str(c) and "32K" in str(c)]
    assert len(warning_calls) == 1


def test_custom_providers_zero_max_tokens_warns_and_stays_none():
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"max_tokens": 0},
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
    assert agent.max_tokens is None
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "Invalid max_tokens" in str(c)]
    assert len(warning_calls) == 1


def test_custom_providers_no_max_tokens_leaves_none():
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"context_length": 256000},
            },
        }
    ]
    agent = _build_agent(
        {"default": "gpt5.4", "provider": "custom",
         "base_url": "http://localhost:4000/v1"},
        custom_providers=custom_providers,
        model="gpt5.4",
    )
    assert agent.max_tokens is None


def test_explicit_max_tokens_not_overridden_by_custom_providers():
    """If max_tokens already set (e.g. via CLI/model config), don't override."""
    custom_providers = [
        {
            "name": "LiteLLM",
            "base_url": "http://localhost:4000/v1",
            "models": {
                "gpt5.4": {"max_tokens": 32000},
            },
        }
    ]
    cfg = {
        "model": {"default": "gpt5.4", "provider": "custom",
                  "base_url": "http://localhost:4000/v1"},
        "custom_providers": custom_providers,
    }
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="gpt5.4",
            api_key="test-key-1234567890",
            base_url="http://localhost:4000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_tokens=8000,
        )
    assert agent.max_tokens == 8000
