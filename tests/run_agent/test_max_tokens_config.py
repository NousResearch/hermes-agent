"""Tests that model.max_tokens config is honored and validated.

Regression for issue #19360: when routing to Anthropic via an
OpenAI-compatible proxy with tools enabled, Anthropic's Messages API
requires ``max_tokens`` in the request body. Previously there was no
way to set it from config.yaml — the AIAgent constructor accepted
``max_tokens`` but no caller ever passed it, so ``self.max_tokens``
was always ``None`` and the OpenAI SDK omitted the field.
"""

from unittest.mock import patch


def _build_agent(model_cfg, model="anthropic/claude-opus-4.6"):
    cfg = {"model": model_cfg}
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


def test_valid_integer_max_tokens_applied():
    """Plain integer max_tokens should land on self.max_tokens."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({
            "default": "claude-opus-4-6",
            "provider": "custom",
            "base_url": "http://localhost:8045/v1",
            "max_tokens": 16384,
        })
    assert agent.max_tokens == 16384
    for c in mock_logger.warning.call_args_list:
        assert "max_tokens" not in str(c) or "Invalid" not in str(c)


def test_string_numeric_max_tokens_parses():
    """max_tokens: '4096' (string) should parse via int()."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({
            "default": "claude-opus-4-6",
            "provider": "custom",
            "base_url": "http://localhost:8045/v1",
            "max_tokens": "4096",
        })
    assert agent.max_tokens == 4096
    for c in mock_logger.warning.call_args_list:
        assert "Invalid" not in str(c) or "max_tokens" not in str(c)


def test_invalid_k_suffix_max_tokens_warns():
    """max_tokens: '4K' is not a plain integer — warn and fall back."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({
            "default": "claude-opus-4-6",
            "provider": "custom",
            "base_url": "http://localhost:8045/v1",
            "max_tokens": "4K",
        })
    assert agent.max_tokens is None
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "max_tokens" in str(c) and "Invalid" in str(c)]
    assert len(warning_calls) == 1
    assert "4K" in str(warning_calls[0])


def test_nonpositive_max_tokens_warns():
    """max_tokens: 0 or negative is not a usable value — warn and fall back."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({
            "default": "claude-opus-4-6",
            "provider": "custom",
            "base_url": "http://localhost:8045/v1",
            "max_tokens": 0,
        })
    assert agent.max_tokens is None
    warning_calls = [c for c in mock_logger.warning.call_args_list
                     if "max_tokens" in str(c) and "Invalid" in str(c)]
    assert len(warning_calls) == 1


def test_missing_max_tokens_leaves_default():
    """No max_tokens key → self.max_tokens stays at the constructor default (None)."""
    with patch("run_agent.logger") as mock_logger:
        agent = _build_agent({
            "default": "claude-opus-4-6",
            "provider": "custom",
            "base_url": "http://localhost:8045/v1",
        })
    assert agent.max_tokens is None
    for c in mock_logger.warning.call_args_list:
        assert "max_tokens" not in str(c) or "Invalid" not in str(c)
