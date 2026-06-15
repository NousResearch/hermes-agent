"""Tests for hermes_cli.apikey_switch."""

from unittest.mock import MagicMock

import pytest

from hermes_cli.apikey_switch import (
    apply_api_key_switch,
    format_apikey_status,
    mask_api_key,
    parse_apikey_args,
    resolve_provider_key_env,
)


def test_parse_apikey_args_happy_paths():
    args, errors = parse_apikey_args("")
    assert not errors
    assert args.key == ""
    assert not args.save
    assert not args.reload

    args, errors = parse_apikey_args("sk-test123")
    assert not errors
    assert args.key == "sk-test123"

    args, errors = parse_apikey_args("--save sk-test123")
    assert not errors
    assert args.key == "sk-test123"
    assert args.save

    args, errors = parse_apikey_args("--reload")
    assert not errors
    assert args.reload
    assert args.key == ""


def test_resolve_provider_key_env_known_providers():
    assert resolve_provider_key_env("openrouter") == "OPENROUTER_API_KEY"
    assert resolve_provider_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert resolve_provider_key_env("deepseek") == "DEEPSEEK_API_KEY"


def test_resolve_provider_key_env_unknown_provider_fallback():
    assert resolve_provider_key_env("foobar") == "FOOBAR_API_KEY"
    assert resolve_provider_key_env("foo-bar") == "FOO_BAR_API_KEY"


def test_mask_api_key():
    assert mask_api_key("") == "(not set)"
    assert mask_api_key(None) == "(not set)"
    masked = mask_api_key("sk-abcdefghijklmnopqrstuvwxyz")
    assert masked.startswith("sk-a")
    assert "..." in masked


def test_format_apikey_status():
    out = format_apikey_status("openrouter", "claude-sonnet-4", "sk-abc123")
    assert "Provider: openrouter" in out
    assert "Model:    claude-sonnet-4" in out
    assert "Key env:  OPENROUTER_API_KEY" in out
    assert "Key:" in out


def test_apply_api_key_switch_no_provider():
    agent = MagicMock()
    result = apply_api_key_switch(agent, "", "", "sk-key")
    assert not result.success
    assert "No provider" in result.message


def test_apply_api_key_switch_no_key():
    agent = MagicMock()
    result = apply_api_key_switch(agent, "openrouter", "claude", "")
    assert not result.success
    assert "No API key" in result.message


def test_apply_api_key_switch_calls_agent():
    agent = MagicMock()
    result = apply_api_key_switch(agent, "openrouter", "claude", "sk-key")
    assert result.success
    agent.switch_api_key.assert_called_once_with("sk-key", provider="openrouter")


def test_apply_api_key_switch_agent_failure():
    agent = MagicMock()
    agent.switch_api_key.side_effect = RuntimeError("bad key")
    result = apply_api_key_switch(agent, "openrouter", "claude", "sk-key")
    assert not result.success
    assert "bad key" in result.message
