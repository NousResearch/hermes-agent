"""Smoke test for the claude_cli package skeleton and error hierarchy."""

import pytest

from agent.claude_cli import errors


def test_error_hierarchy():
    """All claude_cli exceptions inherit from ClaudeCliError."""
    assert issubclass(errors.ClaudeCliUnavailable, errors.ClaudeCliError)
    assert issubclass(errors.ClaudeCliVersionTooOld, errors.ClaudeCliError)
    assert issubclass(errors.ClaudeCliAuthMissing, errors.ClaudeCliError)
    assert issubclass(errors.ClaudeCliIncompatible, errors.ClaudeCliError)
    assert issubclass(errors.HermesDirectAnthropicEgressDetected, errors.ClaudeCliError)
    assert issubclass(errors.ProtocolError, errors.ClaudeCliError)
    assert issubclass(errors.PromptTooLarge, errors.ClaudeCliError)


def test_error_message_attribute():
    """All errors accept a str message and preserve it."""
    e = errors.ClaudeCliVersionTooOld("found 2.0.0, need 2.1.143")
    assert str(e) == "found 2.0.0, need 2.1.143"
