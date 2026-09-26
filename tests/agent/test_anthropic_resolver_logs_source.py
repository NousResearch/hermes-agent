"""The Anthropic token resolver reports which source won.

An explicit ``ANTHROPIC_API_KEY`` deliberately outranks a discovered Claude Code
subscription credential. Acting on that precedence used to leave no log line at any level,
so an inherited or forgotten key silently moved a host from subscription auth to metered API
billing with nothing to read afterwards (#97085).
"""

import logging

import pytest

from agent import anthropic_credentials as ac


@pytest.fixture
def subscription_on_disk(monkeypatch):
    monkeypatch.setattr(ac, "read_claude_code_credentials", lambda: {
        "accessToken": "sk-ant-oat01-subscription", "refreshToken": "rt-1", "expiresAt": 0})
    monkeypatch.setattr(ac, "_resolve_anthropic_pool_token", lambda **kw: None)
    monkeypatch.setattr(ac, "_available_anthropic_token", lambda token, model: token or None)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(ac, "_shadowed_subscription_warned", False, raising=False)


def test_api_key_over_available_subscription_warns_once_without_token_material(
        monkeypatch, caplog, subscription_on_disk):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-explicit-key-material")
    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == "sk-ant-api03-explicit-key-material"
        assert ac.resolve_anthropic_token() == "sk-ant-api03-explicit-key-material"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "ANTHROPIC_API_KEY" in warnings[0].getMessage()
    for record in caplog.records:
        assert "explicit-key-material" not in record.getMessage()
        assert "sk-ant-oat01" not in record.getMessage()


def test_api_key_without_subscription_stays_quiet_at_warning_level(monkeypatch, caplog, subscription_on_disk):
    monkeypatch.setattr(ac, "read_claude_code_credentials", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-only-key")
    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == "sk-ant-api03-only-key"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # The stage is still recoverable from a debug run.
    assert any("ANTHROPIC_API_KEY" in r.getMessage() for r in caplog.records)
