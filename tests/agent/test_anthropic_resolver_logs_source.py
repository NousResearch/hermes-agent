"""The Anthropic token resolver reports which source won.

An explicit ``ANTHROPIC_API_KEY`` deliberately outranks a discovered Claude Code
subscription credential. Acting on that precedence used to leave no log line at any level,
so an inherited or forgotten key silently moved a host from subscription auth to metered API
billing with nothing to read afterwards (#97085).

The API-key branch must not *read* the borrowed login (``read_claude_code_credentials`` is
asserted never-called by ``test_anthropic_adapter.py``), so the shadowing check is a file
existence probe and these tests exercise it with a real path under a temp ``HOME``.
"""

import json
import logging

import pytest

from agent import anthropic_credentials as ac

API_KEY = "api-key-value-not-a-secret"
SUBSCRIPTION_ACCESS = "subscription-access-material-not-a-secret"


@pytest.fixture
def resolver_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(ac, "_resolve_anthropic_pool_token", lambda **kw: None)
    monkeypatch.setattr(ac, "_available_anthropic_token", lambda token, model: token or None)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(ac, "_shadowed_subscription_warned", False, raising=False)
    return tmp_path


def _write_subscription_login(home):
    cred = home / ".claude" / ".credentials.json"
    cred.parent.mkdir(parents=True)
    cred.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": SUBSCRIPTION_ACCESS, "refreshToken": "rt-1", "expiresAt": 0}}))
    return cred


def test_api_key_over_subscription_login_warns_once_without_reading_or_leaking(
        monkeypatch, caplog, resolver_env):
    _write_subscription_login(resolver_env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)

    def _must_not_read():
        raise AssertionError("API-key path must not read the borrowed login")
    monkeypatch.setattr(ac, "read_claude_code_credentials", _must_not_read)

    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == API_KEY
        assert ac.resolve_anthropic_token() == API_KEY
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "ANTHROPIC_API_KEY" in warnings[0].getMessage()
    for record in caplog.records:
        assert API_KEY not in record.getMessage()
        assert SUBSCRIPTION_ACCESS not in record.getMessage()


def test_api_key_without_subscription_login_stays_quiet_at_warning_level(
        monkeypatch, caplog, resolver_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == API_KEY
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # The stage is still recoverable from a debug run.
    assert any("ANTHROPIC_API_KEY" in r.getMessage() for r in caplog.records)
