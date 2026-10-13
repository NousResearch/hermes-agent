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


def test_env_token_replaced_by_refreshable_login_attributes_source_to_the_file(
        monkeypatch, caplog, resolver_env):
    """Source attribution follows the credential actually returned. A static env OAuth token
    that ``_prefer_refreshable_claude_code_token`` swaps for the refreshable file must be
    logged as the file, not the env."""
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-static-env-not-a-secret")
    monkeypatch.setattr(ac, "read_claude_code_credentials",
                        lambda: {"accessToken": SUBSCRIPTION_ACCESS, "refreshToken": "rt-1"})
    monkeypatch.setattr(ac, "_resolve_claude_code_token_from_credentials",
                        lambda creds: SUBSCRIPTION_ACCESS)

    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == SUBSCRIPTION_ACCESS
    sources = [r.getMessage() for r in caplog.records if "Anthropic token source" in r.getMessage()]
    assert len(sources) == 1, sources
    assert "credential file" in sources[0]
    assert not sources[0].startswith("Anthropic token source: ANTHROPIC_TOKEN")


def test_shadow_warning_honours_adopt_external_logins_opt_out(monkeypatch, caplog, resolver_env):
    """``auth.adopt_external_logins: false`` means the Claude Code login is unavailable on
    purpose; telling the user to unset the key would remove the only credential allowed."""
    _write_subscription_login(resolver_env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    from agent import credential_sources
    monkeypatch.setattr(credential_sources, "adopt_external_logins_enabled", lambda: False)

    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        assert ac.resolve_anthropic_token() == API_KEY
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_shadow_warning_fires_once_under_concurrent_resolves(monkeypatch, caplog, resolver_env):
    import threading
    _write_subscription_login(resolver_env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    n = 8
    barrier = threading.Barrier(n)

    def _resolve():
        barrier.wait()
        ac.resolve_anthropic_token()

    with caplog.at_level(logging.DEBUG, logger=ac.__name__):
        threads = [threading.Thread(target=_resolve) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
