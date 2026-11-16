"""An expired borrowed Claude Code login whose refresh cannot reach the token endpoint is not a
missing credential.

A cron job pinned to ``anthropic`` that fires right after a wake (DNS still down) used to be
blocked as ``blocked_config`` with "No Anthropic credentials found. Run 'hermes auth add
anthropic'", and lost its slot until the next period. The login was fine: only the refresh POST
failed. The verdict must name the refresh failure, the preflight must not block on it, and the real
path must classify it as model-unreachable so the bounded re-run ladder retries it.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from agent import anthropic_credentials as ac
from cron.scheduler_preflight import _is_transient_provider_resolve_error, _preflight_check_provider_key
from cron.unreachable_retry import is_model_unreachable_failure
from hermes_cli.auth import AuthError
from hermes_cli.runtime_provider import _anthropic_token_or_raise

JOB = {"id": "ea2e3655cee4", "name": "pulse", "provider": "anthropic", "model": "claude-sonnet-4-6"}


@pytest.fixture
def homes(tmp_path, monkeypatch):
    hermes_home, claude_dir = tmp_path / "hermes", tmp_path / "claude"
    hermes_home.mkdir()
    claude_dir.mkdir()
    (hermes_home / ".env").write_text("")
    (hermes_home / "config.yaml").write_text("model:\n  provider: anthropic\n  default: claude-sonnet-4-6\n")
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    ac.take_refresh_failure()
    return hermes_home, claude_dir


def _write_expired_login(claude_dir):
    (claude_dir / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "sk-ant-oat01-expired", "refreshToken": "sk-ant-ort01-offline",
        "expiresAt": int(time.time() * 1000) - 3_600_000, "scopes": ["user:inference"]}}))


@pytest.fixture
def dns_down(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_offline_refresh_is_reported_as_refresh_failure_not_missing_login(homes, dns_down):
    _write_expired_login(homes[1])

    with pytest.raises(AuthError) as info:
        _anthropic_token_or_raise(model=JOB["model"])

    exc = info.value
    assert exc.code == "refresh_failed"
    assert "No Anthropic credentials found" not in str(exc)
    assert isinstance(exc.__cause__, urllib.error.URLError)
    assert _is_transient_provider_resolve_error(exc)
    # How the scheduler's real path sees it: _resolve_job_runtime wraps the resolve error.
    try:
        raise RuntimeError(str(exc)) from exc
    except RuntimeError as wrapped:
        assert is_model_unreachable_failure(wrapped, agent=None)


def test_preflight_does_not_block_an_offline_refresh(homes, dns_down):
    _write_expired_login(homes[1])

    assert _preflight_check_provider_key(JOB, {"cron": {}}) is None


def test_preflight_still_blocks_a_genuinely_missing_login(homes, dns_down):
    reason = _preflight_check_provider_key(JOB, {"cron": {}})

    assert reason and reason.startswith("provider credential missing: No Anthropic credentials found")


def test_stale_refresh_failure_from_an_earlier_resolve_is_not_reused(homes):
    ac._REFRESH_FAILURE.exc = urllib.error.URLError("stale")

    with pytest.raises(AuthError) as info:
        _anthropic_token_or_raise(model=JOB["model"])

    assert "No Anthropic credentials found" in str(info.value)
    assert info.value.__cause__ is None


def test_refresh_timeout_is_retried_as_model_unreachable(homes, monkeypatch):
    _write_expired_login(homes[1])

    def fake_urlopen(req, timeout=None):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(AuthError) as info:
        _anthropic_token_or_raise(model=JOB["model"])
    assert info.value.code == "refresh_failed"
    assert is_model_unreachable_failure(info.value, agent=None)
    assert _preflight_check_provider_key(JOB, {"cron": {}}) is None


def test_token_endpoint_5xx_is_not_a_missing_login(homes, monkeypatch):
    import io

    _write_expired_login(homes[1])

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(AuthError) as info:
        _anthropic_token_or_raise(model=JOB["model"])
    assert info.value.code == "refresh_failed"
    # A server error is not the narrow never-reached-the-network class the re-run ladder serves.
    assert not is_model_unreachable_failure(info.value, agent=None)
    assert _preflight_check_provider_key(JOB, {"cron": {}}) is None


def test_model_cooldown_still_wins_over_an_incidental_refresh_failure(homes, dns_down, monkeypatch):
    from agent import credential_pool

    _write_expired_login(homes[1])
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-env")

    class _CooledPool:
        def token_is_blocked(self, token, model=None):
            return model is not None

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: _CooledPool())

    with pytest.raises(AuthError) as info:
        _anthropic_token_or_raise(model=JOB["model"])
    assert info.value.code == "model_rate_limited"
    assert ac.take_refresh_failure() is None
