"""Tests for GitHub integration service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.code.github_integration import (
    GitHubAPIClient,
    GitHubAPIError,
    GitHubAppConfig,
    GitHubIntegrationService,
    redact_github_secrets,
)


def test_status_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("HERMES_GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("HERMES_GITHUB_DEV_PAT", raising=False)
    monkeypatch.delenv("HERMES_GITHUB_ALLOW_DEV_PAT", raising=False)
    status = GitHubIntegrationService(db_path=tmp_path / "state.db").status()
    assert status["mode"] == "unconfigured"
    assert status["configured"] is False


def test_app_config_detection_without_secret_leak(monkeypatch, tmp_path):
    key_file = tmp_path / "github-app.pem"
    key_file.write_text("-----BEGIN PRIVATE KEY-----\nFAKE\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_GITHUB_APP_ID", "123456")
    monkeypatch.setenv("HERMES_GITHUB_APP_PRIVATE_KEY_PATH", str(key_file))
    monkeypatch.setenv("HERMES_GITHUB_WEBHOOK_SECRET", "super-secret")
    status = GitHubIntegrationService(db_path=tmp_path / "state.db").status()
    assert status["mode"] == "github_app"
    assert status["app_id_configured"] is True
    assert status["private_key_configured"] is True
    assert status["webhook_secret_configured"] is True
    assert "super-secret" not in str(status)


def test_pat_fallback_requires_env_gate(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("HERMES_GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.setenv("HERMES_GITHUB_DEV_PAT", "ghp_abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("HERMES_GITHUB_ALLOW_DEV_PAT", "0")
    status = GitHubIntegrationService(db_path=tmp_path / "state.db").status()
    assert status["mode"] == "unconfigured"
    monkeypatch.setenv("HERMES_GITHUB_ALLOW_DEV_PAT", "1")
    status = GitHubIntegrationService(db_path=tmp_path / "state.db").status()
    assert status["mode"] == "pat_dev"


def test_installation_token_cache(monkeypatch, tmp_path):
    service = GitHubIntegrationService(db_path=tmp_path / "state.db")
    service._token_cache.clear()
    calls = {"count": 0}

    def _fake_request(_installation_id: int):
        calls["count"] += 1
        return {
            "token": "token-123",
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        }

    monkeypatch.setattr(service, "_request_installation_token", _fake_request)
    monkeypatch.setattr(
        service,
        "config",
        lambda: GitHubAppConfig(
            app_id="1",
            private_key_path="/tmp/key.pem",
            webhook_secret_configured=False,
            dev_pat_configured=False,
            allow_dev_pat=False,
        ),
    )
    first = service.get_installation_token(42)
    second = service.get_installation_token(42)
    assert first == "token-123"
    assert second == "token-123"
    assert calls["count"] == 1


def test_api_error_normalization_and_rate_limit():
    class _Response:
        status_code = 403
        headers = {
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "123",
            "x-ratelimit-resource": "core",
        }

        def json(self):
            return {"message": "Bad token ghp_abcdefghijklmnopqrstuvwxyz"}

    class _HTTP:
        def request(self, *_args, **_kwargs):
            return _Response()

    client = GitHubAPIClient(lambda: "ghp_secret", http_client=_HTTP())
    with pytest.raises(GitHubAPIError) as exc:
        client.request("GET", "/user")
    assert "ghp_" not in exc.value.message
    assert exc.value.status_code == 403
    assert exc.value.rate_limit["remaining"] == "0"


def test_redaction_masks_sensitive_values():
    redacted = redact_github_secrets("Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz")
    assert "ghp_" not in redacted
    assert "[REDACTED]" in redacted
