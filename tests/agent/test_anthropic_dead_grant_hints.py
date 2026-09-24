"""Hermes credential hints and the Anthropic token-endpoint error shape (#113023).

A dead Hermes login must be repaired with ``hermes auth add <provider>``; hints that send the user to
an external CLI's login command do not touch Hermes' own credentials. The token endpoint's
``invalid_grant`` body is surfaced as a structured, classifiable error so the pool can quarantine
instead of benching the dead grant as transient.
"""
from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent import anthropic_credentials as ac


def test_dead_grant_is_classified_and_not_replayed_at_other_endpoints(monkeypatch):
    calls: list = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"invalid_grant","error_description":"revoked"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ac, "_OAUTH_TOKEN_URLS", ["https://a.example/oauth/token", "https://b.example/oauth/token"])

    with pytest.raises(ac.AnthropicOAuthError) as info:
        ac.refresh_anthropic_oauth_pure("sk-ant-ort-dead")

    assert ac.is_terminal_anthropic_refresh_error(info.value)
    assert info.value.code == "invalid_grant" and "revoked" in str(info.value)
    assert calls == ["https://a.example/oauth/token"]  # a dead grant is not replayed at the fallback endpoint
    assert not ac.is_terminal_anthropic_refresh_error(TimeoutError("timed out"))




def test_claude_code_refresher_reports_dead_grant_once_per_process(monkeypatch, caplog):
    """Later attempts with the same dead refresh token neither replay it at the endpoint nor re-warn; a rotated
    (re-login) token is tried again."""
    monkeypatch.setattr(ac, "_DEAD_REFRESH_TOKEN_FINGERPRINTS", set())
    monkeypatch.setattr(ac, "read_claude_code_credentials", lambda: {"accessToken": "old", "refreshToken": "rt-dead", "expiresAt": 1})
    posts = []

    def dead(refresh_token, *, use_json=False):
        posts.append(refresh_token)
        raise ac.AnthropicOAuthError(400, "invalid_grant", "", what="refresh")

    monkeypatch.setattr(ac, "refresh_anthropic_oauth_pure", dead)
    creds = {"accessToken": "old", "refreshToken": "rt-dead"}
    with caplog.at_level(logging.DEBUG, logger=ac.logger.name):
        for _ in range(3):
            assert ac._refresh_oauth_token(creds) is None
    assert posts == ["rt-dead"]
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1
    assert not any("claude setup-token" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(ac, "read_claude_code_credentials", lambda: {"accessToken": "old", "refreshToken": "rt-new", "expiresAt": 1})
    assert ac._refresh_oauth_token({"accessToken": "old", "refreshToken": "rt-new"}) is None
    assert posts == ["rt-dead", "rt-new"]


def test_claude_code_credentials_path_honours_claude_config_dir(monkeypatch, tmp_path):
    """The documented opt-out: CLAUDE_CONFIG_DIR relocates the borrowed file exactly as the Claude CLI does."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    assert ac.claude_code_credentials_path() == tmp_path / "cc" / ".credentials.json"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "  ")
    assert ac.claude_code_credentials_path() == Path.home() / ".claude" / ".credentials.json"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert ac.claude_code_credentials_path() == Path.home() / ".claude" / ".credentials.json"




