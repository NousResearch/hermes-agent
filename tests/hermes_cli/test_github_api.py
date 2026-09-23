"""Credentialed GitHub API requests used by passive, best-effort callers."""

import io
import logging
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from hermes_cli import github_api


class _Response:
    def __init__(self, body=b'{}'):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _reset_token_cache():
    github_api._reset_github_token_cache()
    yield
    github_api._reset_github_token_cache()


def test_github_token_prefers_github_token_over_gh_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")
    monkeypatch.setenv("GH_TOKEN", "gh-token")

    assert github_api.github_token() == "github-token"


def test_github_token_uses_gh_token_when_github_token_is_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gh-token")

    assert github_api.github_token() == "gh-token"


def test_github_token_caches_gh_auth_token_without_exported_tokens(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_api.shutil, "which", lambda _: "/usr/bin/gh")
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="cli-token\n"))
    monkeypatch.setattr(github_api.subprocess, "run", run)

    assert github_api.github_token() == "cli-token"
    assert github_api.github_token() == "cli-token"
    assert run.call_count == 1
    assert run.call_args.kwargs["env"].get("GITHUB_TOKEN") is None
    assert run.call_args.kwargs["env"].get("GH_TOKEN") is None


def test_request_adds_authentication_header(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    opened = []

    def open_request(request, timeout):
        opened.append(request)
        return _Response(b'{"ok": true}')

    with patch("urllib.request.urlopen", side_effect=open_request):
        assert github_api.get_json("https://api.github.com/repos/nousresearch/hermes-agent") == {"ok": True}

    assert opened[0].get_header("Authorization") == "Bearer secret-token"


def test_request_retries_anonymously_after_401(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "stale-token")
    opened = []

    def open_request(request, timeout):
        opened.append(request)
        if len(opened) == 1:
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())
        return _Response(b'{"ok": true}')

    with patch("urllib.request.urlopen", side_effect=open_request):
        assert github_api.get_json("https://api.github.com/repos/nousresearch/hermes-agent") == {"ok": True}

    assert opened[0].get_header("Authorization") == "Bearer stale-token"
    assert opened[1].get_header("Authorization") is None


def test_request_rejects_non_github_host_without_sending_a_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")

    with patch("urllib.request.urlopen") as open_request:
        with pytest.raises(ValueError, match="https://api.github.com"):
            github_api.get_json("http://api.github.com/resource")

    open_request.assert_not_called()


def test_request_logs_actionable_rate_limit_failure(monkeypatch, caplog):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    error = HTTPError("https://api.github.com/rate_limit", 403, "Forbidden", {"X-RateLimit-Remaining": "0"}, io.BytesIO())

    caplog.set_level(logging.DEBUG, logger="hermes_cli.github_api")
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(HTTPError):
            github_api.get_json("https://api.github.com/rate_limit")

    assert "rate limit" in caplog.text.lower()
