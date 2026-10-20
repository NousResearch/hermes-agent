"""The gh CLI token rung must survive wrappers that print to stdout (#138444).

`gh` reached through a PATH wrapper (mise/asdf shims, corporate hooks) can emit
its own status lines to stdout ahead of the real ``gh auth token`` answer. The
capture then holds wrapper noise plus the token, and the embedded newline makes
the ``Authorization`` header invalid — killing ``hermes update`` even though the
anonymous rung of the ladder would have succeeded.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import github_api, source_check


def _probe(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gh", "auth", "token"], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.fixture(autouse=True)
def _isolated_credential_ladder(monkeypatch):
    """No env tokens, and a fresh per-process gh cache for every test."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_api, "_gh_cli_cache", None)
    monkeypatch.setattr(github_api, "_gh_cli_probed", False)


def test_wrapper_noise_ahead_of_the_token_is_dropped():
    token = "gho_" + "x" * 36
    capture = "mise ~/.config/mise/config.toml tools: gh@2.102.0\n" + token + "\n"
    with patch("hermes_cli.github_api.subprocess.run", return_value=_probe(capture)):
        assert github_api._gh_cli_token() == token


def test_clean_token_capture_is_kept_as_is():
    token = "ghp_" + "y" * 36
    with patch("hermes_cli.github_api.subprocess.run", return_value=_probe(token + "\n")):
        assert github_api._gh_cli_token() == token


def test_capture_without_a_github_shaped_last_line_is_ignored():
    capture = "mise ~/.config/mise/config.toml tools: gh@2.102.0\n"
    with patch("hermes_cli.github_api.subprocess.run", return_value=_probe(capture)):
        assert github_api._gh_cli_token() is None


def test_fine_grained_token_shape_is_accepted():
    token = "github_pat_" + "z" * 22 + "_" + "w" * 59
    capture = "wrapper chatter\n" + token
    with patch("hermes_cli.github_api.subprocess.run", return_value=_probe(capture)):
        assert github_api._gh_cli_token() == token


def test_failed_gh_probe_yields_no_token():
    with patch("hermes_cli.github_api.subprocess.run", return_value=_probe("", returncode=1)):
        assert github_api._gh_cli_token() is None


def test_gh_is_probed_at_most_once_per_process():
    token = "ghp_" + "y" * 36
    runner = patch("hermes_cli.github_api.subprocess.run", return_value=_probe(token))
    with runner as mock_run:
        assert github_api.github_token() == token
        assert github_api.github_token() == token
    assert mock_run.call_count == 1


def test_an_invalid_header_token_falls_back_to_anonymous(monkeypatch):
    calls = []

    def _fake_request_with(url, accept, token):
        calls.append(token)
        if token is not None:
            raise ValueError("Invalid header value b'Bearer noisy\\ntoken'")
        return "{}"

    monkeypatch.setattr(source_check, "_request_with", _fake_request_with)
    monkeypatch.setattr(github_api, "github_token", lambda: "noisy\ntoken")
    assert source_check._request("https://api.github.com/fixture") == "{}"
    assert calls == ["noisy\ntoken", None]


def test_the_fallback_never_quotes_the_malformed_token(monkeypatch, caplog):
    """The ValueError message embeds the header; it must not reach any log line."""

    def _fake_request_with(url, accept, token):
        if token is not None:
            raise ValueError("Invalid header value b'Bearer noisy\\nghp_secret'")
        return "{}"

    monkeypatch.setattr(source_check, "_request_with", _fake_request_with)
    monkeypatch.setattr(github_api, "github_token", lambda: "noisy\nghp_secret")
    with caplog.at_level("DEBUG", logger="hermes_cli.source_check"):
        assert source_check._request("https://api.github.com/fixture") == "{}"
    assert "ghp_secret" not in caplog.text


def test_a_value_error_without_a_token_still_propagates(monkeypatch):
    def _fake_request_with(url, accept, token):
        raise ValueError("unrelated decode problem")

    monkeypatch.setattr(source_check, "_request_with", _fake_request_with)
    monkeypatch.setattr(github_api, "github_token", lambda: None)
    with pytest.raises(ValueError, match="unrelated decode problem"):
        source_check._request("https://api.github.com/fixture")
