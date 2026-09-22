"""A rate-limited `hermes update` fetch retries with the user's own GitHub credential (#105857).

GitHub throttles anonymous git by IP with HTTP 429, and git only consults a credential helper
after a 401, so a working `gh auth login` / GITHUB_TOKEN was never sent. The update fetch stays
anonymous first and attaches a credential only when GitHub answers 429.
"""

import base64
import subprocess
from unittest.mock import MagicMock

import pytest

import hermes_cli.update_cmd as update_cmd
from hermes_cli import git_credentials
from hermes_cli.update_cmd_git import _git_fetch

GITHUB_URL = "https://github.com/NousResearch/hermes-agent.git"
RATE_LIMITED = ("remote: This request was rate-limited due to too many requests.\n"
                "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n")


def _auth_header(env) -> str | None:
    return next((env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env.get("GIT_CONFIG_COUNT", "0")))
                 if env[f"GIT_CONFIG_VALUE_{i}"].startswith("Authorization:")), None)


@pytest.fixture
def git(monkeypatch):
    """Fake git: `remote get-url` answers ``git.url``; a fetch is answered 429 unless it carries
    an Authorization header. Records every fetch's header (None = anonymous)."""
    git = MagicMock(url=GITHUB_URL, fetches=[], anonymous_ok=False)

    def run(argv, **kw):
        if "get-url" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=git.url + "\n", stderr="")
        header = _auth_header(kw["env"])
        git.fetches.append(header)
        ok = header is not None or git.anonymous_ok
        return subprocess.CompletedProcess(argv, 0 if ok else 128, stdout="", stderr="" if ok else RATE_LIMITED)

    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(git_credentials, "_env_github_token", lambda: None)
    monkeypatch.setattr(git_credentials, "_credential_fill", lambda origin: None)
    return git


def test_rate_limited_fetch_retries_once_with_the_gh_login(git, monkeypatch):
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: "gho_live")

    result = _git_fetch(["git"], "origin", "main")

    assert result.returncode == 0
    expected = "Authorization: basic " + base64.b64encode(b"x-access-token:gho_live").decode()
    assert git.fetches == [None, expected]

    # Happy path: an anonymous success spends exactly one fetch and resolves no credential.
    git.fetches.clear()
    git.anonymous_ok = True
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: pytest.fail("credential resolved"))
    assert _git_fetch(["git"], "origin", "main").returncode == 0
    assert git.fetches == [None]


def test_rate_limited_fetch_without_a_credential_says_how_to_add_one(git, monkeypatch, capsys):
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: None)

    result = _git_fetch(["git"], "origin", "main")

    assert result.returncode != 0 and git.fetches == [None]
    update_cmd._print_fetch_failure(result.stderr)
    out = capsys.readouterr().out
    assert "rate limiting" in out and "gh auth login" in out and "GITHUB_TOKEN" in out


def test_non_github_remote_never_resolves_a_credential_on_429(git, monkeypatch):
    git.url = "https://git.example.test/mirror/hermes-agent.git"
    monkeypatch.setattr(git_credentials, "_credential_fill", lambda origin: pytest.fail("credential resolved"))

    assert _git_fetch(["git"], "origin", "main").returncode != 0
    assert git.fetches == [None]


def test_existing_callers_do_not_retry_a_429(git, monkeypatch):
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: "gho_live")

    result = git_credentials.run_git_with_credential_fallback(
        ["git", "clone", GITHUB_URL], GITHUB_URL, env={}, capture_output=True, text=True)

    assert result.returncode != 0 and git.fetches == [None]
