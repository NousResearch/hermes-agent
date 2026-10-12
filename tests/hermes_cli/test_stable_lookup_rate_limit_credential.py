"""The stable lookup's tag check on origin retries a GitHub 429 with the user's credential (#105857).

Stable is the default channel of an official checkout, so `hermes update` runs
`git ls-remote --tags origin` before any fetch. It runs in the isolated git env, which reads no
credential helper: on a throttled IP the update stopped there however the user was logged in.
"""

import base64
import subprocess
from unittest.mock import MagicMock

import pytest

from hermes_cli import git_credentials, source_releases

pytestmark = pytest.mark.real_release_channels

SHA = "818c13be1dc4fd28987e1e881a9408224afd4535"
TAG = "v0.21.6"
RATE_LIMITED = ("remote: This request was rate-limited due to too many requests.\n"
                "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/': "
                "The requested URL returned error: 429\n")


def _auth_header(env) -> str | None:
    return next((env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env.get("GIT_CONFIG_COUNT", "0")))
                 if env[f"GIT_CONFIG_VALUE_{i}"].startswith("Authorization:")), None)


@pytest.fixture
def git(monkeypatch):
    """Fake git: origin is ``git.url``; `ls-remote` is answered 429 unless it carries an
    Authorization header. Records every ls-remote's header (None = anonymous)."""
    git = MagicMock(url="https://github.com/NousResearch/hermes-agent.git", lookups=[])

    def run(argv, **kw):
        if "remote.origin.url" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=git.url + "\n", stderr="")
        header = _auth_header(kw["env"])
        git.lookups.append(header)
        if header is None:
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr=RATE_LIMITED)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{SHA}\trefs/tags/{TAG}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(source_releases, "_json", lambda url: (
        {"sha": SHA} if "/commits/" in url else {"tag_name": TAG, "draft": False, "prerelease": False}))
    monkeypatch.setattr(git_credentials, "_env_github_token", lambda: None)
    monkeypatch.setattr(git_credentials, "_credential_fill", lambda origin: None)
    return git


def test_rate_limited_tag_lookup_retries_once_with_the_gh_login(git, monkeypatch):
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: "gho_live")

    target = source_releases._resolve_stable("NousResearch/hermes-agent", ["git"], "/repo")

    assert target.commit == SHA
    expected = "Authorization: basic " + base64.b64encode(b"x-access-token:gho_live").decode()
    assert git.lookups == [None, expected]


def test_rate_limited_tag_lookup_without_a_credential_says_how_to_add_one(git, monkeypatch):
    monkeypatch.setattr(git_credentials, "_gh_cli_token", lambda: None)

    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent", ["git"], "/repo")

    message = str(caught.value)
    assert git.lookups == [None]
    assert "rate limiting" in message and "gh auth login" in message and "GITHUB_TOKEN" in message
    assert "No published stable release" not in message


def test_non_github_origin_never_resolves_a_credential_on_429(git, monkeypatch):
    git.url = "https://git.example.test/mirror/hermes-agent.git"
    monkeypatch.setattr(git_credentials, "_credential_fill", lambda origin: pytest.fail("credential resolved"))

    with pytest.raises(ValueError):
        source_releases._resolve_stable("NousResearch/hermes-agent", ["git"], "/repo")
    assert git.lookups == [None]
