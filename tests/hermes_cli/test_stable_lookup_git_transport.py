"""The stable lookup's git leg: a repository this machine cannot reach is not "no release" (#137244).

The GitHub lane already names its cause (a 403 quota, a dead api.github.com). The
git leg had no such branch: a ``ls-remote`` that timed out or was refused fell
through to ``No published stable release could be verified``, a sentence about the
upstream project for a cause that lives on the local network.
"""
from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.real_release_channels

from hermes_cli import source_releases

SHA = "818c13be1dc4fd28987e1e881a9408224afd4535"
GIT = ["git"]


@pytest.fixture
def published_stable(monkeypatch):
    """The GitHub lane answers v0.21.6, so only the git leg of ``_resolve_release`` is in play."""
    monkeypatch.setattr(source_releases, "_published_fallback",
                        lambda channel, base: {"tag_name": "v0.21.6", "draft": False, "prerelease": False})
    monkeypatch.setattr(source_releases, "_json", lambda url: {"sha": SHA})
    return SHA


def test_a_git_transport_timeout_is_reported_as_an_unreachable_repository(published_stable, monkeypatch):
    def timing_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=[*GIT, "ls-remote"], timeout=60)

    monkeypatch.setattr(source_releases.subprocess, "run", timing_out)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent", GIT, "/tmp")
    message = str(caught.value)
    assert "No published stable release could be verified" not in message
    assert "could not reach the release repository" in message.lower()
    assert "timed out after 60 seconds" in message


def test_a_refused_git_connection_carries_git_s_own_error(published_stable, monkeypatch):
    stderr = (
        "fatal: unable to access 'https://github.com/NousResearch/hermes-agent/': "
        "Failed to connect to github.com port 443 after 21 ms: Couldn't connect to server\n"
    )

    def refused(*args, **kwargs):
        raise subprocess.CalledProcessError(128, [*GIT, "ls-remote", "--tags", "origin"], stderr=stderr)

    monkeypatch.setattr(source_releases.subprocess, "run", refused)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent", GIT, "/tmp")
    message = str(caught.value)
    assert "No published stable release could be verified" not in message
    # The line the user otherwise had to find in ~/.hermes/logs.
    assert "Failed to connect to github.com port 443" in message


def test_a_wrapped_windows_connect_failure_reports_the_cause_not_the_preamble(published_stable, monkeypatch):
    """The same refused connection as Windows spells it: git wraps the cause onto its own line.

    Reported on the issue after the first version of this file (#137244): on Windows the
    sentence came out of a *connect* failure rather than the 60 s timeout, and the line that
    names the cause is the second one -- ``fatal: unable to access …`` names the URL, not why.
    """
    stderr = (
        "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/':\n"
        "  Failed to connect to github.com:443 after 21105 ms: Could not connect to server\n"
    )

    def refused(*args, **kwargs):
        raise subprocess.CalledProcessError(128, [*GIT, "ls-remote", "--tags", "origin"], stderr=stderr)

    monkeypatch.setattr(source_releases.subprocess, "run", refused)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent", GIT, "/tmp")
    message = str(caught.value)
    assert "No published stable release could be verified" not in message
    assert "Could not connect to server" in message
    # The preamble names the URL, so it is the wrong line to carry: the cause is the last one.
    assert "unable to access" not in message


def test_a_git_that_cannot_run_is_not_reported_as_a_missing_release(published_stable, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(source_releases.subprocess, "run", missing)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent", GIT, "/tmp")
    assert "No published stable release could be verified" not in str(caught.value)


def test_a_genuinely_unpublished_release_still_says_nothing_was_published(monkeypatch):
    """The transport branch must not swallow the case the sentence was written for."""

    def unpublished(channel, base):
        raise ValueError("No published stable release")

    monkeypatch.setattr(source_releases, "_published_fallback", unpublished)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent")
    assert "No published stable release could be verified" in str(caught.value)
