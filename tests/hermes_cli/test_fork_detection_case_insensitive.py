"""Regression tests for #60240: case-insensitive fork detection + --yes
skip for the upstream prompt.

Two related bugs in the ``hermes update`` flow:

1. ``_is_fork()`` compares ``origin_url`` to ``OFFICIAL_REPO_URLS`` with a
   plain ``==`` after stripping ``.git`` and ``/``. GitHub owner/repo
   slugs are case-insensitive, so a clone URL like
   ``https://github.com/nousresearch/hermes-agent`` (lowercase) is
   incorrectly classified as a fork.

2. The "Add official repo as 'upstream' remote?" prompt in
   ``_handle_fork_warning`` is wired to bare ``input()`` and never
   consults ``assume_yes`` (the ``--yes`` flag). Unattended updates
   (CI, cron, --yes) hang forever waiting for a response that never
   comes.

Both fixes are small and self-contained.
"""
from __future__ import annotations

import pytest


# Two URLs that GitHub treats as identical (case-insensitive owner +
# case-insensitive repo). Both must be classified as NOT-A-FORK
# because the underlying repository IS the upstream.
LOERCASE_HTTPS = "https://github.com/nousresearch/hermes-agent"
MIXEDCASE_HTTPS = "https://github.com/NousResearch/hermes-agent"
MIXEDCASE_GIT = "https://github.com/NousResearch/hermes-agent.git"
LOWERCASE_SSH = "git@github.com:nousresearch/hermes-agent.git"
MIXEDCASE_SSH = "git@github.com:NousResearch/hermes-agent.git"


class TestForkDetectionCaseInsensitive:
    """The case-insensitive fix for #60240."""

    def test_lowercase_https_is_not_a_fork(self):
        """A user who types the URL by hand with lowercase owner
        (``nousresearch``) — what GitHub auto-redirects to — must NOT
        be classified as a fork.

        GitHub treats owner/repo slugs case-insensitively, so the
        lowercase URL is functionally identical to the canonical
        ``NousResearch/hermes-agent`` upstream.
        """
        from hermes_cli.main import _is_fork
        assert _is_fork(LOERCASE_HTTPS) is False, (
            f"lowercase origin URL {LOERCASE_HTTPS!r} classified as fork; "
            f"GitHub treats owner/repo case-insensitively. See #60240."
        )

    def test_lowercase_https_dot_git_is_not_a_fork(self):
        from hermes_cli.main import _is_fork
        assert _is_fork(LOERCASE_HTTPS + ".git") is False

    def test_lowercase_ssh_is_not_a_fork(self):
        from hermes_cli.main import _is_fork
        assert _is_fork(LOWERCASE_SSH) is False, (
            f"lowercase SSH origin {LOWERCASE_SSH!r} classified as fork. "
            f"See #60240."
        )

    def test_mixedcase_https_is_still_not_a_fork(self):
        """The canonical URL form must continue to be recognized
        (regression guard against the fix going too far).
        """
        from hermes_cli.main import _is_fork
        assert _is_fork(MIXEDCASE_HTTPS) is False
        assert _is_fork(MIXEDCASE_HTTPS + ".git") is False
        assert _is_fork(MIXEDCASE_SSH) is False

    def test_genuine_fork_is_still_a_fork(self):
        """The fix must not regress: a clearly different origin
        (e.g. a personal fork) is still a fork.
        """
        from hermes_cli.main import _is_fork
        assert _is_fork("https://github.com/some-user/hermes-agent") is True
        assert _is_fork("git@github.com:hermes-forker/hermes-agent.git") is True
        assert _is_fork("https://gitlab.com/NousResearch/hermes-agent") is True

    def test_none_url_is_not_a_fork(self):
        """Missing origin (no remote configured) is not a fork — the
        caller decides what to do with the missing URL.
        """
        from hermes_cli.main import _is_fork
        assert _is_fork(None) is False
        assert _is_fork("") is False