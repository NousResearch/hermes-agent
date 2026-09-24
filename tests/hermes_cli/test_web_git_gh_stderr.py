"""Behavioral tests for the gh wrappers' failure reporting (#87731).

`_gh` used to collapse every non-zero exit to `(False, "")` and
`review_create_pr` raised a generic "is gh installed and authenticated?"
message — which lied whenever gh was fine and the failure was real
("no commits between main and feature", a missing upstream, a refused
push). These tests pin the contract that stderr survives the wrapper and
reaches the surfaced error, with the generic text reserved for the case
gh itself reported nothing.
"""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

import hermes_cli.web_git as web_git


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr)


def _make_gh(returncode: int, stdout: str = "", stderr: str = ""):
    """Patch gh as installed (shutil.which) and running with the given result."""
    return patch.object(web_git, "_run", return_value=_proc(returncode, stdout, stderr))


class TestGhRetainsStderr:
    def test_success_returns_ok_with_stdout_and_stderr(self, tmp_path):
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(0, "ok\n", ""):
            ok, out, err = web_git._gh(str(tmp_path), ["auth", "status"])

        assert ok is True
        assert out == "ok\n"
        assert err == ""

    def test_nonzero_exit_keeps_exit_status_and_stderr(self, tmp_path):
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(
            1, "", "no commits between main and feature"
        ):
            ok, out, err = web_git._gh(str(tmp_path), ["pr", "create", "--fill"])

        assert ok is False
        assert out == ""
        assert err == "no commits between main and feature"

    def test_missing_gh_binary_is_not_ok_without_raising(self, tmp_path):
        with patch.object(web_git.shutil, "which", return_value=None):
            ok, out, err = web_git._gh(str(tmp_path), ["auth", "status"])

        assert (ok, out, err) == (False, "", "")


class TestReviewCreatePrSurfacesGhStderr:
    def test_failure_message_carries_gh_stderr_marker(self, tmp_path):
        # A unique marker: the assertion fails if the wrapper regresses to the
        # generic message or drops stderr anywhere between exec and the raise.
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(
            1, "", "gh: no commits between main and bb/fix-wave2g"
        ):
            with pytest.raises(RuntimeError, match="no commits between main and bb/fix-wave2g"):
                web_git.review_create_pr(str(tmp_path))

    def test_failure_message_still_names_gh_when_stderr_is_empty(self, tmp_path):
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(1, "", "   "):
            with pytest.raises(RuntimeError, match="is gh installed and authenticated"):
                web_git.review_create_pr(str(tmp_path))

    def test_success_path_returns_the_pr_url(self, tmp_path):
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(
            0, "https://github.com/org/repo/pull/1234\n", ""
        ):
            result = web_git.review_create_pr(str(tmp_path))

        assert result == {"url": "https://github.com/org/repo/pull/1234"}

    def test_oversized_stderr_is_bounded_to_its_tail(self, tmp_path):
        marker = "TRAILING-MARKER-reason"
        stderr = "x" * 5_000 + "\n" + marker
        with patch.object(web_git.shutil, "which", return_value="/usr/bin/gh"), _make_gh(1, "", stderr):
            with pytest.raises(RuntimeError) as excinfo:
                web_git.review_create_pr(str(tmp_path))

        message = str(excinfo.value)
        assert marker in message
        # The 5_000-char noise prefix does not ride along: the surfaced detail
        # is bounded to the configured tail, not the whole stderr.
        assert len(message) <= len("gh pr create failed: ") + web_git._GH_ERR_TAIL_CHARS
        assert "x" * web_git._GH_ERR_TAIL_CHARS not in message
