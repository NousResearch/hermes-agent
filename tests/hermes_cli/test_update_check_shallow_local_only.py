"""RED tests for #98214 — shallow `update --check` misreports `None` as behind.

Simulates the exact issue scenario: shallow clone, local-only HEAD commits on
top of origin/main, GitHub compare API cannot resolve the local SHA and returns
None. The shallow branch must NOT assert "Update available".
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _shallow_side_effect(ancestor_rc=0):
    """subprocess.run side effect: shallow repo, divergent SHAs, API -> None.

    ancestor_rc: exit code of `git merge-base --is-ancestor <target> HEAD`
      0  -> origin/main is contained in HEAD (ahead-or-equal, not behind)
      1  -> genuinely not an ancestor
      128 -> inconclusive (outside shallow boundary)
    """

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--is-shallow-repository" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="true\n", stderr="")
        if "fetch" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="local-patches\n", stderr="")
        if "rev-parse" in joined and "--verify" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "rev-parse" in joined and "HEAD" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="8" * 40 + "\n", stderr="")
        if joined.rstrip().endswith("rev-parse origin/main"):
            return subprocess.CompletedProcess(cmd, 0, stdout="4" * 40 + "\n", stderr="")
        if "merge-base" in joined and "--is-ancestor" in joined:
            return subprocess.CompletedProcess(cmd, ancestor_rc, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


ARGS = SimpleNamespace(check=True, branch=None)


def _run_check():
    from hermes_cli.main import cmd_update

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch("hermes_cli.banner._github_compare_behind", return_value=None):
        cmd_update(ARGS)


def test_none_with_ancestor_reports_up_to_date(capsys):
    with patch("subprocess.run", side_effect=_shallow_side_effect(ancestor_rc=0)):
        _run_check()
    out = capsys.readouterr().out
    assert "Already up to date." in out, out
    assert "Update available" not in out, out


def test_none_without_ancestor_does_not_claim_behind(capsys):
    with patch("subprocess.run", side_effect=_shallow_side_effect(ancestor_rc=1)):
        _run_check()
    out = capsys.readouterr().out
    # Must not assert an update exists — status is unknown, not "behind".
    assert "Update available" not in out, out
    assert "unknown" in out, out


def test_none_inconclusive_ancestor_does_not_claim_behind(capsys):
    with patch("subprocess.run", side_effect=_shallow_side_effect(ancestor_rc=128)):
        _run_check()
    out = capsys.readouterr().out
    assert "Update available" not in out, out
    assert "unknown" in out, out
