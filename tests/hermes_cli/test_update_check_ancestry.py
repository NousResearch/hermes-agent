"""`hermes update --check` must not read a git failure as proof of being behind.

`banner.py` and `update_cmd.py` are two independent production callers of the
same ancestry rule, and only the banner had direct coverage — the review on
#98167 asked for the second one. These tests drive `_cmd_update_check()`
against a shallow checkout with git and the compare API stubbed.

The distinction being pinned: `merge-base --is-ancestor` exits non-zero both
when the target is genuinely NOT an ancestor and when git could not run at all.
Only the first is evidence about the checkout. Misreporting an ahead checkout
as behind nudges the user into an update that can discard carried work.
"""

import subprocess
from types import SimpleNamespace

import pytest


HEAD_SHA = "1111111111111111111111111111111111111111"
TARGET_SHA = "2222222222222222222222222222222222222222"


def _shallow_git(*, ancestor_result):
    """A subprocess.run stub answering the shallow `--check` git calls.

    `ancestor_result` is an exit code, or an exception to raise for the
    ancestry probe (git could not run).
    """
    seen: dict = {}

    def run(argv, **kwargs):
        if "merge-base" in argv:
            seen["kwargs"] = kwargs
            if isinstance(ancestor_result, BaseException):
                raise ancestor_result
            return SimpleNamespace(returncode=ancestor_result, stdout="", stderr="")
        # Checked before the generic rev-parse arm: shallow detection is itself
        # a rev-parse call, and answering it "false" would route the command
        # down the full-clone path this file is not about.
        if "--is-shallow-repository" in argv:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if "rev-parse" in argv:
            sha = HEAD_SHA if "HEAD" in argv else TARGET_SHA
            return SimpleNamespace(returncode=0, stdout=f"{sha}\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run.seen = seen  # type: ignore[attr-defined]
    return run


@pytest.fixture
def shallow_check(monkeypatch):
    """Run the shallow `--check` path with git and the compare API stubbed."""
    from hermes_cli import update_cmd

    def go(*, ancestor_result, compare_behind):
        stub = _shallow_git(ancestor_result=ancestor_result)
        monkeypatch.setattr(update_cmd.subprocess, "run", stub)
        monkeypatch.setattr(
            "hermes_cli.banner._github_compare_behind",
            lambda head, target: compare_behind,
        )
        try:
            update_cmd._cmd_update_check()
        except SystemExit:
            pass
        return stub

    return go


def test_a_carried_local_commit_reads_as_up_to_date(shallow_check, capsys):
    """Target is an ancestor of HEAD: the checkout is ahead, not behind."""
    shallow_check(ancestor_result=0, compare_behind=None)

    assert "Already up to date" in capsys.readouterr().out


def test_a_genuinely_behind_checkout_still_reports_an_update(shallow_check, capsys):
    """Not an ancestor and the compare API agrees: genuinely behind."""
    shallow_check(ancestor_result=1, compare_behind=3)

    out = capsys.readouterr().out
    assert "Already up to date" not in out
    assert "behind" in out.lower()


@pytest.mark.parametrize(
    "ancestor_result",
    [128, subprocess.TimeoutExpired("git", 5), OSError("git unavailable")],
)
def test_a_failed_ancestry_probe_is_not_reported_as_behind(
    shallow_check, capsys, ancestor_result
):
    """git failing to run says nothing about the checkout.

    With the compare API also unable to answer, the only honest verdict is
    unknown — not "update available".
    """
    shallow_check(
        ancestor_result=ancestor_result,
        compare_behind=None,
    )

    out = capsys.readouterr().out.lower()
    assert "update available" not in out


def test_the_ancestry_probe_runs_under_a_timeout(shallow_check):
    """A wedged git must not stall `--check` indefinitely."""
    stub = shallow_check(ancestor_result=0, compare_behind=None)

    assert stub.seen.get("kwargs", {}).get("timeout"), "ancestry probe has no timeout"
