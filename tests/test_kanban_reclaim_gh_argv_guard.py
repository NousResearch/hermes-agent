"""Guard: every flag the reclaim's ``gh`` probe passes must exist in ``gh``.

Why this file exists — measured, not hypothetical. The squash-merge arm shipped
with ``gh -C <path> pr list ...``. ``-C`` is a *git* flag; ``gh`` has no such
flag and exits 1 with ``unknown shorthand flag: 'C' in -C``. Every unit test for
that arm monkeypatches ``_gh_pr_json`` wholesale, so the mocks stayed green while
the arm was dead on every real host — exactly the bug class the arm was written
to kill.

The guard DERIVES what it checks: it captures the real argv the production code
builds (by intercepting :func:`subprocess.run`, never by restating the command),
then asks the installed ``gh`` binary for that subcommand's own ``--help`` and
requires each captured flag to appear there. Adding a new flag to the probe is
covered automatically; nothing has to be remembered.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_reclaim as kbr

pytestmark = pytest.mark.skipif(
    shutil.which("gh") is None, reason="gh binary not installed on this host"
)


def _captured_gh_argv(monkeypatch, tmp_path: Path) -> list[str]:
    """The exact argv ``_gh_pr_json`` hands to the OS. Derived, never restated."""
    seen: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *args, **kwargs):
        if argv and argv[0] == "gh":
            seen.append(list(argv))
            # Do not actually hit the network from a guard.
            return subprocess.CompletedProcess(list(argv), returncode=1, stdout="", stderr="")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    kbr._gh_pr_json(tmp_path, "some/branch")
    # Restore the real subprocess.run before anyone shells out for real help
    # text; leaving the spy installed makes _gh_help() read empty output and
    # the guard fails for the wrong reason.
    monkeypatch.undo()
    assert seen, "_gh_pr_json did not invoke gh at all"
    return seen[0]


def _gh_help(argv: list[str]) -> str:
    """``--help`` text for the gh subcommand in ``argv`` (its leading non-flag words)."""
    sub: list[str] = []
    for token in argv[1:]:
        if token.startswith("-"):
            break
        sub.append(token)
    assert sub, (
        f"no gh subcommand found in {argv}: the argv starts with a flag "
        f"({argv[1]!r}) before any subcommand. gh takes no global flags there "
        "— this is the git-style `-C` shape that does not exist in gh."
    )
    res = subprocess.run(
        ["gh", *sub, "--help"], capture_output=True, text=True, timeout=60, check=False
    )
    return res.stdout + res.stderr


def test_every_flag_the_gh_probe_passes_exists_in_gh(monkeypatch, tmp_path: Path):
    argv = _captured_gh_argv(monkeypatch, tmp_path)
    help_text = _gh_help(argv)
    flags = [t for t in argv[1:] if t.startswith("-")]
    assert flags, f"expected the probe to pass flags; argv={argv}"
    unknown = [f for f in flags if f not in help_text]
    assert not unknown, (
        f"gh does not accept {unknown} for `{' '.join(argv[:3])}`.\n"
        f"argv was: {argv}\n"
        "This is the -C regression: git flags are not gh flags."
    )


def test_the_probe_locates_the_repo_by_cwd_not_by_a_flag(monkeypatch, tmp_path: Path):
    """``gh`` resolves the repo from the working directory; prove we set it."""
    seen: dict[str, object] = {}
    real_run = subprocess.run

    def spy(argv, *args, **kwargs):
        if argv and argv[0] == "gh":
            seen["argv"] = list(argv)
            seen["cwd"] = kwargs.get("cwd")
            return subprocess.CompletedProcess(list(argv), returncode=1, stdout="", stderr="")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    kbr._gh_pr_json(tmp_path, "some/branch")
    assert seen.get("cwd") == str(tmp_path), (
        "the gh probe must run with cwd set to the worktree, otherwise it "
        f"resolves the wrong repo (or none). saw cwd={seen.get('cwd')!r}"
    )


def test_gh_rejects_the_old_dash_C_form():
    """Positive control: the guard's premise is real, not folklore."""
    res = subprocess.run(
        ["gh", "-C", ".", "pr", "list", "--json", "number"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    combined = res.stdout + res.stderr
    assert res.returncode != 0 and "unknown shorthand flag" in combined, combined
