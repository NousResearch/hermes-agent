"""GitSpawn / GHSA-7x36-8jrh-v4pw regression suite.

A repository delivered as files (zip, sync folder, USB) can carry a
``.git/config`` that names a command in an execution-sink git setting —
``core.fsmonitor``, ``core.hooksPath`` hooks, or an attribute-scoped
``[diff "x"] command=/textconv=`` driver. Hermes gathers workspace context by
running git against the session directory automatically, before any prompt,
approval, or trust gate, so an unhardened probe would execute that command on
the host as the user.

These tests build a real malicious repo and assert that every automatic
context-gathering git path Hermes runs neutralizes every sink. They use a real
``git`` and skip if it is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import (
    NO_DRIVER_DIFF_FLAGS,
    harden_git_argv,
    noninteractive_git_env,
)

_HAS_GIT = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(not _HAS_GIT, reason="git not installed")


# ---------------------------------------------------------------------------
# 1. harden_git_argv unit contract
# ---------------------------------------------------------------------------


class TestHardenGitArgv:
    def test_diff_gets_flags_after_subcommand(self):
        assert harden_git_argv(["diff", "HEAD"]) == [
            "diff", *NO_DRIVER_DIFF_FLAGS, "HEAD",
        ]

    def test_show_log_blame_are_hardened(self):
        for sub in ("show", "log", "blame"):
            out = harden_git_argv([sub, "x"])
            assert out[0] == sub
            assert out[1:3] == list(NO_DRIVER_DIFF_FLAGS)

    def test_status_is_not_touched(self):
        # status rejects --no-ext-diff (`unknown option`), so it must pass through.
        assert harden_git_argv(["status", "--porcelain=2", "--branch"]) == [
            "status", "--porcelain=2", "--branch",
        ]

    def test_worktree_and_other_subcommands_untouched(self):
        assert harden_git_argv(["worktree", "add", "x"]) == ["worktree", "add", "x"]
        assert harden_git_argv(["rev-parse", "HEAD"]) == ["rev-parse", "HEAD"]

    def test_global_options_are_skipped_when_finding_subcommand(self):
        out = harden_git_argv(["-C", "/repo", "diff", "HEAD"])
        assert out == ["-C", "/repo", "diff", *NO_DRIVER_DIFF_FLAGS, "HEAD"]

    def test_dash_c_value_is_not_mistaken_for_subcommand(self):
        # ``-C diff`` is a path; the real subcommand is status → no flags.
        assert harden_git_argv(["-C", "diff", "status"]) == ["-C", "diff", "status"]
        # ``-c diff=x`` is a config pair; the real subcommand is status.
        assert harden_git_argv(["-c", "diff=x", "status"]) == ["-c", "diff=x", "status"]

    def test_config_pair_before_diff_still_hardens(self):
        out = harden_git_argv(["-c", "core.quotePath=false", "diff", "--numstat"])
        assert out == [
            "-c", "core.quotePath=false", "diff", *NO_DRIVER_DIFF_FLAGS, "--numstat",
        ]


# ---------------------------------------------------------------------------
# 2. Real-git E2E: every automatic path neutralizes every sink
# ---------------------------------------------------------------------------


def _make_malicious_repo(tmp: Path) -> tuple[Path, Path]:
    """Build a repo whose .git/config arms fsmonitor, a checkout hook, and an
    attribute-scoped external-diff + textconv driver. Returns (repo, marker_stem):
    a fired sink leaves ``<marker_stem>.<sink>`` on disk."""
    repo = tmp / "poc"
    clean = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=clean)
    (repo / "README").write_text("hi\n")
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "-C", str(repo), *ident, "add", "."], check=True, env=clean)
    subprocess.run(["git", "-C", str(repo), *ident, "commit", "-qm", "init"], check=True, env=clean)

    marker = tmp / "MARKER"
    hooks = repo / "evil-hooks"
    hooks.mkdir()
    hook = hooks / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}.hook\n")
    hook.chmod(0o755)
    with (repo / ".git" / "config").open("a") as f:
        f.write(f'[core]\n\tfsmonitor = "touch {marker}.fsmonitor"\n\thooksPath = {hooks}\n')
        f.write(f'[diff "evil"]\n\tcommand = "touch {marker}.extdiff"\n')
        f.write(f'\ttextconv = "sh -c \'touch {marker}.textconv; cat\'"\n')
    (repo / ".gitattributes").write_text("* diff=evil\n")
    (repo / "README").write_text("changed\n")  # dirty working tree so diffs run
    return repo, marker


def _fired(marker: Path) -> list[str]:
    out = []
    for sink in ("fsmonitor", "hook", "extdiff", "textconv"):
        p = Path(f"{marker}.{sink}")
        if p.exists():
            out.append(sink)
            p.unlink()
    return out


@pytest.fixture()
def malicious_repo(tmp_path):
    repo, marker = _make_malicious_repo(tmp_path)
    yield repo, marker


def test_baseline_unhardened_git_fires_sinks(malicious_repo):
    """Sanity: without hardening the payload actually fires — proves the repo
    is armed and the test can detect a regression."""
    repo, marker = malicious_repo
    subprocess.run(["git", "-C", str(repo), "diff", "HEAD"], capture_output=True)
    fired = _fired(marker)
    assert "fsmonitor" in fired and "extdiff" in fired, fired


def test_coding_workspace_snapshot_is_safe(malicious_repo):
    import agent.coding_context as cc
    repo, marker = malicious_repo
    cc.build_coding_workspace_block(cwd=repo)
    assert _fired(marker) == []


def test_gateway_git_probe_is_safe(malicious_repo):
    from tui_gateway import git_probe
    repo, marker = malicious_repo
    git_probe.branch(str(repo))
    git_probe.run_git(str(repo), "status", "--porcelain")
    assert _fired(marker) == []


def test_working_diff_is_safe(malicious_repo):
    from tools.working_diff import collect_working_diff
    repo, marker = malicious_repo
    collect_working_diff(str(repo), "working")
    assert _fired(marker) == []


def test_goals_fingerprint_is_safe(malicious_repo):
    from hermes_cli.goals import workspace_fingerprint
    repo, marker = malicious_repo
    workspace_fingerprint(str(repo))
    assert _fired(marker) == []


def test_web_git_diff_is_safe(malicious_repo):
    from hermes_cli import web_git
    repo, marker = malicious_repo
    web_git._git(str(repo), ["status", "--porcelain=v2", "-z"])
    web_git._git_out(str(repo), ["diff", "HEAD"])
    assert _fired(marker) == []


def test_context_reference_diff_is_safe(malicious_repo):
    from agent import context_references as cr
    repo, marker = malicious_repo
    ref = type("R", (), {"raw": "@diff"})()
    cr._expand_git_reference(ref, repo, ["diff", "HEAD"], "git diff")
    assert _fired(marker) == []


def test_subagent_worktree_add_is_safe(malicious_repo, tmp_path):
    from tools import subagent_worktree as sw
    repo, marker = malicious_repo
    sw._run_git(["worktree", "add", str(tmp_path / "wt1"), "-b", "safe1"], str(repo))
    assert _fired(marker) == []


def test_noninteractive_env_pins_fsmonitor_and_hooks():
    env = noninteractive_git_env({})
    values = {
        env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(env["GIT_CONFIG_COUNT"]))
    }
    assert values["core.fsmonitor"] == "false"
    assert values["core.hooksPath"] == os.devnull
