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
import time
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import (
    FILTER_DISCOVERY_FAILED,
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


_CLEAN_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                  "GIT_CONFIG_NOSYSTEM": "1"}


def _make_malicious_repo(tmp: Path) -> tuple[Path, Path]:
    """Build a repo whose .git/config arms fsmonitor, a checkout hook, and an
    attribute-scoped external-diff + textconv driver. Returns (repo, marker_stem):
    a fired sink leaves ``<marker_stem>.<sink>`` on disk."""
    repo = tmp / "poc"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=_CLEAN_GIT_ENV)
    (repo / "README").write_text("hi\n")
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "-C", str(repo), *ident, "add", "."], check=True, env=_CLEAN_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), *ident, "commit", "-qm", "init"], check=True, env=_CLEAN_GIT_ENV)

    marker = tmp / "MARKER"
    hooks = repo / "evil-hooks"
    hooks.mkdir()
    marker_shell = marker.as_posix()
    # post-checkout fires on `worktree add`; reference-transaction on every ref update, `branch -D` included.
    for name in ("post-checkout", "reference-transaction"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\ntouch '{marker_shell}.hook'\n")
        hook.chmod(0o755)
    # Let git encode config values; raw Windows backslashes are escapes.
    settings = {
        "core.fsmonitor": f"touch '{marker_shell}.fsmonitor'",
        "core.hooksPath": hooks.as_posix(),
        "diff.evil.command": f"touch '{marker_shell}.extdiff'",
        "diff.evil.textconv": f"touch '{marker_shell}.textconv'; cat",
    }
    for key, value in settings.items():
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True, env=_CLEAN_GIT_ENV)
    (repo / ".gitattributes").write_text("* diff=evil\n")
    (repo / "README").write_text("changed\n")  # dirty working tree so diffs run
    return repo, marker


def _fired(marker: Path) -> list[str]:
    out = []
    for sink in ("fsmonitor", "hook", "extdiff", "textconv", "ssh"):
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


def test_index_reading_probes_and_kanban_gc_git_are_safe(malicious_repo, tmp_path, monkeypatch):
    """``status`` / ``ls-files`` / ``worktree add`` read the index, which runs ``core.fsmonitor``;
    ``worktree add`` also runs the repository's hooks, and ``branch -D`` its reference-transaction
    hook. Recovery hint, completion probe, kanban worktree, worktree-gc ``status``, the reclaimers'
    dirty probe (kanban teardown), and the three unattended branch deletions: the reclaim sweep,
    the orphaned-branch pass and the cleanup after a failed ``worktree add``. The reclaim sweep's
    ``ls-remote`` and the shallow-repo ``fetch --unshallow`` must not run a repo ``core.sshCommand``."""
    from hermes_cli import kanban_db_workspace as kw
    from hermes_cli import worktree_gc, worktree_ops
    from tools.async_delegation_recovery_hints import git_state_hint
    from tui_gateway import server
    repo, marker = malicious_repo
    assert git_state_hint(str(repo)) is not None
    assert "README" in list(server._git_repo_files(str(repo)))
    kw._ensure_git_worktree(repo, tmp_path / "wt2", "safe2")
    assert (tmp_path / "wt2" / "README").exists()
    assert worktree_gc._git(["status", "--porcelain"], cwd=str(repo)).returncode == 0
    dirty = worktree_ops._worktree_is_dirty(str(tmp_path / "wt2"))
    assert _fired(marker) == []
    assert dirty is False  # the probe ran: a skipped one reads as dirty

    subprocess.run(["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "branch", "pr-1"], check=True,
                   env=_CLEAN_GIT_ENV)
    kw._ensure_git_worktree(repo, tmp_path / "wt3", "safe3")
    worktree_ops._reap_prune_verdicts(str(repo), [(tmp_path / "wt2", 0.0, False, "reap", None)], 0.0)
    worktree_ops._prune_orphaned_branches(str(repo))
    worktree_ops._cleanup_failed_worktree_add(str(repo), tmp_path / "wt3", "safe3")
    assert _fired(marker) == []
    branches = subprocess.run(["git", "-C", str(repo), "branch", "--format=%(refname:short)"],
                              capture_output=True, text=True, env=_CLEAN_GIT_ENV).stdout.split()
    assert not {"safe2", "pr-1", "safe3"} & set(branches)  # the deletions ran

    for key, value in {"remote.origin.url": "ssh://git@example.invalid/x.git",
                       "core.sshCommand": f"touch '{marker.as_posix()}.ssh'; false"}.items():
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True, env=_CLEAN_GIT_ENV)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True, env=_CLEAN_GIT_ENV).stdout
    (repo / ".git" / "shallow").write_text(head)  # _repo_is_shallow now reads true
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)  # an env value would mask core.sshCommand
    worktree_ops._fetch_remote_branch_heads(str(repo), timeout=20)
    worktree_ops._deepen_shallow_repo(str(repo), timeout=20)
    assert _fired(marker) == []


def _make_filter_repo(tmp: Path, attrs: str, config, marker: str) -> Path:
    """Committed repo whose ``.gitattributes`` is *attrs*; ``config(repo, marker)`` is appended to ``.git/config``."""
    repo = tmp / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=_CLEAN_GIT_ENV)
    (repo / "README").write_text("hi\n")
    (repo / ".gitattributes").write_text(attrs)
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "-C", str(repo), *ident, "add", "."], check=True, env=_CLEAN_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), *ident, "commit", "-qm", "init"], check=True, env=_CLEAN_GIT_ENV)
    with open(repo / ".git" / "config", "a") as fh:
        fh.write(config(repo, marker))
    return repo


def _evil_filter(marker: str, name: str = "evil") -> str:
    return (f'[filter "{name}"]\n\tsmudge = touch \'{marker}.smudge\'; cat\n'
            f'\tclean = touch \'{marker}.clean\'; cat\n\trequired = true\n')


def _evil_include(condition):
    def config(repo: Path, marker: str) -> str:
        (repo / ".git" / "evil.inc").write_text(_evil_filter(marker))
        return f'[includeIf "{condition(repo)}"]\n\tpath = evil.inc\n'
    return config


@pytest.mark.parametrize("attrs, config, refused", [
    pytest.param("README filter=evil\n", lambda r, m: _evil_filter(m), False, id="plain"),
    pytest.param("README filter=Evil\n",
                 lambda r, m: '[filter "evil"]\n\tsmudge = cat\n\tclean = cat\n' + _evil_filter(m, "Evil"),
                 False, id="case_collision"),
    pytest.param("README filter=evil\n", _evil_include(lambda r: "onbranch:safe"), True, id="onbranch_include"),
    pytest.param("README filter=evil\n",
                 _evil_include(lambda r: f"gitdir:{(r / '.git' / 'worktrees').as_posix()}/"), True,
                 id="gitdir_include"),
    pytest.param("README filter=evil\n",
                 lambda r, m: "".join(f'[filter "f{i}"]\n\tclean = cat\n' for i in range(300)), True,
                 id="filter_flood"),
    # Malformed config: `git config` exits 3, which is neither "found" (0) nor "none" (1).
    pytest.param("README filter=evil\n", lambda r, m: _evil_filter(m) + '[filter "evil"\n', True,
                 id="broken_config"),
])
def test_repo_named_filters_never_run_from_kanban_gc_or_hints(tmp_path, attrs, config, refused):
    """A filter driver is named by ``.gitattributes``, so the fixed env pins cannot reach it:
    ``worktree add`` runs its smudge command and ``status`` its clean command. Subsection names are
    case-sensitive, so ``[filter "Evil"]`` must be neutralized next to a benign ``[filter "evil"]``.
    Discovery that cannot be trusted refuses the git call: any ``includeIf`` (``onbranch:`` matches the
    new branch, ``gitdir:`` the ``.git/worktrees/<name>`` dir of ``worktree add``) loads filters
    discovery never saw, a huge filter inventory would overflow the argv/env limit (E2BIG), and a
    config git cannot parse yields no trustworthy inventory at all."""
    from hermes_cli import kanban_db_workspace as kw
    from hermes_cli import worktree_gc
    from tools.async_delegation_recovery_hints import git_state_hint
    marker = (tmp_path / "FILTER").as_posix()
    repo = _make_filter_repo(tmp_path, attrs, config, marker)

    if refused:
        res = kw._git(repo, "worktree", "add", "-b", "safe", str(tmp_path / "wt"), "HEAD", timeout=30)
        assert (res.returncode, res.stderr) == (1, FILTER_DISCOVERY_FAILED)
        assert sorted(p.name for p in tmp_path.glob("FILTER.*")) == []
        return

    kw._ensure_git_worktree(repo, tmp_path / "wt", "safe")
    assert (tmp_path / "wt" / "README").read_text() == "hi\n"
    (repo / "README").write_text("hi\n")  # same size, new mtime: status must re-hash it
    os.utime(repo / "README", (time.time() + 60, time.time() + 60))
    assert worktree_gc._git(["status", "--porcelain"], cwd=str(repo)).returncode == 0
    assert git_state_hint(str(repo)) is not None
    # The automatic session-start snapshot (status) and a delegated subagent's worktree (checkout).
    from agent.coding_context import build_coding_workspace_block
    from tools.subagent_worktree import create_subagent_worktree
    assert "- Status:" in build_coding_workspace_block(repo)
    sub = create_subagent_worktree(str(repo), "filters")
    assert sub is not None and (Path(sub["path"]) / "README").read_text() == "hi\n"
    # The subagent's automatic finalization and kanban teardown re-hash a touched file (status);
    # hermes -w checks out a worktree of its own.
    from hermes_cli import worktree_ops
    from tools.subagent_worktree import finalize_subagent_worktree
    for tree in (Path(sub["path"]), tmp_path / "wt"):
        os.utime(tree / "README", (time.time() + 120, time.time() + 120))
    assert "inspection_failed" not in finalize_subagent_worktree(sub, prune=False)
    assert worktree_ops._worktree_is_dirty(str(tmp_path / "wt")) is False
    assert worktree_ops._worktree_add(str(repo), tmp_path / "wt-w", "hermes/filters", "HEAD", "HEAD")
    assert sorted(p.name for p in tmp_path.glob("FILTER.*")) == []


def test_noninteractive_env_pins_fsmonitor_and_hooks():
    env = noninteractive_git_env({})
    values = {
        env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(env["GIT_CONFIG_COUNT"]))
    }
    assert values["core.fsmonitor"] == "false"
    assert values["core.hooksPath"] == os.devnull
