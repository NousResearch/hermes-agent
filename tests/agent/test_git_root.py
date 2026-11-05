"""Validity and walk policy for the shared git-root predicate (``agent.git_root``).

``agent.coding_context._git_root``, ``agent.prompt_builder._find_git_root`` and
``hermes_cli.web_routers.files._fs_find_git_root`` all delegate the repository test to
``agent.git_root.is_valid_git_root``, so every walk test below asserts the policy through
all three resolvers rather than through one of them.

What is pinned here, and why:

* **R1 — a ``.git`` directory needs ``HEAD`` + ``objects`` + ``refs``.** That is the same
  trio git's own ``is_git_directory`` requires. ``HEAD`` + ``objects`` alone is *not* a
  repository: ``git rev-parse`` says "not a git repository" for it, so accepting it would
  re-open the phantom bug one notch narrower. The test asserts git's verdict alongside
  ours so the rule cannot drift back below git's.
* **R2 — an invalid ``.git`` makes the walk continue upward.** git's discovery skips an
  invalid ``.git`` and keeps climbing. The resolvers are generator/loop based and already
  behave that way; this test is the tripwire that a future "stop at the first ``.git``"
  refactor does not silently lose a real ancestor root (and with it the AGENTS.md chain).
* **R3 — a relative ``gitdir:`` is resolved from the directory containing the ``.git``
  file.** Pinned with a **real submodule** checkout: git's own emitted pointer is the
  contract, and a hand-written ``.git`` file can encode a layout git never writes.
* **R7 — the rejection set**: no-refs layout, garbage ``.git`` file, stale ``gitdir:``
  target, and a bare repo as cwd (deliberately *not* a root: no working tree, so nothing
  to edit and nothing to verify).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent import coding_context as cc
from agent import prompt_builder as pb
from agent.git_root import gitdir_target, is_valid_git_root

_HAS_GIT = shutil.which("git") is not None
requires_git = pytest.mark.skipif(not _HAS_GIT, reason="git not installed")


def _git_env(home: Path) -> dict[str, str]:
    """A hermetic git environment: no user/system config, no prompts, no upward discovery."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        # Stop git's own discovery at the tmpdir, so "not a git repository" means exactly
        # that and not "nothing found before some ancestor repo".
        "GIT_CEILING_DIRECTORIES": str(home),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }


def _git(*args: str, cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=_git_env(home),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path, home: Path) -> None:
    """A real ``git init`` repo with one commit (a submodule parent must have one)."""
    _git("init", "-q", str(root), cwd=home, home=home)
    (root / "README.md").write_text("init\n", encoding="utf-8")
    _git("add", "-A", cwd=root, home=home)
    _git("-c", "commit.gpgsign=false", "commit", "-qm", "init", cwd=root, home=home)


def _web_git_root(start: Path) -> str | None:
    """The files API resolver, imported lazily (it pulls in FastAPI)."""
    file_routes = pytest.importorskip("hermes_cli.web_routers.files")
    return file_routes._fs_find_git_root(start)


# ── R1 / R7: the directory layout ───────────────────────────────────────────


def test_git_dir_needs_refs_as_well_as_head_and_objects(tmp_path):
    """R1: ``HEAD`` + ``objects`` without ``refs/`` is not a repository — git agrees."""
    root = tmp_path / "norefs"
    dot_git = root / ".git"
    (dot_git / "objects").mkdir(parents=True)
    (dot_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    assert is_valid_git_root(root) is False

    if _HAS_GIT:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-dir"],
            env=_git_env(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        # The contract this rule is derived from: git rejects the only-HEAD+objects layout.
        assert probe.returncode != 0, f"git accepted a .git without refs/: {probe.stdout!r}"

    # Completing git's trio flips the verdict — the rule is not simply "reject everything".
    (dot_git / "refs").mkdir()
    assert is_valid_git_root(root) is True


# ── R2: the walk must survive an invalid entry ──────────────────────────────


def test_invalid_git_dir_does_not_stop_the_walk(tmp_path):
    """R2: git skips an invalid ``.git`` and keeps climbing — so must we."""
    outer = tmp_path / "outer"
    outer.mkdir()
    if _HAS_GIT:
        _init_repo(outer, tmp_path)
    else:  # same layout git init writes, without git
        dot = outer / ".git"
        (dot / "objects").mkdir(parents=True)
        (dot / "refs").mkdir()
        (dot / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    inner = outer / "inner"
    (inner / ".git").mkdir(parents=True)  # empty == phantom == invalid
    deep = inner / "deep"
    deep.mkdir()

    assert is_valid_git_root(inner) is False
    # Not None: stopping at the invalid entry would drop the real ancestor root, and with
    # it the root-anchored AGENTS.md / .hermes.md chain.
    assert cc._git_root(deep) == outer.resolve()
    assert pb._find_git_root(deep) == outer.resolve()
    assert _web_git_root(deep) == str(outer)


# ── R7: the rejection set ──────────────────────────────────────────────────


def test_garbage_git_file_is_rejected_and_walk_continues(tmp_path):
    """A ``.git`` *file* with no ``gitdir:`` line is not a repository."""
    outer = tmp_path / "outer"
    outer.mkdir()
    dot = outer / ".git"
    (dot / "objects").mkdir(parents=True)
    (dot / "refs").mkdir()
    (dot / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    inner = outer / "inner"
    inner.mkdir()
    garbage = inner / ".git"
    garbage.write_text("this is not a git pointer\n", encoding="utf-8")
    deep = inner / "deep"
    deep.mkdir()

    assert gitdir_target(garbage) is None
    assert is_valid_git_root(inner) is False
    # Walk continues past the garbage entry, exactly as git's discovery would.
    assert cc._git_root(deep) == outer.resolve()
    assert pb._find_git_root(deep) == outer.resolve()


def test_stale_gitdir_pointer_is_rejected(tmp_path):
    """A ``gitdir:`` pointing at a target that no longer exists fails closed."""
    stale = tmp_path / "stale"
    stale.mkdir()
    pointer = stale / ".git"
    pointer.write_text("gitdir: ../gone/.git/modules/demo\n", encoding="utf-8")

    assert gitdir_target(pointer) is None
    assert is_valid_git_root(stale) is False
    # Never invent a root: the resolver must not answer "stale" just because a .git exists.
    assert cc._git_root(stale) != stale.resolve()
    assert pb._find_git_root(stale) != stale.resolve()


def test_bare_repo_as_cwd_is_not_detected(tmp_path):
    """A bare repo has HEAD+objects+refs but no working tree — deliberately not a root."""
    bare = tmp_path / "bare.git"
    if _HAS_GIT:
        _git("init", "-q", "--bare", str(bare), cwd=tmp_path, home=tmp_path)
    else:
        for sub in ("objects", "refs"):
            (bare / sub).mkdir(parents=True)
        (bare / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    # It really is a repository layout...
    assert (bare / "HEAD").is_file() and (bare / "objects").exists() and (bare / "refs").exists()
    # ...but there is no `.git` entry, because there is no working tree to edit or verify.
    assert is_valid_git_root(bare) is False
    assert cc._git_root(bare) != bare.resolve()
    assert pb._find_git_root(bare) != bare.resolve()
    assert _web_git_root(bare) != str(bare)


# ── R3: the relative gitdir base, proved with a real submodule ──────────────


@requires_git
def test_submodule_relative_gitdir_is_accepted(tmp_path):
    """R3: a submodule's relative ``gitdir:`` resolves from the file's own directory."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _init_repo(upstream, tmp_path)

    superproject = tmp_path / "super"
    superproject.mkdir()
    _init_repo(superproject, tmp_path)
    _git(
        "-c", "protocol.file.allow=always", "submodule", "add", str(upstream), "vendor/dep",
        cwd=superproject, home=tmp_path,
    )

    sub = superproject / "vendor" / "dep"
    pointer = sub / ".git"
    assert pointer.is_file(), "git should write a .git FILE for a submodule working tree"

    text = pointer.read_text(encoding="utf-8").strip()
    assert text.startswith("gitdir:"), text
    raw = text.split(":", 1)[1].strip()
    assert not Path(raw).is_absolute(), (
        "git emitted an absolute gitdir, so this test would not pin the relative-base rule: "
        + raw
    )

    # The contract: our resolution must equal git's own resolution of the same pointer.
    git_says = subprocess.run(
        ["git", "-C", str(sub), "rev-parse", "--absolute-git-dir"],
        env=_git_env(tmp_path), capture_output=True, text=True, check=True,
    ).stdout.strip()
    resolved = gitdir_target(pointer)
    assert resolved is not None
    assert str(resolved) == git_says

    assert is_valid_git_root(sub) is True
    # A submodule working tree is its own project root, not a pass-through to the parent.
    assert cc._git_root(sub) == sub.resolve()
    assert pb._find_git_root(sub) == sub.resolve()
