"""Per-thread workspace forks: one ``git worktree`` per repository for each chat thread.

Opt-in (``gateway.workspaces.per_thread: worktree``). A chat bound to a directory (``/workspace`` or
``gateway.workspaces.dir``) gets its own branch ``hermes/<thread>`` in every repository there, so two
threads editing the same project never share a checkout:

* the directory IS a repository: the fork is ``<repo>/.worktrees/<thread>``;
* the directory HOLDS repositories (direct children with ``.git``): each gets
  ``<child>/.worktrees/<thread>``, and the fork root ``<dir>/.worktrees/<thread>`` links them by name
  and carries an ``AGENTS.md`` "Repositories in this workspace" index.

Trees live where the stale-worktree sweep looks (``<repo>/.worktrees``, the repo recorded in
``worktree_registry``), are locked by the serving gateway's pid, and branch from the freshly fetched
remote base (``worktree_ops._resolve_worktree_base``; the fetch is skipped while ``FETCH_HEAD`` is
younger than ``fetch_fresh_seconds``). A fork never waits on a clone: a configured repository that is
not cloned yet is listed in the index with the partial-clone command the agent runs when it needs it.
Worktree operations on one repository are serialized, so two threads forking it at once cannot race.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PerThread = Literal["off", "worktree"]
_PER_THREAD: dict[str, PerThread] = {"off": "off", "worktree": "worktree"}

_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


@dataclass(frozen=True)
class RepoSpec:
    """A repository the workspace expects: ``path`` relative to the bound directory, ``url`` to clone."""
    path: str
    url: str


@dataclass(frozen=True)
class WorkspaceSettings:
    """``gateway.workspaces`` from config.yaml."""
    per_thread: PerThread = "off"
    fetch_fresh_seconds: float = 60.0
    dir: str = ""
    repos: tuple[RepoSpec, ...] = ()

    @classmethod
    def from_config(cls, user_config: Any) -> WorkspaceSettings:
        section = ((user_config or {}).get("gateway") or {}).get("workspaces") if isinstance(user_config, dict) else None
        if not isinstance(section, dict):
            return cls()
        per_thread = _PER_THREAD.get(str(section.get("per_thread") or "off").strip().lower(), "off")
        try:
            fresh = max(0.0, float(section.get("fetch_fresh_seconds", 60)))
        except (TypeError, ValueError):
            fresh = 60.0
        raw_dir = str(section.get("dir") or "").strip()
        repos = tuple(
            RepoSpec(str(r["path"]).strip().strip("/"), str(r["url"]).strip())
            for r in section.get("repos") or () if isinstance(r, dict) and r.get("path") and r.get("url"))
        return cls(per_thread, fresh, os.path.expanduser(raw_dir) if raw_dir else "", repos)


@dataclass(frozen=True)
class Fork:
    root: str
    trees: tuple[tuple[str, str], ...]  # (repo root, worktree path)


def fork_name(session_key: str) -> str:
    """Deterministic, filesystem- and ref-safe name for a chat thread's fork."""
    route = re.sub(r"^agent:[^:]*:", "", session_key)  # the profile namespace is the same for every thread
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", route).strip("-._")[-40:].strip("-._")
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:8]
    return f"thread-{slug}-{digest}" if slug else f"thread-{digest}"


def _repo_lock(repo: Path) -> threading.Lock:
    with _repo_locks_guard:
        return _repo_locks.setdefault(os.path.realpath(repo), threading.Lock())


def _local_repos(directory: Path) -> list[Path]:
    if (directory / ".git").exists():
        return [directory]
    return sorted(c for c in directory.iterdir() if c.is_dir() and (c / ".git").exists())


def _lock_tree(repo: Path, tree: Path) -> None:
    """Hold the tree for this gateway: a lock left by a gateway that has since restarted is replaced."""
    from hermes_cli.worktree_ops import _git_quiet
    _git_quiet(["worktree", "unlock", str(tree)], repo, log="fork unlock")
    _git_quiet(["worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(tree)], repo, log="fork lock")


def fresh_base(repo: Path, fresh_seconds: float) -> tuple[str, str]:
    """``(ref, label)`` of the remote default branch, fetched unless ``FETCH_HEAD`` is younger than
    *fresh_seconds*; a failed fetch keeps the cached ref (forking must not wait on a slow remote)."""
    from hermes_cli._subprocess_compat import noninteractive_git_env
    from hermes_cli.update_cmd_check import tracking_refspec
    from hermes_cli.worktree_ops import _git, _git_out
    head = _git_out(["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"], repo) or next(
        (ref for ref in ("origin/main", "origin/master")
         if _git(["rev-parse", "--verify", "--quiet", ref], repo).returncode == 0), None)
    if not head:
        return "HEAD", "local HEAD (no origin default branch)"
    fetch_head = Path(_git_out(["rev-parse", "--absolute-git-dir"], repo) or repo / ".git") / "FETCH_HEAD"
    if fetch_head.exists() and (age := time.time() - fetch_head.stat().st_mtime) < fresh_seconds:
        return head, f"{head} (fetched {int(age)}s ago)"
    try:
        fetched = _git(["fetch", "origin", tracking_refspec("origin", head.split("/", 1)[1])], repo,
                       timeout=30, stdin=subprocess.DEVNULL, env=noninteractive_git_env()).returncode == 0
    except subprocess.TimeoutExpired:
        fetched = False
    return head, f"{head} (fetched)" if fetched else f"{head} (cached; fetch failed)"


def _add_tree(repo: Path, tree: Path, branch: str, fresh_seconds: float) -> str | None:
    """Create *tree* on *branch* -> the base it started from, or None when git refused."""
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    from hermes_cli.worktree_ops import _ensure_worktrees_gitignored, _git, _worktree_add
    _ensure_worktrees_gitignored(str(repo))
    tree.parent.mkdir(parents=True, exist_ok=True)
    if _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo).returncode == 0:
        # The thread's branch outlived its tree (the sweep removed it): check the branch out again.
        env = noninteractive_repo_git_env(repo)
        if env is None:
            return None
        result = _git(["worktree", "add", str(tree), branch], repo, timeout=120, stdin=subprocess.DEVNULL, env=env)
        return branch if result.returncode == 0 else None
    base_ref, label = fresh_base(repo, fresh_seconds)
    added = _worktree_add(str(repo), tree, branch, base_ref, label)
    return added[1] if added else None


def ensure_tree(repo: Path, name: str, fresh_seconds: float) -> tuple[Path, str] | None:
    """The thread's worktree of *repo* (created when missing) -> ``(path, state)``, or None."""
    from hermes_cli.worktree_ops import _copy_worktree_includes
    from hermes_cli.worktree_registry import record_worktree_repo
    tree, branch = repo / ".worktrees" / name, f"hermes/{name}"
    with _repo_lock(repo):
        if tree.is_dir():
            state = f"worktree on `{branch}`"
        else:
            label = _add_tree(repo, tree, branch, fresh_seconds)
            if label is None:
                logger.warning("Could not create the thread worktree %s in %s", tree, repo)
                return None
            _copy_worktree_includes(str(repo), tree)
            state = f"worktree on `{branch}` from {label}"
        _lock_tree(repo, tree)
    record_worktree_repo(str(repo))
    return tree, state


def relock(fork: Fork) -> None:
    """Re-take the trees' locks for this gateway process (after a restart the old pid is dead)."""
    for repo, tree in fork.trees:
        if Path(tree).is_dir():
            with _repo_lock(Path(repo)):
                _lock_tree(Path(repo), Path(tree))


def _index(directory: Path, root: Path, name: str, rows: list[str]) -> str:
    return (
        "# Repositories in this workspace\n\n"
        f"This is the thread fork `{name}` of `{directory}`: every repository below has its own git worktree "
        f"on branch `hermes/{name}`, shared with no other chat. Work and commit there.\n\n"
        + "\n".join(rows) + "\n\n"
        "A repository marked *not cloned* has no local copy yet. Clone it only when the task needs it, "
        "with the command shown (a partial clone: history now, file contents on demand), then add its "
        f"worktree into this fork:\n`git -C {directory}/<path> worktree add {root}/<path> -b hermes/{name}`\n")


def _link(root: Path, label: str, tree: Path) -> None:
    link = root / label
    if not link.exists():
        try:
            os.symlink(tree, link, target_is_directory=True)
        except OSError as exc:  # no symlink privilege (Windows): the index still names the real path
            logger.debug("fork link %s skipped: %s", link, exc)


def fork_workspace(directory: str, session_key: str, settings: WorkspaceSettings) -> Fork | None:
    """Fork *directory* for *session_key*'s thread; None when it holds no repository to fork."""
    source, name = Path(directory), fork_name(session_key)
    repos = _local_repos(source)
    if repos == [source]:
        made = ensure_tree(source, name, settings.fetch_fresh_seconds)
        return Fork(str(made[0]), ((str(source), str(made[0])),)) if made else None
    root = source / ".worktrees" / name
    rows, trees = [], []
    for repo in repos:
        made = ensure_tree(repo, name, settings.fetch_fresh_seconds)
        if made is None:
            rows.append(f"- `{repo.name}`: `{repo}` (fork failed; work in the shared checkout with care)")
            continue
        root.mkdir(parents=True, exist_ok=True)
        _link(root, repo.name, made[0])
        trees.append((str(repo), str(made[0])))
        rows.append(f"- `{repo.name}`: `{made[0]}` ({made[1]})")
    cloned = {repo.name for repo in repos}
    for spec in settings.repos:
        if spec.path not in cloned:
            rows.append(f"- `{spec.path}`: not cloned: `git clone --filter=blob:none {spec.url} {source / spec.path}`")
    if not rows:
        return None
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text(_index(source, root, name, rows), encoding="utf-8")
    return Fork(str(root), tuple(trees))
