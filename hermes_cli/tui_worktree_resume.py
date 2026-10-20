"""Bring back a ``hermes --tui -w`` worktree a resume needs (N8).

Quitting a clean, idle ``--tui -w`` session removes its tree and branch, but the session's policy
froze that cwd, and the owner refuses every turn in a missing directory (``workspace_unavailable``).
So the launcher records each tree it creates in the profile's ``tui_worktrees.json``, and a resume
whose session cwd is such a missing tree recreates it at the same path before attaching: from its
branch when that still exists, else a new branch of the same name from the repo's current HEAD.

Only a path this profile's launcher recorded, directly under its repo's ``.worktrees/``, is ever
created; any other missing directory is left alone (the resume warns as before).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_RECORD_NAME = "tui_worktrees.json"
_MAX_RECORDS = 500
_holder = threading.local()


def _record_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _RECORD_NAME


def _key(path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _read() -> dict:
    from utils import read_json_or_empty
    trees = read_json_or_empty(_record_path()).get("trees")
    return trees if isinstance(trees, dict) else {}


def record_tui_worktree(wt_info: dict) -> None:
    """Remember a tree the ``--tui -w`` launcher created, or a classic ``/worktree new`` moved its
    gateway session into. Never raises."""
    try:
        from hermes_cli.auth import _file_lock
        from utils import atomic_json_write

        path = _record_path()
        with _file_lock(path.with_name(path.name + ".lock"), _holder, 5.0,
                        "Timed out waiting for the TUI worktree record lock"):
            trees = {k: v for k, v in _read().items()
                     if isinstance(v, dict) and (Path(str(v.get("repo_root") or "")) / ".git").exists()}
            trees[_key(wt_info["path"])] = {"path": str(wt_info["path"]), "branch": wt_info["branch"],
                                            "repo_root": str(wt_info["repo_root"])}
            atomic_json_write(path, {"version": 1, "trees": dict(list(trees.items())[-_MAX_RECORDS:])})
    except Exception:  # health: allow BLE001 -- a lost record only means no recreate on resume
        logger.debug("TUI worktree record failed for %s", wt_info.get("path"), exc_info=True)


def _recorded_tree_for(cwd: str) -> Optional[dict]:
    """The launcher's record of the missing tree *cwd* is (or is inside), or None."""
    try:
        trees = _read()
    except Exception:  # health: allow BLE001 -- unreadable record: create nothing
        return None
    probe = Path(cwd)
    for candidate in (probe, *probe.parents):
        entry = trees.get(_key(candidate))
        if not isinstance(entry, dict):
            continue
        repo = Path(str(entry.get("repo_root") or ""))
        tree = Path(str(entry.get("path") or ""))
        branch = str(entry.get("branch") or "")
        # Only a direct child of the repo's Hermes worktree dir, as _setup_worktree makes them.
        if (branch and (repo / ".git").exists()
                and _key(tree.parent) == _key(repo / ".worktrees") and _key(tree) == _key(candidate)):
            return {"path": str(tree), "branch": branch, "repo_root": str(repo)}
        return None
    return None


def _branch_exists(repo_root: str, branch: str) -> bool:
    from hermes_cli.worktree_ops import _git
    result = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo_root)
    return result.returncode == 0


def recreate_tui_worktree(cwd: str, session_id: str) -> Optional[str]:
    """Recreate the launcher-made tree a resumed session's missing *cwd* names -> its path, or None.

    Reuses the branch when it survived (its commits come back), else makes it again from the repo's
    current HEAD. The tree is locked for the session so the stale-tree prune keeps it. Never raises.
    """
    if not cwd or os.path.isdir(cwd):
        return None
    entry = _recorded_tree_for(cwd)
    if entry is None:
        return None
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    from hermes_cli.worktree_ops import (
        _copy_worktree_includes,
        _git,
        _git_quiet,
        _retain_worktree_for_session,
    )
    from hermes_constants import get_hermes_home

    repo_root, tree, branch = entry["repo_root"], Path(entry["path"]), entry["branch"]
    if tree.exists():  # a file, or a dir that appeared since: never overwrite it
        return None
    env = noninteractive_repo_git_env(repo_root)
    if env is None:
        return None
    try:
        # A tree deleted by hand leaves its admin entry; drop it so the path can be added again.
        _git_quiet(["worktree", "prune"], repo_root, timeout=15, stdin=subprocess.DEVNULL, env=env)
        tree.parent.mkdir(parents=True, exist_ok=True)
        args = ([str(tree), branch] if _branch_exists(repo_root, branch)
                else [str(tree), "-b", branch, "HEAD"])
        result = _git(["worktree", "add", *args], repo_root, timeout=120, stdin=subprocess.DEVNULL, env=env)
        if result.returncode != 0:
            logger.warning("could not recreate worktree %s: %s", tree, result.stderr.strip())
            return None
        _copy_worktree_includes(repo_root, tree)
        _retain_worktree_for_session(repo_root, str(tree), session_id, get_hermes_home() / "state.db")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not recreate worktree %s: %s", tree, exc)
        return None
    return str(tree)
