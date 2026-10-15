"""Completion gate for git-backed workspaces.

When a task completes whose workspace is a git checkout (worktree or dir),
establish whether the work is landed, in-flight, or unlanded. Record the
state in the completion metadata so the board answers "is anything stuck
behind an open PR?" by query, not audit.

Design choice: complete with an explicit ``unlanded_work`` metadata field
rather than refusing. Rationale: refusing teaches workers to lie (omit
mention of code changes), and the operator's complaint is about SILENT
unlanded work — the metadata field makes the board queryable. A hard
block would create worse failure modes (workers bypass it, cards get
stuck in limbo).

Degrades gracefully: network failure, missing ``gh`` CLI, or offline
machine produces a warning on the card, never a hard block.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], *, cwd: Optional[Path] = None, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a command; never raises on non-zero exit or timeout."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return subprocess.CompletedProcess(cmd, 1, "", "command unavailable or timed out")


def _git(repo_root: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return _run(["git", "-C", str(repo_root), *args], timeout=timeout)


def _gh(repo_root: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """``gh`` CLI in the repo's cwd so it picks up the right remote."""
    return _run(["gh", *args], cwd=repo_root, timeout=timeout)


def _is_git_repo(path: Path) -> bool:
    if not path.exists():
        return False
    result = _git(path, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _get_repo_root(path: Path) -> Optional[Path]:
    if not _is_git_repo(path):
        return None
    result = _git(path, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def _get_current_branch(repo_root: Path) -> Optional[str]:
    result = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else None


def _has_remote_branch(repo_root: Path, branch: str) -> bool:
    result = _git(repo_root, "rev-parse", "--verify", f"origin/{branch}")
    return result.returncode == 0


def _is_ancestor_of_main(repo_root: Path, branch: str) -> bool:
    """True if branch tip is an ancestor of origin/main (landed)."""
    # Best-effort fetch; offline degrades to whatever refs are cached.
    _git(repo_root, "fetch", "origin", "main", "--prune", timeout=8)
    result = _git(repo_root, "merge-base", "--is-ancestor", branch, "origin/main")
    return result.returncode == 0


def _has_unmerged_commits(repo_root: Path, branch: str) -> bool:
    """True if branch has commits not on origin/main."""
    _git(repo_root, "fetch", "origin", "main", "--prune", timeout=8)
    result = _git(repo_root, "rev-list", "--count", f"origin/main..{branch}")
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) > 0
    except ValueError:
        return False


def _find_open_pr_for_branch(repo_root: Path, branch: str) -> Optional[dict]:
    """Find an open PR for this branch via ``gh pr list``.

    Returns {"number": int, "url": str, "title": str, "matches_card": bool}
    or None. Degrades to None if gh is unavailable or network fails.
    """
    result = _gh(
        repo_root,
        "pr", "list",
        "--head", branch,
        "--state", "open",
        "--json", "number,url,title,body",
        "--limit", "5",
        timeout=10,
    )
    if result.returncode != 0:
        return None

    try:
        prs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    if not isinstance(prs, list) or not prs:
        return None

    pr = prs[0]
    if not isinstance(pr, dict):
        return None

    return {
        "number": pr.get("number"),
        "url": pr.get("url"),
        "title": str(pr.get("title", "")),
        "body": str(pr.get("body", "") or ""),
    }


def _pr_references_card(pr: dict, card_id: str) -> bool:
    """True if the PR title or body references the card id."""
    return card_id in pr.get("title", "") or card_id in pr.get("body", "")


def evaluate_git_completion(
    workspace_path: Optional[str],
    branch_name: Optional[str],
    card_id: str,
) -> Optional[dict]:
    """Evaluate the git state of a completing task.

    Returns a dict with keys:
      - branch: str (current branch)
      - pushed: bool (branch exists on origin)
      - landed: bool (branch tip is ancestor of origin/main)
      - unmerged_commits: bool (branch has commits not on origin/main)
      - pr: dict or None (open PR info if found)
      - unlanded: bool (work exists but is neither landed nor in open PR)
      - degraded: bool (network/gh failure — gate couldn't fully evaluate)

    Returns None if the workspace is not a git repo or has no branch.
    """
    if not workspace_path:
        return None

    ws = Path(workspace_path)
    if not _is_git_repo(ws):
        return None

    repo_root = _get_repo_root(ws)
    if repo_root is None:
        return None

    branch = branch_name or _get_current_branch(repo_root)
    if not branch:
        return None

    pushed = _has_remote_branch(repo_root, branch)
    unmerged = _has_unmerged_commits(repo_root, branch)

    # Landed check requires origin/main to be reachable.
    landed = False
    degraded = False
    if pushed:
        landed = _is_ancestor_of_main(repo_root, branch)
        if not landed:
            # Could be offline (fetch failed) — check if origin/main even exists locally.
            probe = _git(repo_root, "rev-parse", "--verify", "origin/main")
            if probe.returncode != 0:
                degraded = True

    pr = _find_open_pr_for_branch(repo_root, branch)
    if pr is None and pushed and unmerged:
        # gh might be unavailable — mark degraded rather than assuming no PR.
        probe = _run(["which", "gh"])
        if probe.returncode != 0:
            degraded = True

    unlanded = unmerged and not landed and pr is None

    return {
        "branch": branch,
        "pushed": pushed,
        "landed": landed,
        "unmerged_commits": unmerged,
        "pr": pr,
        "unlanded": unlanded,
        "degraded": degraded,
    }


def apply_git_gate_to_metadata(
    metadata: Optional[dict],
    git_state: Optional[dict],
    card_id: str,
) -> Optional[dict]:
    """Merge git gate state into completion metadata.

    If git_state is None (non-git workspace or no branch), metadata is unchanged.
    If unlanded, adds ``unlanded_work`` field with branch, pr state, merge owner.
    Always adds ``git_completion`` field with full state for queryability.
    """
    if git_state is None:
        return metadata

    if metadata is None:
        metadata = {}

    pr_number = None
    pr_url = None
    pr_matches_card = False
    if git_state.get("pr"):
        pr_number = git_state["pr"].get("number")
        pr_url = git_state["pr"].get("url")
        pr_matches_card = _pr_references_card(git_state["pr"], card_id)

    metadata["git_completion"] = {
        "branch": git_state["branch"],
        "pushed": git_state["pushed"],
        "landed": git_state["landed"],
        "unmerged_commits": git_state["unmerged_commits"],
        "pr_number": pr_number,
        "pr_url": pr_url,
        "pr_matches_card": pr_matches_card,
        "degraded": git_state.get("degraded", False),
    }

    if git_state["unlanded"]:
        metadata["unlanded_work"] = {
            "branch": git_state["branch"],
            "pr_state": "none",
            "merge_owner": "nobody — card completed without landing the work",
            "message": (
                f"Branch {git_state['branch']} has unmerged commits but no open PR. "
                "Work is not on main. The card completed without landing the work."
            ),
        }
    elif git_state.get("pr") and not pr_matches_card:
        # PR exists but doesn't reference the card — note it for the audit.
        metadata["unlanded_work"] = {
            "branch": git_state["branch"],
            "pr_state": f"open (PR #{pr_number}) but does not reference {card_id}",
            "pr_url": pr_url,
            "merge_owner": "PR author — card id not in PR title/body",
            "message": (
                f"Branch {git_state['branch']} has open PR #{pr_number} but it does not "
                f"reference {card_id}. Work is not yet on main."
            ),
        }

    return metadata
