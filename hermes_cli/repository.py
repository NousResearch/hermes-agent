"""
Repository state — Git local reads + optional GitHub API.

Environment variables (all optional):
  HERMES_REPOSITORY_NAME           — display name (default: dirname of local_path)
  HERMES_REPOSITORY_OWNER          — GitHub owner
  HERMES_REPOSITORY_FULL_NAME      — owner/repo
  HERMES_REPOSITORY_LOCAL_PATH     — absolute path to local git repo
  HERMES_REPOSITORY_DEFAULT_BRANCH — default branch (default: main)
  GITHUB_TOKEN                     — GitHub PAT; never returned to clients
"""

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)
_GIT_TIMEOUT = 10  # seconds per subprocess call


def _run_git(args: List[str], cwd: str) -> Optional[str]:
    """Run git in cwd; return stripped stdout or None on any error."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as exc:
        _log.debug("git %s: %s", args, exc)
        return None


def _repo_config() -> Dict[str, str]:
    return {
        "name": os.getenv("HERMES_REPOSITORY_NAME", ""),
        "owner": os.getenv("HERMES_REPOSITORY_OWNER", ""),
        "full_name": os.getenv("HERMES_REPOSITORY_FULL_NAME", ""),
        "local_path": os.getenv("HERMES_REPOSITORY_LOCAL_PATH", ""),
        "default_branch": os.getenv("HERMES_REPOSITORY_DEFAULT_BRANCH", "main"),
        "github_token": os.getenv("GITHUB_TOKEN", ""),
    }


def _read_local(local_path: str, default_branch: str) -> Dict[str, Any]:
    """Read local git state. Partial result on error — never raises."""
    out: Dict[str, Any] = {
        "current_branch": None,
        "local_head_sha": None,
        "local_head_short_sha": None,
        "last_local_commit": None,
        "ahead_by": 0,
        "behind_by": 0,
        "has_uncommitted_changes": False,
        "modified_files": 0,
        "staged_files": 0,
        "untracked_files": 0,
    }

    if not local_path or not Path(local_path).is_dir():
        return out
    if not _run_git(["rev-parse", "--git-dir"], local_path):
        return out

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], local_path)
    if branch == "HEAD":
        branch = _run_git(["rev-parse", "--short", "HEAD"], local_path) or "detached"
    out["current_branch"] = branch

    head = _run_git(["rev-parse", "HEAD"], local_path)
    if head:
        out["local_head_sha"] = head
        out["local_head_short_sha"] = head[:8]

    SEP = "\x1f"
    log_raw = _run_git(
        ["log", "-1", f"--pretty=format:%H{SEP}%h{SEP}%s{SEP}%an{SEP}%ae{SEP}%cI"],
        local_path,
    )
    if log_raw:
        parts = log_raw.split(SEP)
        if len(parts) >= 6:
            out["last_local_commit"] = {
                "sha": parts[0],
                "short_sha": parts[1],
                "message": parts[2],
                "author_name": parts[3],
                "author_email": parts[4],
                "committed_at": parts[5],
            }

    # ahead/behind relative to remote tracking branch
    remote_ref = f"origin/{branch or default_branch}"
    diverge = _run_git(
        ["rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"], local_path
    )
    if diverge:
        parts_d = diverge.split()
        if len(parts_d) == 2:
            try:
                out["ahead_by"] = int(parts_d[0])
                out["behind_by"] = int(parts_d[1])
            except ValueError:
                pass

    porcelain = _run_git(["status", "--porcelain=v1"], local_path)
    if porcelain is not None:
        modified = staged = untracked = 0
        for line in porcelain.splitlines():
            if len(line) < 2:
                continue
            x, y = line[0], line[1]
            if x == "?" and y == "?":
                untracked += 1
            else:
                if x not in (" ", "?"):
                    staged += 1
                if y not in (" ", "?"):
                    modified += 1
        out["modified_files"] = modified
        out["staged_files"] = staged
        out["untracked_files"] = untracked
        out["has_uncommitted_changes"] = (modified + staged + untracked) > 0

    return out


def _fetch_github_commit(
    owner: str, repo: str, branch: str, token: str
) -> Optional[Dict[str, Any]]:
    """Fetch latest commit from GitHub API. Returns None on any failure."""
    if not owner or not repo:
        return None
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "hermes-agent/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        sha = data.get("sha", "")
        commit = data.get("commit", {})
        committer = commit.get("committer") or commit.get("author") or {}
        author_login = (data.get("author") or {}).get("login", "")
        message = (commit.get("message") or "").split("\n")[0]
        return {
            "sha": sha,
            "short_sha": sha[:8] if sha else "",
            "message": message,
            "author_name": author_login or committer.get("name", ""),
            "committed_at": committer.get("date", ""),
            "url": data.get("html_url", ""),
        }
    except Exception as exc:
        _log.debug("GitHub API: %s", exc)
        return None


def _compute_sync_status(
    has_changes: bool, ahead: int, behind: int, github_ok: bool
) -> str:
    if not github_ok:
        return "dirty" if has_changes else "unknown"
    if ahead > 0 and behind > 0:
        return "diverged"
    if behind > 0:
        return "behind"
    if ahead > 0:
        return "ahead"
    if has_changes:
        return "dirty"
    return "clean"


def get_repository_state() -> Dict[str, Any]:
    """Build RepositoryState dict. Never raises — returns partial data on error."""
    cfg = _repo_config()
    local_path = cfg["local_path"]
    default_branch = cfg["default_branch"]

    name = cfg["name"] or (Path(local_path).name if local_path else "unknown")
    owner = cfg["owner"]
    full_name = cfg["full_name"] or (f"{owner}/{name}" if owner and name else "")
    repo_slug = full_name.split("/")[-1] if full_name else name

    local = _read_local(local_path, default_branch)
    current_branch = local.get("current_branch") or default_branch

    token = cfg["github_token"]
    github_commit = _fetch_github_commit(owner, repo_slug, current_branch, token) if owner else None
    github_ok = github_commit is not None

    ahead = local.get("ahead_by", 0)
    behind = local.get("behind_by", 0)
    has_changes = local.get("has_uncommitted_changes", False)

    return {
        "id": full_name or name,
        "name": name,
        "owner": owner,
        "full_name": full_name,
        "provider": "github",
        "local_path": local_path or None,
        "default_branch": default_branch,
        "current_branch": current_branch,
        "local_head_sha": local.get("local_head_sha"),
        "local_head_short_sha": local.get("local_head_short_sha"),
        "remote_head_sha": github_commit.get("sha") if github_commit else None,
        "remote_head_short_sha": github_commit.get("short_sha") if github_commit else None,
        "ahead_by": ahead,
        "behind_by": behind,
        "has_uncommitted_changes": has_changes,
        "modified_files": local.get("modified_files", 0),
        "staged_files": local.get("staged_files", 0),
        "untracked_files": local.get("untracked_files", 0),
        "sync_status": _compute_sync_status(has_changes, ahead, behind, github_ok),
        "last_local_commit": local.get("last_local_commit"),
        "last_remote_commit": github_commit,
        "github_available": github_ok,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
