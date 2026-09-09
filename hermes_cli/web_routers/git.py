"""Git dashboard routes — the remote half of the desktop coding rail + review pane.

The desktop runs these as Electron-local git; over a remote gateway that's the wrong
filesystem, so they are mirrored here with the same auth gate + path hardening as
/api/fs. Logic lives in ``hermes_cli.web_git``; these are thin executor-offloaded
wrappers (git/gh can block).
"""

import asyncio
import shutil
import subprocess
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from hermes_cli import web_git as _web_git
from hermes_cli.web_deps import late
from hermes_cli.web_server_files import _fs_path
from hermes_cli.web_models import (
    GitBranchSwitchBody,
    GitCommitBody,
    GitFileBody,
    GitPathBody,
    GitPrListBody,
    GitWorktreeAddBody,
    GitWorktreeRemoveBody,
)

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.


async def _git_op(fn, *args):
    """Run a (blocking) git op off the event loop; map a failed mutation to 400."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, fn, *args)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "git operation failed")


def _git_path(path: str) -> str:
    return str(_fs_path(path))


@router.get("/api/git/status")
async def git_status_route(path: str):
    return await _git_op(_web_git.repo_status, _git_path(path))


# Cached `gh auth status` for the desktop composer's GitHub suggestion pill. GitHub
# deliberately has NO MCP catalog entry (hosted MCP needs a per-host OAuth app; the
# gh-CLI skills are the better integration), so the pill offers `/github-auth` —
# only to users who aren't already authenticated.
_GH_AUTH_TTL_S = 300.0
_gh_auth_cache: Optional[tuple] = None  # (monotonic_ts, payload)


def _probe_gh_auth() -> dict:
    gh = shutil.which("gh")
    if not gh:
        return {"available": False, "authenticated": False}
    try:
        # Exits 0 when at least one host is logged in; DEVNULL stdin guards against any prompt.
        proc = subprocess.run(
            [gh, "auth", "status"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
        return {"available": True, "authenticated": proc.returncode == 0}
    except Exception:
        return {"available": True, "authenticated": False}


@router.get("/api/git/gh-auth")
async def gh_auth_status_route(refresh: bool = False):
    """``{"available", "authenticated"}`` for the `gh` CLI; cached 5 min
    (``refresh=true`` bypasses so the pill withdraws right after a login)."""
    global _gh_auth_cache
    if not refresh and _gh_auth_cache and time.monotonic() - _gh_auth_cache[0] < _GH_AUTH_TTL_S:
        return _gh_auth_cache[1]
    payload = await asyncio.to_thread(_probe_gh_auth)
    _gh_auth_cache = (time.monotonic(), payload)
    return payload


@router.get("/api/git/worktrees")
async def git_worktrees_route(path: str):
    return {"worktrees": await _git_op(_web_git.worktree_list, _git_path(path))}


@router.get("/api/git/branches")
async def git_branches_route(path: str):
    return {"branches": await _git_op(_web_git.branch_list, _git_path(path))}


@router.get("/api/git/base-branches")
async def git_base_branches_route(path: str):
    return {"branches": await _git_op(_web_git.base_branch_list, _git_path(path))}


@router.get("/api/git/review/list")
async def git_review_list_route(path: str, scope: str = "uncommitted", base: Optional[str] = None):
    return await _git_op(_web_git.review_list, _git_path(path), scope, base)


@router.get("/api/git/review/diff")
async def git_review_diff_route(
    path: str, file: str, scope: str = "uncommitted", base: Optional[str] = None, staged: bool = False
):
    return {"diff": await _git_op(_web_git.review_diff, _git_path(path), file, scope, base, staged)}


@router.get("/api/git/file-diff")
async def git_file_diff_route(path: str, file: str):
    return {"diff": await _git_op(_web_git.file_diff_vs_head, _git_path(path), file)}


@router.get("/api/git/review/commit-context")
async def git_commit_context_route(path: str):
    return await _git_op(_web_git.review_commit_context, _git_path(path))


@router.get("/api/git/review/rev-parse")
async def git_rev_parse_route(path: str, ref: Optional[str] = None):
    return {"sha": await _git_op(_web_git.review_rev_parse, _git_path(path), ref)}


@router.get("/api/git/review/ship-info")
async def git_ship_info_route(path: str):
    return await _git_op(_web_git.review_ship_info, _git_path(path))


@router.post("/api/git/review/pr-list")
async def git_pr_list_route(body: GitPrListBody):
    return await _git_op(_web_git.review_pr_list, _git_path(body.path), body.branches, body.numbers)


@router.post("/api/git/review/stage")
async def git_stage_route(body: GitFileBody):
    return await _git_op(_web_git.review_stage, _git_path(body.path), body.file)


@router.post("/api/git/review/unstage")
async def git_unstage_route(body: GitFileBody):
    return await _git_op(_web_git.review_unstage, _git_path(body.path), body.file)


@router.post("/api/git/review/revert")
async def git_revert_route(body: GitFileBody):
    return await _git_op(_web_git.review_revert, _git_path(body.path), body.file)


@router.post("/api/git/review/commit")
async def git_commit_route(body: GitCommitBody):
    return await _git_op(_web_git.review_commit, _git_path(body.path), body.message, body.push)


@router.post("/api/git/review/push")
async def git_push_route(body: GitPathBody):
    return await _git_op(_web_git.review_push, _git_path(body.path))


@router.post("/api/git/review/create-pr")
async def git_create_pr_route(body: GitPathBody):
    return await _git_op(_web_git.review_create_pr, _git_path(body.path))


@router.post("/api/git/worktree/add")
async def git_worktree_add_route(body: GitWorktreeAddBody):
    options = {
        key: value
        for key, value in body.model_dump(include={"name", "branch", "base", "existingBranch"}).items()
        if value
    }
    return await _git_op(_web_git.worktree_add, _git_path(body.path), options)


@router.post("/api/git/worktree/remove")
async def git_worktree_remove_route(body: GitWorktreeRemoveBody):
    return await _git_op(
        _web_git.worktree_remove, _git_path(body.path), _git_path(body.worktreePath), body.force
    )


@router.post("/api/git/branch/switch")
async def git_branch_switch_route(body: GitBranchSwitchBody):
    return await _git_op(_web_git.branch_switch, _git_path(body.path), body.branch)
