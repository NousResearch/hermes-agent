"""Git dashboard routes (extracted verbatim from web_server.py).

Handler bodies are byte-identical to their previous in-web_server form; the
helpers they call (``_git_op``, ``_git_path``) still live in web_server and are
reached via the late-binding seam in :mod:`hermes_cli.web_deps`, so
``monkeypatch.setattr(web_server, ...)`` keeps working.
"""

from typing import Optional

from fastapi import APIRouter

from hermes_cli import web_git as _web_git  # noqa: F401 — used by handlers
from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    GitPathBody,
    GitFileBody,
    GitCommitBody,
    GitPrListBody,
    GitWorktreeAddBody,
    GitWorktreeRemoveBody,
    GitBranchSwitchBody,
)

router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_git_op = late("_git_op")
_git_path = late("_git_path")


@router.get("/api/git/status")
async def git_status_route(path: str):
    return await _git_op(_web_git.repo_status, _git_path(path))


# ─── gh CLI auth probe ───────────────────────────────────────────────────────
# Cached `gh auth status` result. Consumed by the desktop composer's GitHub
# suggestion pill: GitHub deliberately has NO MCP catalog entry (its hosted
# MCP requires a per-host OAuth app — generic DCR 404s — and the bundled
# github/* skills via gh CLI are the more capable integration), so the pill
# offers the `/github-auth` skill instead, and only to users who aren't
# already authenticated. The probe is read-only and never prompts.

_GH_AUTH_TTL_S = 300.0
_gh_auth_cache: Optional[tuple] = None  # (monotonic_ts, payload)


@router.get("/api/git/gh-auth")
async def gh_auth_status_route(refresh: bool = False):
    """Report whether the `gh` CLI is present and authenticated.

    Returns ``{"available": bool, "authenticated": bool}``. Cached for five
    minutes (`refresh=true` bypasses — the pill uses it after a completed
    login so the suggestion withdraws immediately).
    """
    global _gh_auth_cache
    import asyncio
    import time

    if not refresh and _gh_auth_cache and time.monotonic() - _gh_auth_cache[0] < _GH_AUTH_TTL_S:
        return _gh_auth_cache[1]

    def _probe() -> dict:
        import shutil
        import subprocess

        gh = shutil.which("gh")
        if not gh:
            return {"available": False, "authenticated": False}
        try:
            # `gh auth status` exits 0 when at least one host is logged in.
            # Never interactive; DEVNULL stdin guards against any prompt.
            proc = subprocess.run(
                [gh, "auth", "status"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=10,
            )
            return {"available": True, "authenticated": proc.returncode == 0}
        except Exception:
            return {"available": True, "authenticated": False}

    payload = await asyncio.to_thread(_probe)
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
        for key, value in {
            "name": body.name,
            "branch": body.branch,
            "base": body.base,
            "existingBranch": body.existingBranch,
        }.items()
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
