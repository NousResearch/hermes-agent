"""Chat workspace picker: which host directories a NEW dashboard chat can start in.

The dashboard's ``/chat`` tab used to spawn every fresh TUI in the dashboard process's own
launch directory, so from a phone or browser there was no way to say "work in ~/code/foo".
This lists the same projects and discovered repositories the Desktop sidebar shows (the
per-profile ``projects.db`` plus session-derived and scanned git roots) so the SPA can offer
a picker, and ``/api/pty?cwd=`` (``chat_ws``) honours the choice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import http_failure, scoped_to_thread

router = APIRouter()

_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")


def resolve_chat_cwd(raw: Optional[str]) -> Optional[str]:
    """Validate a ``?cwd=`` for a fresh chat: an existing host directory, or None when unset.

    A dead or relative path fails closed (400) instead of silently falling back to the launch
    dir — the user picked a workspace and would otherwise get a session in the wrong place.
    """
    text = (raw or "").strip()
    if not text:
        return None
    resolved = os.path.abspath(os.path.expanduser(text))
    if not os.path.isdir(resolved):
        raise HTTPException(status_code=400, detail=f"Working directory does not exist: {text}")
    return resolved


def _collect_workspaces(profile: Optional[str], scan: bool) -> dict:
    # The dashboard hosts the in-process gateway (``web_server`` imports ``tui_gateway.server``
    # at startup), so the sidebar's repo-discovery helpers are already bound there.
    import tui_gateway.server as gateway
    from hermes_cli import projects_db as pdb

    # Browsing never bootstraps the owner's session store: a profile whose gateway has not
    # yet initialized state.db simply contributes no session-derived roots (``db=None``).
    try:
        db = _open_session_db_for_profile(profile, read_only=True)
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        db = None
    try:
        with pdb.connect_closing() as conn:
            policy = gateway._repo_discovery_policy()
            if scan and policy["enabled"]:
                gateway._scan_discovered_repos_remote(conn, policy)
            projects = [p.to_dict() for p in pdb.list_projects(conn)]
            repos = gateway._discover_repos_payload(
                db, conn=conn, backfill=False, include_cached=policy["enabled"])
    finally:
        if db is not None:
            db.close()
    default_cwd = gateway._completion_cwd({"profile": profile} if profile else {})
    return {
        "projects": projects, "repos": repos, "default_cwd": default_cwd,
        "home": str(Path.home()), "scan_enabled": bool(policy["enabled"])}


@router.get("/api/chat/workspaces")
async def get_chat_workspaces(profile: Optional[str] = None, scan: bool = False):
    """Projects + discovered repos a fresh chat may start in; ``scan=1`` rescans the
    configured discovery roots on the host first (headless installs have no Desktop to do it)."""
    with http_failure("GET /api/chat/workspaces failed", 500, "Failed to list workspaces"):
        return await scoped_to_thread(profile, lambda: _collect_workspaces(profile, scan))
