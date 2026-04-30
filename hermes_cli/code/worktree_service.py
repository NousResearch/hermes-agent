#!/usr/bin/env python3
"""Safe git capability and worktree foundation for Code Mode."""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from hermes_state import SessionDB


def _run_git(path: Path, args: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(path),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_worktree_list(output: str) -> list[dict[str, Any]]:
    worktrees: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line.removeprefix("worktree ")
        elif line.startswith("HEAD "):
            current["head"] = line.removeprefix("HEAD ")
        elif line.startswith("branch "):
            current["branch"] = line.removeprefix("branch ")
        elif line == "detached":
            current["detached"] = True
        elif line == "bare":
            current["bare"] = True
    if current:
        worktrees.append(current)
    return worktrees


class WorktreeService:
    def __init__(self, db_path=None):
        self._db_path = db_path

    def _db(self) -> SessionDB:
        return SessionDB(db_path=self._db_path) if self._db_path else SessionDB()

    @staticmethod
    def detect_git_capabilities(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {
                "path": str(path),
                "is_git_repo": False,
                "supports_worktree": False,
                "current_branch": None,
                "is_dirty": False,
                "has_commits": False,
                "toplevel": None,
            }

        try:
            is_repo = _run_git(path, ["rev-parse", "--is-inside-work-tree"]).stdout.strip() == "true"
        except Exception:
            is_repo = False
        if not is_repo:
            return {
                "path": str(path),
                "is_git_repo": False,
                "supports_worktree": False,
                "current_branch": None,
                "is_dirty": False,
                "has_commits": False,
                "toplevel": None,
            }

        toplevel = None
        try:
            r = _run_git(path, ["rev-parse", "--show-toplevel"])
            if r.returncode == 0:
                toplevel = r.stdout.strip()
        except Exception:
            pass
        current_branch = None
        try:
            r = _run_git(path, ["branch", "--show-current"])
            if r.returncode == 0:
                current_branch = r.stdout.strip() or None
        except Exception:
            pass
        has_commits = False
        try:
            has_commits = _run_git(path, ["rev-parse", "--verify", "HEAD"]).returncode == 0
        except Exception:
            pass
        is_dirty = False
        if has_commits:
            try:
                is_dirty = bool(_run_git(path, ["status", "--porcelain"]).stdout.strip())
            except Exception:
                pass
        supports_worktree = False
        try:
            supports_worktree = _run_git(path, ["worktree", "list", "--porcelain"]).returncode == 0
        except Exception:
            pass
        return {
            "path": str(path),
            "is_git_repo": True,
            "supports_worktree": supports_worktree,
            "current_branch": current_branch,
            "is_dirty": is_dirty,
            "has_commits": has_commits,
            "toplevel": toplevel,
        }

    def _workspace_path(self, workspace_id: str) -> Path:
        db = self._db()
        try:
            workspace = db.get_code_workspace(workspace_id)
        finally:
            db.close()
        if not workspace or not workspace.get("path"):
            raise ValueError(f"Workspace not found or path missing: {workspace_id}")
        return Path(workspace["path"])

    def detect_capabilities_for_workspace(self, workspace_id: str) -> dict[str, Any]:
        try:
            return self.detect_git_capabilities(self._workspace_path(workspace_id))
        except Exception:
            return {
                "path": None,
                "is_git_repo": False,
                "supports_worktree": False,
                "current_branch": None,
                "is_dirty": False,
                "has_commits": False,
                "toplevel": None,
            }

    def list_worktrees_for_workspace(self, workspace_id: str) -> list[dict[str, Any]]:
        try:
            path = self._workspace_path(workspace_id)
        except Exception:
            return []
        caps = self.detect_git_capabilities(path)
        if not caps["is_git_repo"] or not caps["supports_worktree"]:
            return []
        try:
            result = _run_git(path, ["worktree", "list", "--porcelain"])
            if result.returncode != 0:
                return []
            return _parse_worktree_list(result.stdout)
        except Exception:
            return []

    def create_checkpoint_metadata(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        path = self._workspace_path(workspace_id)
        caps = self.detect_git_capabilities(path)
        head_sha = None
        if caps["is_git_repo"] and caps["has_commits"]:
            try:
                result = _run_git(path, ["rev-parse", "HEAD"])
                if result.returncode == 0:
                    head_sha = result.stdout.strip()
            except Exception:
                pass

        checkpoint = {
            "id": str(uuid.uuid4()),
            "workspace_id": workspace_id,
            "name": name or "checkpoint",
            "git_branch": caps.get("current_branch"),
            "git_commit": head_sha,
            "dirty": bool(caps.get("is_dirty")),
            "metadata": {"capabilities": caps},
            "created_at": time.time(),
        }
        db = self._db()
        try:
            db.create_code_checkpoint(checkpoint)
        finally:
            db.close()
        return checkpoint
