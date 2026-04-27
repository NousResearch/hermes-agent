#!/usr/bin/env python3
"""
GitService — safe Git operations for Hermes Code Mode.

Read-only operations execute directly. Destructive operations require
approval or are blocked. All Git commands run with cwd inside the workspace.
"""

import json
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GitActionSafety:
    SAFE = "safe"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"


class GitService:
    """Business logic for safe Git operations in a workspace.

    Delegates persistence to GitSnapshotDB and WorkspaceDB (hermes_state).
    Does not execute destructive Git commands without safety checks.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _workspace_db(self):
        from hermes_state import WorkspaceDB

        return WorkspaceDB(db_path=self._db_path)

    def _snapshot_db(self):
        from hermes_state import GitSnapshotDB

        return GitSnapshotDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB

        return CodeSessionDB(db_path=self._db_path)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, payload)
            except Exception:
                pass

    def _add_timeline_event(
        self,
        code_session_id: Optional[str],
        event_type: str,
        message: str,
        payload: dict,
    ):
        if not code_session_id:
            return
        db = self._session_db()
        try:
            db.add_event(code_session_id, event_type, message=message, payload=payload)
        except Exception:
            pass
        finally:
            db.close()

    def _get_workspace_path(self, workspace_id: str) -> Path:
        wdb = self._workspace_db()
        try:
            workspace = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        ws_path = Path(workspace["path"]).resolve()
        if not ws_path.exists() or not ws_path.is_dir():
            raise ValueError(f"Workspace path does not exist: {ws_path}")

        return ws_path

    def _is_git_repo(self, workspace_path: Path) -> bool:
        try:
            result = self._run_git(
                workspace_path, ["rev-parse", "--is-inside-work-tree"]
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False

    def _run_git(
        self, workspace_path: Path, args: List[str], timeout: int = 10
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _validate_branch_name(branch_name: str) -> tuple[bool, str]:
        if not branch_name or not branch_name.strip():
            return False, "Branch name is empty"
        bn = branch_name.strip()
        if bn.startswith("-"):
            return False, "Branch name cannot start with '-'"
        if " " in bn or "\t" in bn or "\n" in bn:
            return False, "Branch name cannot contain whitespace"
        if ".." in bn:
            return False, "Branch name cannot contain '..'"
        if bn.endswith("."):
            return False, "Branch name cannot end with '.'"
        # Refuse names that look like paths or contain backslash
        if "/" in bn and bn.startswith("/"):
            return False, "Branch name cannot start with '/'"
        if "\\" in bn:
            return False, "Branch name cannot contain backslash"
        # Check for control characters
        if any(ord(c) < 32 for c in bn):
            return False, "Branch name cannot contain control characters"
        # Git ref name rules: ~ ^ : are special
        if any(c in bn for c in ["~", "^", ":"]):
            return False, "Branch name contains invalid characters"
        return True, ""

    def get_status(
        self, workspace_id: str, code_session_id: Optional[str] = None
    ) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            return {
                "workspace_id": workspace_id,
                "is_git_repo": False,
                "branch": None,
                "remote_url": None,
                "dirty": False,
                "ahead": None,
                "behind": None,
                "files": [],
                "summary": {
                    "modified": 0,
                    "added": 0,
                    "deleted": 0,
                    "renamed": 0,
                    "untracked": 0,
                    "staged": 0,
                    "unstaged": 0,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        branch = self._get_branch_raw(ws_path)
        remote_url = self._get_remote_raw(ws_path)

        files, summary = self._parse_status(ws_path)
        dirty = (
            summary["modified"]
            + summary["added"]
            + summary["deleted"]
            + summary["renamed"]
            + summary["untracked"]
            + summary["staged"]
            + summary["unstaged"]
            > 0
        )

        result = {
            "workspace_id": workspace_id,
            "is_git_repo": True,
            "branch": branch,
            "remote_url": remote_url,
            "dirty": dirty,
            "ahead": None,
            "behind": None,
            "files": files,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._add_timeline_event(
            code_session_id,
            "git.status_checked",
            message=f"Git status checked for workspace {workspace_id}",
            payload={
                "workspace_id": workspace_id,
                "branch": branch,
                "dirty": dirty,
                "summary": summary,
            },
        )

        return result

    def _get_branch_raw(self, workspace_path: Path) -> Optional[str]:
        result = self._run_git(workspace_path, ["branch", "--show-current"])
        if result.returncode == 0:
            branch = result.stdout.strip()
            if branch:
                return branch
        # Fallback
        result = self._run_git(workspace_path, ["rev-parse", "--abbrev-ref", "HEAD"])
        if result.returncode == 0:
            return result.stdout.strip() or None
        return None

    def _get_remote_raw(self, workspace_path: Path) -> Optional[str]:
        result = self._run_git(workspace_path, ["remote", "get-url", "origin"])
        if result.returncode == 0:
            return result.stdout.strip() or None
        return None

    def _parse_status(self, workspace_path: Path) -> tuple[List[dict], dict]:
        result = self._run_git(workspace_path, ["status", "--porcelain=v1"])
        files: List[dict] = []
        summary = {
            "modified": 0,
            "added": 0,
            "deleted": 0,
            "renamed": 0,
            "untracked": 0,
            "staged": 0,
            "unstaged": 0,
        }

        if result.returncode != 0:
            return files, summary

        for line in result.stdout.splitlines():
            if len(line) < 3:
                continue
            # Format: XY path or XY old -> new
            x = line[0]
            y = line[1]
            # After XY there's a space, then the path
            rest = line[3:]

            if x == "?" and y == "?":
                status = "untracked"
                summary["untracked"] += 1
            elif x == "A" or y == "A":
                status = "added"
                summary["added"] += 1
            elif x == "D" or y == "D":
                status = "deleted"
                summary["deleted"] += 1
            elif x == "R" or y == "R":
                status = "renamed"
                summary["renamed"] += 1
            elif x == "M" or y == "M":
                status = "modified"
                summary["modified"] += 1
            else:
                status = "unknown"

            staged = x not in (" ", "?", "") and x != "?"
            unstaged = y not in (" ", "?", "") and y != "?"

            if staged:
                summary["staged"] += 1
            if unstaged:
                summary["unstaged"] += 1

            # For renamed, extract the new path after ->
            path = rest
            if " -> " in rest:
                path = rest.split(" -> ", 1)[1]

            files.append(
                {
                    "path": path,
                    "index_status": x if x != " " else "",
                    "worktree_status": y if y != " " else "",
                    "status": status,
                    "staged": staged,
                    "unstaged": unstaged,
                }
            )

        return files, summary

    def get_diff(self, workspace_id: str, path: Optional[str] = None) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            raise ValueError("Workspace is not a Git repository")

        args = ["diff", "--no-ext-diff", "--"]
        if path:
            args.append(path)

        result = self._run_git(ws_path, args)
        diff_text = result.stdout if result.returncode == 0 else ""

        from hermes_state import count_diff_changes

        additions, deletions = count_diff_changes(diff_text)

        return {
            "workspace_id": workspace_id,
            "path": path,
            "diff": diff_text,
            "additions": additions,
            "deletions": deletions,
        }

    def get_branch(self, workspace_id: str) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            raise ValueError("Workspace is not a Git repository")

        branch = self._get_branch_raw(ws_path)
        return {
            "workspace_id": workspace_id,
            "branch": branch,
        }

    def get_remote(self, workspace_id: str) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            raise ValueError("Workspace is not a Git repository")

        remote_url = self._get_remote_raw(ws_path)
        return {
            "workspace_id": workspace_id,
            "remote_url": remote_url,
        }

    def list_files(self, workspace_id: str) -> List[dict]:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            raise ValueError("Workspace is not a Git repository")

        files, _ = self._parse_status(ws_path)
        return files

    def create_snapshot(
        self, workspace_id: str, code_session_id: Optional[str] = None
    ) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            raise ValueError("Workspace is not a Git repository")

        status = self.get_status(workspace_id, code_session_id=code_session_id)

        diff_result = self.get_diff(workspace_id)
        diff_stat = f"+{diff_result['additions']}/-{diff_result['deletions']}"

        db = self._snapshot_db()
        try:
            snapshot = db.create_snapshot(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                branch=status.get("branch"),
                remote_url=status.get("remote_url"),
                dirty=status.get("dirty", False),
                summary=status.get("summary"),
                files=status.get("files"),
                diff_stat=diff_stat,
            )
        finally:
            db.close()

        self._add_timeline_event(
            code_session_id,
            "git.snapshot.created",
            message=f"Git snapshot created for workspace {workspace_id}",
            payload={
                "workspace_id": workspace_id,
                "snapshot_id": snapshot["id"],
                "branch": status.get("branch"),
                "dirty": status.get("dirty"),
            },
        )

        return snapshot

    def prepare_branch(self, workspace_id: str, branch_name: str) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            return {
                "action": "create_branch",
                "safety": GitActionSafety.BLOCKED,
                "branch_name": branch_name,
                "reason": "Workspace is not a Git repository",
            }

        valid, reason = self._validate_branch_name(branch_name)
        if not valid:
            return {
                "action": "create_branch",
                "safety": GitActionSafety.BLOCKED,
                "branch_name": branch_name,
                "reason": reason,
            }

        # Check if dirty
        _, summary = self._parse_status(ws_path)
        dirty = (
            summary["modified"]
            + summary["added"]
            + summary["deleted"]
            + summary["renamed"]
            + summary["untracked"]
            + summary["staged"]
            + summary["unstaged"]
            > 0
        )

        if dirty:
            return {
                "action": "create_branch",
                "safety": GitActionSafety.NEEDS_APPROVAL,
                "branch_name": branch_name,
                "reason": "Workspace has uncommitted changes",
            }

        # Check if branch already exists
        result = self._run_git(ws_path, ["branch", "--list", branch_name])
        if result.stdout.strip():
            return {
                "action": "create_branch",
                "safety": GitActionSafety.BLOCKED,
                "branch_name": branch_name,
                "reason": f"Branch '{branch_name}' already exists",
            }

        return {
            "action": "create_branch",
            "safety": GitActionSafety.SAFE,
            "branch_name": branch_name,
            "reason": "",
        }

    def create_branch(
        self, workspace_id: str, branch_name: str, code_session_id: Optional[str] = None
    ) -> dict:
        preparation = self.prepare_branch(workspace_id, branch_name)

        if preparation["safety"] != GitActionSafety.SAFE:
            return {
                "result": {
                    "safety": preparation["safety"],
                    "executed": False,
                    "reason": preparation["reason"],
                }
            }

        ws_path = self._get_workspace_path(workspace_id)

        # Prefer git switch -c, fallback to checkout -b
        result = self._run_git(ws_path, ["switch", "-c", branch_name])
        if result.returncode != 0:
            # Fallback to checkout -b
            result = self._run_git(ws_path, ["checkout", "-b", branch_name])

        if result.returncode != 0:
            return {
                "result": {
                    "safety": GitActionSafety.BLOCKED,
                    "executed": False,
                    "reason": result.stderr.strip() or "Failed to create branch",
                }
            }

        self._add_timeline_event(
            code_session_id,
            "git.branch.created",
            message=f"Branch '{branch_name}' created for workspace {workspace_id}",
            payload={"workspace_id": workspace_id, "branch": branch_name},
        )

        return {
            "result": {
                "safety": GitActionSafety.SAFE,
                "executed": True,
                "branch": branch_name,
                "reason": "",
            }
        }

    def prepare_commit(
        self, workspace_id: str, message: str, code_session_id: Optional[str] = None
    ) -> dict:
        ws_path = self._get_workspace_path(workspace_id)

        if not self._is_git_repo(ws_path):
            return {
                "action": "commit",
                "safety": GitActionSafety.BLOCKED,
                "reason": "Workspace is not a Git repository",
            }

        status = self.get_status(workspace_id)
        diff_result = self.get_diff(workspace_id)

        self._add_timeline_event(
            code_session_id,
            "git.commit.prepared",
            message=f"Commit prepared for workspace {workspace_id}",
            payload={
                "workspace_id": workspace_id,
                "branch": status.get("branch"),
                "message": message,
            },
        )

        return {
            "action": "commit",
            "safety": GitActionSafety.NEEDS_APPROVAL,
            "reason": "Commit requires human approval",
            "workspace_id": workspace_id,
            "branch": status.get("branch"),
            "files": status.get("files", []),
            "diff_stat": f"+{diff_result['additions']}/-{diff_result['deletions']}",
            "message": message,
            "executed": False,
        }
