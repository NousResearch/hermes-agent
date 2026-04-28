#!/usr/bin/env python3
"""
WorktreeService — Git worktree and checkpoint foundation for Hermes Code Mode.

Provides:
  - Git capability detection (is repo? supports worktree? dirty state?)
  - Task-branch preparation (creates branch, optionally worktree)
  - Checkpoint metadata (records branch state at a point in time)
  - Graceful degradation when outside a Git repo

Does NOT perform destructive Git operations (reset --hard, clean -fdx, etc.)
without explicit approval.
"""

import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(path: Path, args: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(path),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class GitCapabilities:
    """Snapshot of detected Git capabilities for a path."""

    def __init__(
        self,
        path: Path,
        is_git_repo: bool,
        supports_worktree: bool,
        current_branch: Optional[str],
        is_dirty: bool,
        has_commits: bool,
        toplevel: Optional[Path],
    ):
        self.path = path
        self.is_git_repo = is_git_repo
        self.supports_worktree = supports_worktree
        self.current_branch = current_branch
        self.is_dirty = is_dirty
        self.has_commits = has_commits
        self.toplevel = toplevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "is_git_repo": self.is_git_repo,
            "supports_worktree": self.supports_worktree,
            "current_branch": self.current_branch,
            "is_dirty": self.is_dirty,
            "has_commits": self.has_commits,
            "toplevel": str(self.toplevel) if self.toplevel else None,
        }


def detect_git_capabilities(path: Path) -> GitCapabilities:
    """Detect Git capabilities for *path*. Never raises — degrades gracefully."""
    if not path.exists():
        return GitCapabilities(
            path=path, is_git_repo=False, supports_worktree=False,
            current_branch=None, is_dirty=False, has_commits=False, toplevel=None,
        )

    # is_git_repo
    try:
        r = _run_git(path, ["rev-parse", "--is-inside-work-tree"])
        is_git_repo = r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        is_git_repo = False

    if not is_git_repo:
        return GitCapabilities(
            path=path, is_git_repo=False, supports_worktree=False,
            current_branch=None, is_dirty=False, has_commits=False, toplevel=None,
        )

    # toplevel
    try:
        r = _run_git(path, ["rev-parse", "--show-toplevel"])
        toplevel = Path(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        toplevel = None

    # current_branch
    try:
        r = _run_git(path, ["branch", "--show-current"])
        current_branch = r.stdout.strip() or None
    except Exception:
        current_branch = None

    # has_commits — empty repos have no HEAD
    try:
        r = _run_git(path, ["rev-parse", "--verify", "HEAD"])
        has_commits = r.returncode == 0
    except Exception:
        has_commits = False

    # is_dirty — only meaningful if there are commits
    is_dirty = False
    if has_commits:
        try:
            r = _run_git(path, ["status", "--porcelain"])
            is_dirty = r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            pass

    # supports_worktree — git worktree list exits 0 on git >= 2.5
    try:
        r = _run_git(path, ["worktree", "list", "--porcelain"])
        supports_worktree = r.returncode == 0
    except Exception:
        supports_worktree = False

    return GitCapabilities(
        path=path,
        is_git_repo=is_git_repo,
        supports_worktree=supports_worktree,
        current_branch=current_branch,
        is_dirty=is_dirty,
        has_commits=has_commits,
        toplevel=toplevel,
    )


class WorktreeService:
    """Worktree and checkpoint service for Hermes Code Mode.

    All destructive Git operations require explicit approval and are
    never performed automatically.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _git_service(self):
        from hermes_cli.code.git_service import GitService
        return GitService(db_path=self._db_path, realtime_hub=self._realtime_hub)

    def _workspace_db(self):
        from hermes_state import WorkspaceDB
        return WorkspaceDB(db_path=self._db_path)

    def _get_workspace_path(self, workspace_id: str) -> Path:
        wdb = self._workspace_db()
        try:
            workspace = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()
        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")
        return Path(workspace["path"])

    def detect_capabilities(self, workspace_id: str) -> Dict[str, Any]:
        """Return GitCapabilities dict for workspace. Never raises."""
        try:
            path = self._get_workspace_path(workspace_id)
            caps = detect_git_capabilities(path)
        except Exception as exc:
            logger.warning("detect_capabilities failed for %s: %s", workspace_id, exc)
            caps = GitCapabilities(
                path=Path("."), is_git_repo=False, supports_worktree=False,
                current_branch=None, is_dirty=False, has_commits=False, toplevel=None,
            )
        return caps.to_dict()

    def prepare_task_branch(
        self,
        workspace_id: str,
        branch_name: str,
        use_worktree: bool = False,
        worktree_base_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Prepare a task branch.

        If *use_worktree* is True and worktrees are supported, creates a git
        worktree instead of switching the current checkout. This is safe:
        the main workspace is undisturbed.

        Never performs destructive operations (no reset --hard, no force-push).
        Raises ValueError if the branch name is invalid or Git is unavailable.
        """
        path = self._get_workspace_path(workspace_id)
        caps = detect_git_capabilities(path)

        if not caps.is_git_repo:
            raise ValueError(f"Workspace {workspace_id} is not a Git repository")
        if not caps.has_commits:
            raise ValueError("Cannot create a task branch in a repository with no commits")

        # Delegate branch creation to GitService (already has validation)
        git_svc = self._git_service()

        if use_worktree and caps.supports_worktree:
            return self._create_worktree(
                workspace_id=workspace_id,
                path=path,
                branch_name=branch_name,
                worktree_base_dir=worktree_base_dir,
            )
        else:
            result = git_svc.prepare_branch(workspace_id=workspace_id, branch_name=branch_name)
            result["worktree_path"] = None
            return result

    def _create_worktree(
        self,
        workspace_id: str,
        path: Path,
        branch_name: str,
        worktree_base_dir: Optional[Path],
    ) -> Dict[str, Any]:
        # Validate branch name
        from hermes_cli.code.git_service import GitService
        valid, reason = GitService._validate_branch_name(branch_name)
        if not valid:
            raise ValueError(f"Invalid branch name: {reason}")

        # Choose worktree location
        if worktree_base_dir is None:
            toplevel = path
            r = _run_git(path, ["rev-parse", "--show-toplevel"])
            if r.returncode == 0:
                toplevel = Path(r.stdout.strip())
            worktree_base_dir = toplevel.parent / ".hermes-worktrees"

        worktree_base_dir.mkdir(parents=True, exist_ok=True)
        safe_name = branch_name.replace("/", "_").replace(" ", "_")
        worktree_path = worktree_base_dir / f"{safe_name}-{uuid.uuid4().hex[:6]}"

        # Create branch if it doesn't exist
        r = _run_git(path, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"])
        branch_exists = r.returncode == 0

        if branch_exists:
            wt_args = ["worktree", "add", str(worktree_path), branch_name]
        else:
            wt_args = ["worktree", "add", "-b", branch_name, str(worktree_path)]

        r = _run_git(path, wt_args, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(
                f"git worktree add failed: {r.stderr.strip() or r.stdout.strip()}"
            )

        return {
            "workspace_id": workspace_id,
            "branch": branch_name,
            "worktree_path": str(worktree_path),
            "worktree": True,
            "created_at": _utc_now(),
        }

    def create_checkpoint(
        self,
        workspace_id: str,
        label: Optional[str] = None,
        code_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a checkpoint (snapshot of current Git state metadata).

        Does NOT modify any files or create commits. Pure metadata capture.
        """
        path = self._get_workspace_path(workspace_id)
        caps = detect_git_capabilities(path)

        checkpoint: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "workspace_id": workspace_id,
            "code_session_id": code_session_id,
            "label": label or "checkpoint",
            "created_at": _utc_now(),
            "git": caps.to_dict(),
        }

        # Capture current HEAD commit SHA
        if caps.is_git_repo and caps.has_commits:
            try:
                r = _run_git(path, ["rev-parse", "HEAD"])
                if r.returncode == 0:
                    checkpoint["head_sha"] = r.stdout.strip()
            except Exception:
                pass

        return checkpoint

    def list_worktrees(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List existing worktrees for workspace. Returns empty list if not a Git repo."""
        try:
            path = self._get_workspace_path(workspace_id)
        except ValueError:
            return []

        caps = detect_git_capabilities(path)
        if not caps.is_git_repo or not caps.supports_worktree:
            return []

        try:
            r = _run_git(path, ["worktree", "list", "--porcelain"])
            if r.returncode != 0:
                return []
            return _parse_worktree_list(r.stdout)
        except Exception as exc:
            logger.warning("list_worktrees failed: %s", exc)
            return []


def _parse_worktree_list(output: str) -> List[Dict[str, Any]]:
    """Parse `git worktree list --porcelain` output into a list of dicts."""
    worktrees = []
    current: Dict[str, Any] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree "):]
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
        elif line == "bare":
            current["bare"] = True
        elif line == "detached":
            current["detached"] = True
    if current:
        worktrees.append(current)
    return worktrees
