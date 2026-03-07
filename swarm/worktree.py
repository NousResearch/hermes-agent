from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Optional


class WorktreeError(Exception):
    pass


class WorktreeManager:
    def __init__(self, repo_root: Path, worktree_base: Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.worktree_base = (
            Path(worktree_base).resolve()
            if worktree_base
            else self.repo_root / ".swarm-worktrees"
        )
        self._lock = threading.Lock()

    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorktreeError(
                f"git {' '.join(args)} failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        return result

    def _branch_name(self, agent_id: str) -> str:
        return f"swarm/{agent_id}"

    def _worktree_path(self, agent_id: str) -> Path:
        return self.worktree_base / agent_id

    def create(self, agent_id: str, base_branch: str = "main") -> Path:
        with self._lock:
            wt_path = self._worktree_path(agent_id)
            branch = self._branch_name(agent_id)
            wt_path.parent.mkdir(parents=True, exist_ok=True)
            self._git("worktree", "add", "-b", branch, str(wt_path), base_branch)
            return wt_path

    def get_path(self, agent_id: str) -> Optional[Path]:
        wt_path = self._worktree_path(agent_id)
        return wt_path if wt_path.exists() else None

    def list_active(self) -> list[dict]:
        with self._lock:
            result = self._git("worktree", "list", "--porcelain")
        entries: list[dict] = []
        current: dict = {}
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current = {"path": line.split(" ", 1)[1]}
            elif line.startswith("branch "):
                ref = line.split(" ", 1)[1]
                current["branch"] = ref
                # extract agent_id from refs/heads/swarm/<id>
                prefix = "refs/heads/swarm/"
                if ref.startswith(prefix):
                    current["agent_id"] = ref[len(prefix):]
            elif line == "":
                if "agent_id" in current:
                    entries.append(current)
                current = {}
        if "agent_id" in current:
            entries.append(current)
        return entries

    def has_changes(self, agent_id: str) -> bool:
        wt_path = self._worktree_path(agent_id)
        if not wt_path.exists():
            raise WorktreeError(f"No worktree for agent {agent_id}")
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def commit(self, agent_id: str, message: str) -> str:
        wt_path = self._worktree_path(agent_id)
        if not wt_path.exists():
            raise WorktreeError(f"No worktree for agent {agent_id}")
        with self._lock:
            self._git("add", "-A", cwd=wt_path)
            self._git("commit", "-m", message, cwd=wt_path)
            result = self._git("rev-parse", "HEAD", cwd=wt_path)
        return result.stdout.strip()

    def get_diff(self, agent_id: str) -> str:
        wt_path = self._worktree_path(agent_id)
        if not wt_path.exists():
            raise WorktreeError(f"No worktree for agent {agent_id}")
        # staged + unstaged
        staged = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        unstaged = subprocess.run(
            ["git", "diff"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        return (staged.stdout + unstaged.stdout).strip()

    def merge_to(
        self, agent_id: str, target_branch: str = "main", strategy: str = "merge"
    ) -> bool:
        branch = self._branch_name(agent_id)
        with self._lock:
            # save current branch
            orig = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            try:
                self._git("checkout", target_branch)
                if strategy == "squash":
                    self._git("merge", "--squash", branch)
                    self._git("commit", "-m", f"squash merge {branch}")
                elif strategy == "rebase":
                    self._git("rebase", branch)
                else:
                    self._git("merge", "--no-ff", branch)
                return True
            except WorktreeError:
                # abort merge on conflict
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=self.repo_root,
                    capture_output=True,
                )
                return False
            finally:
                subprocess.run(
                    ["git", "checkout", orig],
                    cwd=self.repo_root,
                    capture_output=True,
                )

    def cleanup(self, agent_id: str) -> None:
        wt_path = self._worktree_path(agent_id)
        branch = self._branch_name(agent_id)
        with self._lock:
            if wt_path.exists():
                self._git("worktree", "remove", "--force", str(wt_path))
            # delete the branch (ignore errors if already gone)
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=self.repo_root,
                capture_output=True,
            )

    def cleanup_all(self) -> None:
        for entry in self.list_active():
            self.cleanup(entry["agent_id"])
        # prune stale worktree metadata
        with self._lock:
            self._git("worktree", "prune")
