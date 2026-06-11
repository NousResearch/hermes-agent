from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import RepoConfig


class RepoManagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Worktree:
    branch_name: str
    path: Path


RunCommand = Callable[[list[str], Path | None], str]
_UNSAFE_GIT_REF_CHARS_RE = re.compile(r"[\s~^:?*\\[\]\x00-\x1f\x7f]")


class RepoManager:
    def __init__(
        self,
        *,
        config: RepoConfig,
        workspace_root: str | Path,
        run_command: RunCommand | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root)
        self._run_command = run_command or self._default_run
        self.timeout_seconds = timeout_seconds

    def prepare_worktree(self, *, linear_identifier: str, summary: str) -> Worktree:
        if not self.config.url:
            raise RepoManagerError("mobile_bug_agent.repo.url is not configured.")

        local_name = safe_repo_local_name(self.config.local_name)
        default_branch = safe_default_branch(self.config.default_branch)
        branch_prefix = safe_branch_prefix(self.config.branch_prefix)
        repo_path = self.workspace_root / "repos" / local_name
        worktrees_root = self.workspace_root / "worktrees"
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        worktrees_root.mkdir(parents=True, exist_ok=True)

        if repo_path.exists():
            if not repo_path.is_dir():
                raise RepoManagerError(f"repo path exists but is not a directory: {repo_path}")
            self._run_command(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "fetch",
                    "origin",
                    default_branch,
                ],
                None,
            )
        else:
            self._run_command(["git", "clone", self.config.url, str(repo_path)], None)

        branch_name = self._branch_name(
            branch_prefix=branch_prefix,
            linear_identifier=linear_identifier,
            summary=summary,
        )
        worktree_path = worktrees_root / branch_name.replace("/", "-")
        if worktree_path.exists():
            if not worktree_path.is_dir():
                raise RepoManagerError(f"worktree path exists but is not a directory: {worktree_path}")
            if not _looks_like_git_worktree(worktree_path):
                raise RepoManagerError(f"worktree path exists but is not a git worktree: {worktree_path}")
            current_branch = self._run_command(
                ["git", "-C", str(worktree_path), "branch", "--show-current"],
                None,
            ).strip()
            if current_branch != branch_name:
                raise RepoManagerError(
                    f"worktree branch mismatch: expected {branch_name}, got {current_branch or 'detached HEAD'}"
                )
            status = self._run_command(
                ["git", "-C", str(worktree_path), "status", "--porcelain"],
                None,
            ).strip()
            if status:
                raise RepoManagerError(
                    "worktree has uncommitted changes; clean or archive the existing Monica "
                    f"worktree before retrying: {worktree_path}"
                )
            return Worktree(branch_name=branch_name, path=worktree_path)

        self._run_command(
            [
                "git",
                "-C",
                str(repo_path),
                "worktree",
                "add",
                "-B",
                branch_name,
                str(worktree_path),
                f"origin/{default_branch}",
            ],
            None,
        )
        return Worktree(branch_name=branch_name, path=worktree_path)

    def _branch_name(self, *, branch_prefix: str, linear_identifier: str, summary: str) -> str:
        ident = _slug(linear_identifier, fallback="slack", lowercase=False)
        summary_slug = _slug(summary, fallback="bug", lowercase=True)
        return f"{branch_prefix}/{ident}-{summary_slug}"

    def _default_run(self, cmd: list[str], cwd: Path | None = None) -> str:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=False,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepoManagerError(
                "\n".join(
                    part
                    for part in [
                        f"command timed out after {self.timeout_seconds}s: {' '.join(cmd)}",
                        f"cwd: {cwd}" if cwd else "",
                    ]
                    if part
                )
            ) from exc
        except FileNotFoundError as exc:
            executable = cmd[0] if cmd else "command"
            raise RepoManagerError(f"executable not found: {executable}") from exc
        if proc.returncode != 0:
            stdout = _tail(proc.stdout)
            stderr = _tail(proc.stderr)
            raise RepoManagerError(
                "\n".join(
                    part
                    for part in [
                        f"command failed ({proc.returncode}): {' '.join(cmd)}",
                        f"cwd: {cwd}" if cwd else "",
                        f"stdout: {stdout}" if stdout else "",
                        f"stderr: {stderr}" if stderr else "",
                    ]
                    if part
                )
            )
        return proc.stdout


def _slug(value: str, *, fallback: str, lowercase: bool) -> str:
    source = value.strip().lower() if lowercase else value.strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", source).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug or fallback)[:80]


def _looks_like_git_worktree(path: Path) -> bool:
    return (path / ".git").exists()


def safe_repo_local_name(value: str) -> str:
    name = value.strip()
    path = Path(name)
    if not name or path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
        raise RepoManagerError("repo.local_name must be a simple directory name.")
    if "chandler" in name.lower():
        raise RepoManagerError("repo.local_name must not point at a Chandler directory.")
    return name


def safe_branch_prefix(value: str) -> str:
    prefix = str(value or "").strip()
    if not is_safe_git_branch_name(prefix):
        raise RepoManagerError("repo.branch_prefix must be a safe git branch prefix.")
    if "chandler" in prefix.lower():
        raise RepoManagerError("repo.branch_prefix must not point at Chandler.")
    return prefix


def safe_default_branch(value: str) -> str:
    branch = str(value or "").strip()
    if not is_safe_git_branch_name(branch):
        raise RepoManagerError("repo.default_branch must be a safe git branch name.")
    return branch


def is_safe_git_branch_name(value: str) -> bool:
    branch = str(value or "").strip()
    if not branch or branch.startswith("/") or branch.endswith("/") or "//" in branch:
        return False
    parts = branch.split("/")
    return not (
        branch == "@"
        or "@{" in branch
        or ".." in branch
        or any(
            not part
            or part in {".", ".."}
            or part.startswith((".", "-"))
            or part.endswith(".")
            or part.endswith(".lock")
            or _UNSAFE_GIT_REF_CHARS_RE.search(part)
            for part in parts
        )
    )


def _tail(value: str, *, limit: int = 2000) -> str:
    return value.strip()[-limit:]
