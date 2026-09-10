"""Explicit local Git operations used by the stack lifecycle."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .stack import _branch


@dataclass(frozen=True, slots=True)
class GitEvidence:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class GitStackError(RuntimeError):
    pass


class GitStackRunner:
    def __init__(self, repository: Path, *, environment: Mapping[str, str] | None = None) -> None:
        self.repository = Path(repository)
        self._environment = None if environment is None else dict(environment)

    def _run(self, *args: str) -> GitEvidence:
        argv = ("git", "-C", str(self.repository), *args)
        result = subprocess.run(
            argv, check=False, capture_output=True, text=True, env=self._environment
        )
        evidence = GitEvidence(argv, result.returncode, result.stdout, result.stderr)
        if result.returncode:
            raise GitStackError(result.stderr.strip() or "git command failed")
        return evidence

    def branch_head(self, branch: str) -> str:
        _branch(branch, "branch")
        return self._run("rev-parse", f"refs/remotes/origin/{branch}").stdout.strip()

    def merge_base_into_branch(self, branch: str, base_branch: str) -> GitEvidence:
        _branch(branch, "branch")
        _branch(base_branch, "base_branch")
        self._run("fetch", "origin", base_branch, branch)
        self._run("switch", branch)
        return self._run("merge", "--no-edit", "--no-ff", f"origin/{base_branch}")

    def push_branch(self, branch: str) -> GitEvidence:
        _branch(branch, "branch")
        return self._run(
            "push",
            "origin",
            f"HEAD:refs/heads/{branch}",
        )
