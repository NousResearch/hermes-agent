"""Git subprocess helpers used by hermes_parity.

All git calls belong in this module so the higher-level command code can stay
testable and so repo-mutating operations have a single audit point.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class GitError(RuntimeError):
    """Raised when a git command fails."""


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> GitResult:
    cmd = ["git", "-C", str(repo), *args]
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result = GitResult(tuple(args), proc.returncode, proc.stdout, proc.stderr)
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed with exit {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return result


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    proc = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(f"not inside a git repository: {proc.stderr.strip()}")
    return Path(proc.stdout.strip()).resolve()


def require_remotes(repo: Path, names: Iterable[str] = ("fork", "origin")) -> None:
    existing = {
        line.split("\t", 1)[0]
        for line in run_git(repo, ["remote", "-v"]).stdout.splitlines()
        if line
    }
    missing = [name for name in names if name not in existing]
    if missing:
        raise GitError(
            "missing required git remote(s): "
            + ", ".join(missing)
            + ". hermes_parity expects remotes named 'fork' and 'origin'."
        )


def git_version() -> tuple[int, int, int]:
    text = subprocess.run(
        ["git", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).stdout
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def supports_merge_tree_write_tree() -> bool:
    return git_version() >= (2, 38, 0)


def require_git_version(minimum: tuple[int, int, int] = (2, 38, 0)) -> None:
    version = git_version()
    if version < minimum:
        raise GitError(
            "git >= "
            + ".".join(str(part) for part in minimum)
            + " is required; found "
            + ".".join(str(part) for part in version)
        )


def fetch(repo: Path, remote: str) -> None:
    run_git(repo, ["fetch", remote])


def rev_parse(repo: Path, rev: str) -> str:
    return run_git(repo, ["rev-parse", rev]).stdout.strip()


def current_branch(repo: Path) -> str:
    result = run_git(repo, ["branch", "--show-current"], check=False)
    return result.stdout.strip() or "(detached)"


def head_sha(repo: Path, rev: str = "HEAD") -> str:
    return rev_parse(repo, rev)


def tree_sha(repo: Path, rev: str = "HEAD") -> str:
    return rev_parse(repo, f"{rev}^{{tree}}")


def merge_base(repo: Path, left: str, right: str) -> str:
    return run_git(repo, ["merge-base", left, right]).stdout.strip()


def ahead_behind(repo: Path, left: str, right: str) -> tuple[int, int]:
    out = run_git(repo, ["rev-list", "--left-right", "--count", f"{left}...{right}"]).stdout
    ahead, behind = out.strip().split()
    return int(ahead), int(behind)


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    out = run_git(repo, ["diff", "--name-only", f"{base}..{head}"]).stdout
    return [line for line in out.splitlines() if line]


def worktree_changed_files(repo: Path, ref: str) -> list[str]:
    """Files that differ between *ref* and the current working tree
    (staged + unstaged). Unlike ``changed_files(ref, "HEAD")`` this sees a
    staged-but-uncommitted merge, which is exactly the state parity gates
    run in before ``finish`` commits."""
    out = run_git(repo, ["diff", "--name-only", ref]).stdout
    return [line for line in out.splitlines() if line]


def porcelain_status(repo: Path) -> list[tuple[str, str]]:
    out = run_git(repo, ["status", "--porcelain=v1", "-z"], check=True).stdout
    if not out:
        return []
    parts = out.split("\0")
    rows: list[tuple[str, str]] = []
    index = 0
    while index < len(parts) - 1:
        entry = parts[index]
        index += 1
        if not entry:
            continue
        code = entry[:2]
        path = entry[3:]
        if code.strip() in {"R", "C"} and index < len(parts) - 1:
            path = parts[index]
            index += 1
        rows.append((code, path))
    return rows


def is_dirty(repo: Path) -> bool:
    """True when TRACKED files are modified/staged. Untracked files ('??')
    do not count: a live dev checkout virtually always carries untracked
    cruft, and `start` never touches the source checkout's working tree —
    it only creates a new worktree from committed refs."""
    return any(code != "??" for code, _ in porcelain_status(repo))


def merge_no_commit(repo: Path, target: str) -> GitResult:
    return run_git(repo, ["merge", "--no-commit", "--no-ff", target], check=False)


def merge_head_exists(repo: Path) -> bool:
    return (repo / ".git" / "MERGE_HEAD").exists() or run_git(repo, ["rev-parse", "-q", "--verify", "MERGE_HEAD"], check=False).returncode == 0


def has_staged_changes(repo: Path) -> bool:
    return run_git(repo, ["diff", "--cached", "--quiet"], check=False).returncode == 1


def commit(repo: Path, message_file: Path) -> None:
    run_git(repo, ["commit", "-F", str(message_file)])


def push(repo: Path, remote: str, branch: str) -> None:
    run_git(repo, ["push", remote, branch])


def conflict_entries(repo: Path) -> list[tuple[str, str]]:
    return [(code, path) for code, path in porcelain_status(repo) if "U" in code or code == "AA"]


def conflict_marker_lines(repo: Path) -> list[str]:
    result = run_git(
        repo,
        ["grep", "-nE", r"^(<<<<<<<|=======|>>>>>>>)"],
        check=False,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or "git grep failed")
    return [line for line in result.stdout.splitlines() if line]


def merge_tree_prediction(repo: Path, ours: str, theirs: str) -> str | None:
    if not supports_merge_tree_write_tree():
        return None
    result = run_git(repo, ["merge-tree", "--write-tree", ours, theirs], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_worktree_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_worktree(repo: Path, worktree: Path, branch: str, start_point: str) -> None:
    print(f"rollback: git -C {repo} worktree remove --force {worktree}")
    run_git(repo, ["worktree", "add", "-b", branch, str(worktree), start_point])


def remove_worktree(repo: Path, worktree: Path, *, force: bool = False) -> None:
    flag = "--force" if force else None
    print(f"rollback: git -C {repo} worktree add {worktree} <branch-or-sha>")
    args = ["worktree", "remove"]
    if flag:
        args.append(flag)
    args.append(str(worktree))
    run_git(repo, args)


def executable(name: str) -> str | None:
    return shutil.which(name)


def env_with_repo_pythonpath(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo) if not current else f"{repo}{os.pathsep}{current}"
    return env
