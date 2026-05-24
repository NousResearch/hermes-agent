"""Git worktree provisioning and kanban worker workspace binding."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def git_repo_root(start: Path) -> Optional[Path]:
    """Return the git toplevel for ``start``, or ``None``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(start),
        )
        if result.returncode == 0:
            top = result.stdout.strip()
            if top:
                return Path(top).resolve()
    except Exception:
        pass
    return None


def _repo_root_candidates(workspace: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    _add(workspace)
    if workspace.parent != workspace:
        _add(workspace.parent)
    parts = workspace.parts
    for idx, part in enumerate(parts):
        if part == ".worktrees" and idx > 0:
            _add(Path(*parts[:idx]))
            break
    try:
        from hermes_cli.config import get_hermes_home

        _add(get_hermes_home() / "hermes-agent")
    except Exception:
        pass
    _add(Path.cwd())
    return candidates


def infer_repo_root_for_worktree(workspace: Path) -> Optional[Path]:
    for candidate in _repo_root_candidates(workspace):
        root = git_repo_root(candidate)
        if root is not None:
            return root
    return None


def default_worktree_path(task_id: str, *, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or infer_repo_root_for_worktree(Path.cwd())
    if root is not None:
        return (root / ".worktrees" / task_id).resolve()
    return (Path.cwd() / ".worktrees" / task_id).resolve()


def _is_usable_worktree(path: Path) -> bool:
    if not path.is_dir():
        return False
    git_entry = path / ".git"
    return git_entry.exists()


def _copy_worktreeinclude(repo_root: Path, wt_path: Path) -> None:
    include_file = repo_root / ".worktreeinclude"
    if not include_file.is_file():
        return
    repo_root_resolved = repo_root.resolve()
    wt_path_resolved = wt_path.resolve()
    try:
        for line in include_file.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = repo_root / entry
            dst = wt_path / entry
            try:
                src_resolved = src.resolve(strict=False)
                dst_resolved = dst.resolve(strict=False)
            except (OSError, ValueError):
                continue
            if not _path_is_within_root(src_resolved, repo_root_resolved):
                continue
            if not _path_is_within_root(dst_resolved, wt_path_resolved):
                continue
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            elif src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(src_resolved), str(dst))
                except (OSError, NotImplementedError):
                    if sys.platform == "win32":
                        shutil.copytree(
                            str(src_resolved),
                            str(dst),
                            symlinks=True,
                            dirs_exist_ok=False,
                        )
    except Exception as exc:
        logger.debug("worktreeinclude copy failed: %s", exc)


def ensure_worktree_workspace(
    task: Any,
    workspace: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """Create the git worktree for a kanban task when missing.

    Raises ``ValueError`` when no git repository can be resolved.
    """
    wt_path = Path(workspace).expanduser().resolve()
    if _is_usable_worktree(wt_path):
        return wt_path

    root = repo_root or infer_repo_root_for_worktree(wt_path)
    if root is None:
        raise ValueError(
            f"cannot find git repository for worktree {wt_path}; "
            "set an absolute workspace_path under the repo's .worktrees/ directory"
        )

    branch = (getattr(task, "branch_name", None) or "").strip() or f"wt/{task.id}"
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    gitignore = root / ".gitignore"
    ignore_entry = ".worktrees/"
    try:
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if ignore_entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write(f"{ignore_entry}\n")
    except Exception:
        pass

    def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(root),
        )

    add_new = _run(
        ["git", "worktree", "add", str(wt_path), "-b", branch, "HEAD"]
    )
    if add_new.returncode != 0:
        add_existing = _run(["git", "worktree", "add", str(wt_path), branch])
        if add_existing.returncode != 0:
            detail = (add_new.stderr or add_new.stdout or add_existing.stderr or "").strip()
            raise ValueError(
                f"git worktree add failed for {wt_path}: {detail or 'unknown error'}"
            )

    _copy_worktreeinclude(root, wt_path)
    if not _is_usable_worktree(wt_path):
        raise ValueError(f"worktree path {wt_path} was created but is not usable")
    return wt_path


def apply_kanban_worker_workspace(agent: Any = None) -> Optional[str]:
    """Bind the current process (and optional agent) to ``HERMES_KANBAN_WORKSPACE``."""
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip()
    if not workspace or not os.path.isdir(workspace):
        return None
    try:
        os.chdir(workspace)
    except OSError as exc:
        logger.warning("kanban worker could not chdir to %s: %s", workspace, exc)
    if agent is not None:
        agent.session_cwd = workspace
    os.environ["TERMINAL_CWD"] = workspace
    os.environ["HERMES_CURSOR_AUX_CWD"] = workspace
    return workspace
