"""Workspace isolation helpers for AutoResearch candidate evaluation."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from autoresearch.models import WorkspaceInfo


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def git_repo_root(start: Path) -> Optional[Path]:
    """Return the enclosing git repo root, if any."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _ensure_gitignore_entry(repo_root: Path, entry: str) -> None:
    gitignore = repo_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    lines = existing.splitlines()
    if entry in lines:
        return
    with open(gitignore, "a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(f"{entry}\n")


def _copy_worktreeinclude_entries(repo_root: Path, workspace_path: Path) -> None:
    include_file = repo_root / ".worktreeinclude"
    if not include_file.exists():
        return

    repo_root_resolved = repo_root.resolve()
    workspace_resolved = workspace_path.resolve()
    for raw_line in include_file.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        src = repo_root / entry
        dst = workspace_path / entry
        try:
            src_resolved = src.resolve(strict=False)
            dst_resolved = dst.resolve(strict=False)
        except (OSError, ValueError):
            continue
        if not _path_is_within_root(src_resolved, repo_root_resolved):
            continue
        if not _path_is_within_root(dst_resolved, workspace_resolved):
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        elif src.is_dir() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src_resolved), str(dst))


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        snapshot[rel] = str(path.stat().st_mtime_ns)
    return snapshot


def create_candidate_workspace(
    project_root: Path,
    run_id: str,
    candidate_id: str,
    editable_files: list[str],
) -> WorkspaceInfo:
    """Create an isolated workspace for one candidate."""
    workspace_path = (
        project_root
        / ".hermes"
        / "autoresearch"
        / "workspaces"
        / run_id
        / candidate_id
    )
    workspace_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = git_repo_root(project_root)
    method = "copy"
    branch = None
    snapshot: dict[str, str] = {}

    if repo_root is not None:
        _ensure_gitignore_entry(repo_root, ".hermes/autoresearch/")
        branch = f"hermes/autoresearch-{uuid.uuid4().hex[:10]}"
        result = subprocess.run(
            ["git", "worktree", "add", str(workspace_path), "-b", branch, "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            method = "git_worktree"
            _copy_worktreeinclude_entries(repo_root, workspace_path)

    if method != "git_worktree":
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        shutil.copytree(
            project_root,
            workspace_path,
            ignore=shutil.ignore_patterns(
                ".git",
                ".hermes",
                ".worktrees",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
            ),
        )
        snapshot = _snapshot_tree(workspace_path)

    base_contents = {}
    for relpath in editable_files:
        target = workspace_path / relpath
        if target.exists():
            base_contents[relpath] = target.read_text(encoding="utf-8")
        else:
            base_contents[relpath] = None

    return WorkspaceInfo(
        path=workspace_path,
        method=method,
        source_root=project_root,
        repo_root=repo_root,
        branch=branch,
        editable_base_contents=base_contents,
        snapshot=snapshot,
    )


def list_changed_files(workspace: WorkspaceInfo) -> list[str]:
    """Return changed relative paths inside a candidate workspace."""
    if workspace.method == "git_worktree":
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace.path),
            capture_output=True,
            text=True,
            timeout=15,
        )
        changed: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if path:
                changed.append(path.replace("\\", "/"))
        return sorted(set(changed))

    current = _snapshot_tree(workspace.path)
    changed = set(workspace.snapshot).symmetric_difference(current)
    for relpath, stamp in workspace.snapshot.items():
        if relpath in current and current[relpath] != stamp:
            changed.add(relpath)
    return sorted(changed)

