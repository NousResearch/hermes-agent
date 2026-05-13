"""Deterministic workflow worktree metadata allocation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def allocate_worktrees(normalized_dag: dict[str, Any], *, policy: dict[str, Any], workspace_path: str | Path) -> dict[str, Any]:
    """Return a copy of ``normalized_dag`` with deterministic worktree metadata.

    This function allocates metadata only. It does not run ``git worktree add``
    or touch the filesystem.
    """

    allocated = copy.deepcopy(normalized_dag)
    worktree_policy = policy.get("worktrees") if isinstance(policy.get("worktrees"), dict) else {}
    if worktree_policy.get("enabled") is False:
        return allocated

    workspace = Path(workspace_path).expanduser().resolve()
    root_name = worktree_policy.get("root", ".worktrees")
    if not isinstance(root_name, str) or not root_name.strip():
        raise ValueError("worktrees.root must be a non-empty relative path")
    root_path = Path(root_name)
    if root_path.is_absolute() or ".." in root_path.parts:
        raise ValueError("worktrees.root must stay inside the workspace")
    worktrees_root = (workspace / root_path).resolve()
    if not _is_relative_to(worktrees_root, workspace):
        raise ValueError("worktrees.root must stay inside the workspace")

    branch_prefix = str(worktree_policy.get("branch_prefix") or "workflow").strip().strip("/")
    if not _is_safe_branch_prefix(branch_prefix):
        raise ValueError("worktrees.branch_prefix must be a non-empty safe git branch prefix")
    workflow_id = str(allocated.get("workflow_id") or "workflow")
    allocated_branches: dict[str, str] = {}
    allocated_paths: dict[Path, str] = {}
    for node in allocated.get("nodes", []):
        if node.get("role") != "engineer":
            continue
        node_workspace = node.get("workspace")
        if not isinstance(node_workspace, dict) or node_workspace.get("kind") != "worktree":
            continue
        node_id = node["id"]
        node_workspace["base_ref"] = node_workspace.get("base_ref") or "origin/main"
        node_workspace["branch"] = node_workspace.get("branch") or f"{branch_prefix}/{workflow_id}/{node_id}"
        branch = node_workspace["branch"]
        if branch in allocated_branches:
            raise ValueError(f"duplicate worktree branch allocation: {branch}")
        allocated_branches[branch] = node_id
        worktree_path = Path(node_workspace.get("worktree_path") or worktrees_root / f"{workflow_id}-{node_id}")
        if not worktree_path.is_absolute():
            worktree_path = (workspace / worktree_path).resolve()
        else:
            worktree_path = worktree_path.resolve()
        if not _is_relative_to(worktree_path, worktrees_root):
            raise ValueError("allocated worktree path must stay inside worktrees.root")
        if worktree_path in allocated_paths:
            raise ValueError(f"duplicate worktree path allocation: {worktree_path}")
        allocated_paths[worktree_path] = node_id
        node_workspace["worktree_path"] = str(worktree_path)
    return allocated


def _is_safe_branch_prefix(value: str) -> bool:
    if not value or value.startswith("/") or value.endswith("/") or ".." in value.split("/"):
        return False
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        return False
    return all(part and not part.startswith(".") and not part.endswith(".") for part in value.split("/"))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
