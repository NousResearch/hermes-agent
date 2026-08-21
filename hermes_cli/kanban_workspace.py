"""User-facing descriptions of Kanban workspace normalization."""

from __future__ import annotations

from typing import Optional, Protocol


class _TaskWorkspace(Protocol):
    workspace_kind: str
    workspace_path: Optional[str]
    project_id: Optional[str]


def workspace_spec(kind: str, path: Optional[str] = None) -> str:
    """Return the CLI-style representation of a workspace selection."""
    return f"{kind}:{path}" if path else kind


def supersession_warning(
    requested_workspace: Optional[str], task: _TaskWorkspace
) -> Optional[str]:
    """Describe an explicit workspace replaced by project normalization.

    ``None`` means the caller omitted the workspace and intentionally receives
    the project convention silently.  A project link is required so ordinary
    normalization does not produce a misleading warning.
    """
    if requested_workspace != "scratch" or not task.project_id:
        return None
    selected = workspace_spec(task.workspace_kind, task.workspace_path)
    if task.workspace_kind != "worktree":
        return None
    return (
        f"requested workspace {requested_workspace!r} was superseded by "
        f"project-linked workspace {selected!r}"
    )