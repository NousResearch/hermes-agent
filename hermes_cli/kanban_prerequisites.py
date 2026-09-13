"""Shared prerequisite validation for Kanban task execution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def validate_worktree_anchor(
    workspace_kind: str,
    workspace_path: Optional[str],
    *,
    project_repo: Optional[str] = None,
) -> None:
    """Require every worktree task to carry an absolute repository anchor."""
    if workspace_kind != "worktree":
        return
    anchor = workspace_path or project_repo
    if not anchor:
        raise ValueError(
            "workspace_kind=worktree requires an explicit absolute repository path, "
            "a project-linked primary repository, or a board default_workdir"
        )
    if not Path(anchor).expanduser().is_absolute():
        raise ValueError(
            f"workspace_kind=worktree repository anchor {anchor!r} is not absolute"
        )


def validate_forced_skills(
    assignee: Optional[str], skills: Optional[Iterable[str]]
) -> None:
    """Resolve every forced skill against its assignee without reading secrets."""
    requested = list(skills or ())
    if not requested:
        return
    if not assignee:
        raise ValueError("forced skills require an assignee")

    from hermes_cli.profiles import (
        get_profile_dir,
        profile_exists,
        validate_profile_name,
    )

    try:
        validate_profile_name(assignee)
        assignee_exists = profile_exists(assignee)
    except ValueError:
        assignee_exists = False
    if not assignee_exists:
        missing = requested
    else:
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.skills_tool import skill_is_loadable

        token = set_hermes_home_override(get_profile_dir(assignee))
        try:
            missing = [
                name
                for name in requested
                if not skill_is_loadable(name, profile_only=True)
            ]
        finally:
            reset_hermes_home_override(token)
    if missing:
        raise ValueError(
            f"forced skill(s) unavailable to assignee {assignee!r}: {', '.join(sorted(missing))}"
        )


def validate_task(task, *, board: Optional[str] = None) -> None:
    """Validate the stored prerequisites of a task before it becomes runnable."""
    workspace_path = task.workspace_path
    if task.workspace_kind == "worktree" and not workspace_path:
        from hermes_cli.kanban_db import read_board_metadata

        workspace_path = read_board_metadata(board)["default_workdir"]
    validate_worktree_anchor(task.workspace_kind, workspace_path)
    validate_forced_skills(task.assignee, task.skills)
