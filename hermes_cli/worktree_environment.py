"""Safe linking of project-local Python environments into Git worktrees."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

_LOG = logging.getLogger(__name__)


def bootstrap_worktree_environments(
    repo_root: Path,
    target: Path,
    *,
    environment_names: Iterable[str] = (".venv", "venv"),
) -> None:
    """Link ignored project environments into a child worktree when absent.

    Environment contents are never copied. A symlinked source is accepted only
    when its resolved target remains inside the project root, and a pre-existing
    destination (including a broken symlink) is left untouched.
    """
    try:
        source_root = repo_root.resolve(strict=True)
        target_root = target.resolve(strict=True)
    except OSError as exc:
        _LOG.warning("worktree environment bootstrap roots unavailable: %s", exc)
        return

    for environment_name in environment_names:
        source = repo_root / environment_name
        destination = target / environment_name
        try:
            if destination.exists() or destination.is_symlink() or not source.exists():
                continue
            resolved_source = source.resolve(strict=True)
            resolved_destination = destination.resolve(strict=False)
        except OSError as exc:
            _LOG.warning(
                "worktree environment bootstrap skipped %s: %s", environment_name, exc
            )
            continue
        try:
            source_is_project_local = resolved_source.is_relative_to(source_root)
            destination_is_worktree_local = resolved_destination.is_relative_to(target_root)
        except ValueError:
            source_is_project_local = False
            destination_is_worktree_local = False
        if not source_is_project_local or not destination_is_worktree_local:
            _LOG.warning("worktree environment bootstrap refused unsafe %s", environment_name)
            continue
        if not resolved_source.is_dir():
            continue
        try:
            os.symlink(str(resolved_source), str(destination), target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            _LOG.warning(
                "worktree environment bootstrap could not link %s: %s",
                environment_name,
                exc,
            )
