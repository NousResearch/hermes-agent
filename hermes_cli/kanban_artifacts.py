"""Scratch-workspace artifact preservation, split out of ``kanban_db`` (a facade the
file-length ratchet may not grow). ``persist_scratch_completion_artifacts`` copies
declared artifacts out of a managed scratch workspace before teardown deletes
them; it reaches the facade's helpers lazily so importing it creates no cycle.
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Optional


def persist_scratch_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict,
) -> None:
    """Copy scratch-workspace completion artifacts before cleanup removes them."""
    from hermes_cli import kanban_db as kb
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return

    workspace = kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return
    is_managed, board = kb._managed_scratch_path_info(workspace)
    if not is_managed:
        return

    try:
        workspace_root = workspace.resolve()
    except OSError:
        return

    attachment_dir = kb.task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        kb._discard_staged_copies(used_destinations, attachment_dir)

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        try:
            resolved_src = src.resolve()
        except OSError:
            persisted.append(artifact)
            continue

        if not resolved_src.is_relative_to(workspace_root):
            persisted.append(artifact)
            continue

        problem = None
        if not src.is_file():
            problem = f"declared scratch artifact is unavailable or not a regular file: {artifact}"
        elif resolved_src.stat().st_size > kb.KANBAN_ATTACHMENT_MAX_BYTES:
            problem = (
                f"declared scratch artifact exceeds the "
                f"{kb.KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
            )
        if problem:
            _discard_copies()
            raise kb.ArtifactPreservationError(problem)

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = kb._unique_attachment_path(attachment_dir, resolved_src.name, used_destinations)
            kb._copy_capped(resolved_src, dest, artifact)
        except Exception as exc:
            if dest is not None:
                with contextlib.suppress(OSError):
                    dest.unlink(missing_ok=True)
            _discard_copies()
            if isinstance(exc, kb.ArtifactPreservationError):
                raise
            raise kb.ArtifactPreservationError(
                f"could not preserve declared scratch artifact {artifact}: {exc}"
            ) from exc
        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [
            path for path in persisted if path.startswith(str(attachment_dir.resolve()))
        ]
