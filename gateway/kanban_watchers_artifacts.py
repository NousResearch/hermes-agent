"""Durable task attachment confinement for the kanban notifier."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def confined_artifact_paths(candidates, task, board):
    """Reject unstaged files and symlinked durable roots before adapter delivery."""
    allowed_roots: list[Path] = []
    task_id = getattr(task, "id", None) if task is not None else None
    if task_id:
        try:
            from hermes_cli import kanban_db as _kb
            from hermes_cli.kanban_db_artifacts import anchor_to_configured_root

            declared_root = _kb.task_attachments_dir(str(task_id), board=board)
            # Same boundary as staging: only the configured Kanban root may be
            # an alias. The durable root is an authority boundary, not just a
            # containment hint, so any symlinked component below it is refused
            # instead of being resolved into an attacker-selected directory.
            lexical_root = anchor_to_configured_root(declared_root)
            resolved_root = lexical_root.resolve(strict=True)
            if lexical_root == resolved_root and resolved_root.is_dir():
                allowed_roots.append(resolved_root)
            else:
                logger.warning(
                    "kanban notifier: refusing unsafe task attachment root: %s",
                    declared_root,
                )
        except (OSError, RuntimeError, ValueError):
            pass

    confined: list[str] = []
    for candidate in candidates:
        try:
            resolved = Path(candidate).resolve(strict=True)
            if any(resolved.is_relative_to(root) for root in allowed_roots):
                confined.append(str(resolved))
            else:
                logger.warning(
                    "kanban notifier: skipping unstaged task artifact: %s",
                    candidate,
                )
        except (OSError, RuntimeError):
            continue
    candidates = confined
    if not candidates:
        return []

    return candidates
