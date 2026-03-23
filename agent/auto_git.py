"""Automatic git commit helpers.

Provides lightweight git operations for auto-committing file changes made by
the agent. All functions are safe to call from non-git directories (they
return False / empty string gracefully).
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _run_git(args: List[str], cwd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def is_git_repo(cwd: str) -> bool:
    """Check if *cwd* is inside a git repository."""
    try:
        result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def get_git_diff(cwd: str) -> str:
    """Return combined staged + unstaged diff for the repo at *cwd*."""
    try:
        # Unstaged changes
        unstaged = _run_git(["diff"], cwd)
        # Staged changes
        staged = _run_git(["diff", "--cached"], cwd)
        parts = []
        if staged.stdout.strip():
            parts.append(staged.stdout.strip())
        if unstaged.stdout.strip():
            parts.append(unstaged.stdout.strip())
        return "\n".join(parts)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def get_changed_files(cwd: str) -> List[str]:
    """Return list of changed (staged + unstaged + untracked) file paths."""
    try:
        result = _run_git(["status", "--porcelain"], cwd)
        if result.returncode != 0:
            return []
        files = []
        for line in result.stdout.splitlines():
            if len(line) > 3:
                files.append(line[3:].strip())
        return files
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def generate_commit_message(diff: str, changed_files: Optional[List[str]] = None) -> str:
    """Generate a concise commit message from a diff without using an LLM.

    Parses filenames and classifies changes as Add/Edit/Delete.
    """
    if not diff and not changed_files:
        return "Auto-commit: update files"

    # Parse filenames from diff headers
    added = set()
    modified = set()
    deleted = set()

    for line in diff.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
        if m:
            modified.add(m.group(2))
            continue
        if line.startswith("new file mode"):
            # The last file added to modified is actually new
            if modified:
                last = list(modified)[-1]
                modified.discard(last)
                added.add(last)
        elif line.startswith("deleted file mode"):
            if modified:
                last = list(modified)[-1]
                modified.discard(last)
                deleted.add(last)

    # Also use changed_files list if provided
    if changed_files and not (added or modified or deleted):
        modified.update(changed_files)

    parts = []
    if added:
        names = ", ".join(sorted(added)[:5])
        if len(added) > 5:
            names += f" (+{len(added) - 5} more)"
        parts.append(f"Add {names}")
    if modified:
        names = ", ".join(sorted(modified)[:5])
        if len(modified) > 5:
            names += f" (+{len(modified) - 5} more)"
        parts.append(f"Edit {names}")
    if deleted:
        names = ", ".join(sorted(deleted)[:5])
        if len(deleted) > 5:
            names += f" (+{len(deleted) - 5} more)"
        parts.append(f"Delete {names}")

    if not parts:
        return "Auto-commit: update files"

    msg = "; ".join(parts)
    # Truncate to reasonable commit message length
    if len(msg) > 120:
        msg = msg[:117] + "..."
    return msg


def auto_commit(cwd: str, message: Optional[str] = None) -> bool:
    """Stage all changes and commit with the given (or generated) message.

    Returns True if a commit was made, False otherwise.
    """
    if not is_git_repo(cwd):
        return False

    try:
        # Check if there are any changes to commit
        status = _run_git(["status", "--porcelain"], cwd)
        if not status.stdout.strip():
            return False  # Nothing to commit

        changed_files = get_changed_files(cwd)

        # Generate message if not provided
        if not message:
            diff = get_git_diff(cwd)
            message = generate_commit_message(diff, changed_files)

        # Stage all changes
        result = _run_git(["add", "-A"], cwd)
        if result.returncode != 0:
            logger.warning("git add failed: %s", result.stderr)
            return False

        # Commit
        result = _run_git(["commit", "-m", message, "--no-verify"], cwd)
        if result.returncode != 0:
            logger.warning("git commit failed: %s", result.stderr)
            return False

        logger.info("Auto-committed: %s", message)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Auto-commit failed: %s", e)
        return False


def undo_last_commit(cwd: str) -> bool:
    """Undo the last commit with ``git reset --soft HEAD~1``.

    Returns True on success, False otherwise.
    """
    if not is_git_repo(cwd):
        return False

    try:
        result = _run_git(["reset", "--soft", "HEAD~1"], cwd)
        if result.returncode != 0:
            logger.warning("git reset failed: %s", result.stderr)
            return False
        logger.info("Undid last commit (soft reset)")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Undo commit failed: %s", e)
        return False


def maybe_auto_commit(cwd: str, config: Optional[dict] = None) -> bool:
    """Auto-commit if the ``auto_commit`` config option is enabled.

    Safe to call unconditionally -- returns False if auto-commit is disabled
    or if *cwd* is not a git repo.
    """
    if config is None:
        config = {}

    # Check config: auto_commit must be explicitly enabled
    enabled = config.get("auto_commit", False)
    if not enabled:
        # Also check environment variable as fallback
        enabled = os.getenv("HERMES_AUTO_COMMIT", "").lower() in ("1", "true", "yes")

    if not enabled:
        return False

    return auto_commit(cwd)
