"""Workspace identity guard for session resume / restore boundaries.

Prevents cross-workspace context leakage when resuming sessions: if a stored
session carries a reliable workspace identity (``git_repo_root`` or ``cwd``)
that differs from the current working directory, block or warn at the restore
boundary instead of silently injecting foreign history into the agent's prompt.

Design decisions:
- Workspace identity matching order: ``git_repo_root`` > ``cwd`` (never
  ``git_branch`` — branch changes are normal and should not prevent resume).
- If stored session has a reliable workspace identity and current workspace
  differs → **block restore** with a clear error.
- If stored session has no workspace identity because legacy/null fields →
  **allow but warn**.
- Auto-continue (``hermes -c`` with no explicit ID): filter candidates by
  workspace when possible; skip mismatches; allow legacy as last resort with
  warning.
- Explicit ``--resume SESSION_ID``: strict validation and fail on known mismatch.

This is the primary fix for context-compaction/session-state leakage across
independent agent sessions (e.g. scout vs hermes-agent workspaces).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


@dataclass
class WorkspaceGuardResult:
    """Structured result from workspace identity validation."""

    ok: bool  # True if restore is safe (no mismatch or legacy session)
    warning: str | None = None  # Non-empty string for legacy/warning cases
    reason: str | None = None  # Human-readable explanation
    stored_workspace: str = ""  # Resolved workspace identity from DB row
    current_workspace: str = ""  # Resolved workspace identity of current cwd

    @property
    def blocked(self) -> bool:
        return not self.ok


def _resolve_git_repo_root(cwd: str | None) -> str:
    """Resolve the git repo root for a given working directory.

    Returns empty string if cwd is missing, not a git repo, or probe fails.
    Uses Path.resolve() to normalise paths before probing.
    """
    if not cwd:
        return ""
    try:
        resolved = str(Path(cwd).resolve())
    except (OSError, ValueError):
        return ""

    # Walk up from resolved cwd looking for .git directory or file
    current = Path(resolved)
    while True:
        git_path = current / ".git"
        if git_path.exists():
            # Return the parent of .git (the repo root), not .git itself
            return str(current.parent.resolve()) if current.name == ".git" else str(current.resolve())
        parent = current.parent
        if parent == current:
            break  # reached filesystem root
        current = parent

    return ""


def _resolve_workspace_identity(session_row: Dict[str, Any]) -> str:
    """Extract workspace identity from a session DB row.

    Returns the most reliable available signal: git_repo_root > cwd.
    Empty string means no reliable identity (legacy/null fields).
    """
    repo_root = (session_row.get("git_repo_root") or "").strip()
    if repo_root:
        return repo_root

    cwd = (session_row.get("cwd") or "").strip()
    if cwd:
        # Try to resolve cwd as a workspace identity too, but git_repo_root
        # is preferred when both are available.
        resolved = _resolve_git_repo_root(cwd)
        return resolved if resolved else cwd

    return ""


def _current_workspace_identity(current_cwd: str | None) -> str:
    """Resolve the current working directory's workspace identity."""
    if not current_cwd:
        return ""
    # Try git_repo_root first (most reliable cross-workspace signal)
    resolved = _resolve_git_repo_root(current_cwd)
    if resolved:
        return resolved
    # Fall back to resolved cwd for non-git directories
    try:
        return str(Path(current_cwd).resolve())
    except (OSError, ValueError):
        return ""


def validate_session_workspace(
    session_row: Dict[str, Any],
    current_cwd: str | None = None,
) -> WorkspaceGuardResult:
    """Validate whether resuming a session is safe given the current workspace.

    Args:
        session_row: Session metadata dict from DB (must have cwd/git_repo_root).
        current_cwd: Current working directory of the caller process.

    Returns:
        WorkspaceGuardResult with ok/warning/blocked status.
    """
    stored = _resolve_workspace_identity(session_row)
    current = _current_workspace_identity(current_cwd) if current_cwd else ""

    # No stored identity → legacy session, allow but warn
    if not stored:
        return WorkspaceGuardResult(
            ok=True,
            warning="Session has no workspace identity recorded; resuming without cross-workspace check.",
            reason="legacy_session",
            stored_workspace="",
            current_workspace=current,
        )

    # No current identity → can't validate, allow but warn
    if not current:
        return WorkspaceGuardResult(
            ok=True,
            warning="Current working directory has no workspace identity; resuming without cross-workspace check.",
            reason="no_current_identity",
            stored_workspace=stored,
            current_workspace="",
        )

    # Both present — compare
    if stored == current:
        return WorkspaceGuardResult(
            ok=True,
            stored_workspace=stored,
            current_workspace=current,
        )

    # Mismatch → block restore
    return WorkspaceGuardResult(
        ok=False,
        reason="workspace_mismatch",
        stored_workspace=stored,
        current_workspace=current,
    )


def format_workspace_mismatch_error(result: WorkspaceGuardResult) -> str:
    """Format a user-facing error message for workspace mismatch."""
    return (
        f"⚠️  Cross-workspace resume blocked.\n\n"
        f"This session was created in: {result.stored_workspace}\n"
        f"You are currently in:         {result.current_workspace}\n\n"
        f"Hermes prevents resuming sessions from a different workspace to avoid "
        f"context leakage. Start a new session or resume from the original workspace."
    )


def format_legacy_warning(message: str) -> str:
    """Format a user-facing warning for legacy/null-session cases."""
    return f"[dim]⚠ {message}[/dim]"


def filter_sessions_by_workspace(
    sessions: List[Dict[str, Any]],
    current_cwd: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter session list by workspace compatibility.

    Returns (compatible, incompatible) — both lists contain the original dicts.
    Compatible sessions are those that match the current workspace or have no
    stored identity (legacy). Incompatible ones have a mismatched git_repo_root.

    This is used for auto-continue (--resume with no explicit ID): prefer
    compatible sessions over legacy ones, and skip incompatible ones entirely.
    """
    current = _current_workspace_identity(current_cwd) if current_cwd else ""
    compatible: List[Dict[str, Any]] = []
    incompatible: List[Dict[str, Any]] = []

    for session in sessions:
        stored = _resolve_workspace_identity(session)
        if not stored:
            # Legacy — compatible but lower priority (will be last resort)
            compatible.append(session)
        elif current and stored == current:
            compatible.append(session)
        else:
            incompatible.append(session)

    return compatible, incompatible


def stamp_compaction_metadata(
    summary_text: str,
    session_row: Dict[str, Any],
) -> str:
    """Stamp a compaction summary with workspace metadata.

    Adds a hidden header block to the compacted summary so future restores
    can validate workspace identity even after raw message history is gone.
    This is secondary defense — primary guard uses DB/session metadata.

    The marker format is designed to be parseable by restore code but invisible
    to models (prefixed with a comment-like token that models treat as neutral).
    """
    repo_root = _resolve_workspace_identity(session_row)
    if not repo_root:
        return summary_text

    # Use a structured metadata block that models won't act on but restore
    # code can parse. Format: <!-- HERMES_WORKSPACE:<value> -->
    marker = f"\n<!-- HERMES_WORKSPACE:{repo_root} -->"
    return summary_text + marker


def extract_workspace_from_compaction(summary_text: str) -> str:
    """Extract workspace identity from a stamped compaction summary."""
    try:
        import re
        match = re.search(r"<!--\s*HERMES_WORKSPACE:(.+?)\s*-->", summary_text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return ""


def _message_content_text(content: Any) -> str:
    """Best-effort text extraction from string or multimodal message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def augment_session_row_from_compaction(
    session_row: Dict[str, Any],
    messages: list[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """Fill missing workspace identity from stamped compaction summaries.

    Does not mutate the input row. Existing git_repo_root/cwd wins.
    """
    if _resolve_workspace_identity(session_row):
        return session_row
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        workspace = extract_workspace_from_compaction(
            _message_content_text(msg.get("content", ""))
        )
        if workspace:
            updated = dict(session_row)
            updated["git_repo_root"] = workspace
            return updated
    return session_row
