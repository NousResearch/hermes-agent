"""Centralized audit logging for skill-directory mutations.

Issue #100449: skill files can be written through paths other than
``skill_manage`` (e.g. ``write_file``, ``patch``, ``terminal``). Those
bypasses leave no trace in ``.usage.json`` or the staged-pending store, so
the ``skills-integrity`` watchdog cannot attribute drift to a session.

This module provides a single, append-only audit log that records every
mutation touching the local skills tree, regardless of which tool performed
it. It is a *detection* mechanism, not a prevention boundary; preventing
direct writes would require OS-level privilege separation (#99729).

Logged events live in ``<HERMES_HOME>/skills/.audit.log`` as newline-delimited
JSON objects. Each record contains:

    timestamp   ISO-8601 UTC timestamp
    tool        Tool name (e.g. "write_file", "patch", "terminal")
    path        Resolved absolute path that was mutated
    action      "create", "modify", "remove", or "unknown"
    origin      "foreground" / "background_review" (best-effort)
    session_id  Hermes session id, when available
    tool_call_id  Model tool-call id, when available

The log is append-only and best-effort: a failure to write the audit record
never blocks the underlying operation, because the goal is observability,
not enforcement.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Directory-relative audit log location under the active HERMES_HOME.
_SKILLS_AUDIT_LOG = "skills/.audit.log"


def _skill_audit_log_path() -> Path:
    return get_hermes_home() / _SKILLS_AUDIT_LOG


def _resolve_hermes_skill_dirs() -> list[Path]:
    """Return the absolute skill directories we consider in-scope for audit.

    Includes both the active profile's ``skills/`` tree and the global
    default profile's ``~/.hermes/skills/`` tree, in case a profile-relative
    write needs to be correlated against the canonical store.
    """
    from hermes_constants import get_default_hermes_root, get_hermes_home

    dirs: list[Path] = []
    for base in (get_hermes_home(), get_default_hermes_root()):
        try:
            real = base.resolve()
        except Exception:
            continue
        for candidate in (real / "skills", real / "profiles"):
            if candidate not in dirs:
                dirs.append(candidate)
    return dirs


def is_skill_dir_path(path: str) -> bool:
    """Return True if ``path`` lives under a Hermes skills directory.

    Recognizes both the default ``~/.hermes/skills/`` tree and profile-local
    ``~/.hermes/profiles/<name>/skills/`` trees. Symlinks are resolved so that
    path-traversal tricks (e.g. via ``..`` or a symlink) still match.
    """
    if not path:
        return False
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False

    for skill_root in _resolve_hermes_skill_dirs():
        try:
            resolved.relative_to(skill_root)
            return True
        except ValueError:
            continue
    return False


def _current_session_id() -> Optional[str]:
    """Best-effort read of the active Hermes session id from the environment."""
    for key in ("HERMES_SESSION_ID", "HERMES_ACTIVE_SESSION"):
        val = os.getenv(key)
        if val:
            return val
    return None


def _current_origin() -> str:
    """Best-effort origin classification for the current execution context."""
    try:
        from tools.skill_provenance import get_current_write_origin

        return get_current_write_origin()
    except Exception:
        return "foreground"


def append_skill_audit_record(
    tool: str,
    path: str,
    action: str = "unknown",
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one audit record for a skill-directory mutation.

    Args:
        tool: Tool that performed the mutation, e.g. ``write_file``,
            ``patch``, ``terminal``.
        path: Absolute or relative path that was touched. Resolved internally.
        action: Mutation class: ``create``, ``modify``, ``remove``,
            or ``unknown``.
        session_id: Optional Hermes session id. If omitted, read from env.
        tool_call_id: Optional model tool-call id for correlation.
        extra: Optional dict merged into the record.
    """
    try:
        if not is_skill_dir_path(path):
            return
    except Exception:
        # Safety: never crash the caller on audit classification failure.
        logger.debug("skill_audit classification failed for %r", path, exc_info=True)
        return

    log_path = _skill_audit_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "tool": tool,
        "path": str(Path(path).expanduser().resolve()),
        "action": action,
        "origin": _current_origin(),
        "session_id": session_id or _current_session_id() or "unknown",
        "tool_call_id": tool_call_id or "unknown",
        "record_id": uuid.uuid4().hex[:8],
    }
    if extra:
        record.update(extra)

    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.debug("Failed to append skill audit record to %s: %s", log_path, e)
