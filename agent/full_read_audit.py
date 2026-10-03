"""Full-read audit: gate synthesis on verified whole-folder reads (#124875).

When the user asks to "read all files" (explicit completeness quantifier), the
turn snapshots the workspace inventory at admission. The stop gate then consults
the read tracker's per-task coverage (``has_complete_read``) before a final text
response is delivered; a missing file blocks the answer with a nudge naming it.
The finalizer re-checks as a backstop for budget-exhaustion paths.

Fail-open everywhere: no workspace anchor, oversized inventory, or walker error
means no audit (debug log only). The turn loop must never break because of this.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger("agent.conversation_loop")

# Only explicit completeness quantifiers arm the audit (a vague "take a look"
# must not). Matched against the flattened user text, case-insensitive.
_FULL_READ_PATTERNS = (
    r"read\s+all\b",
    r"read\s+every",
    r"read\s+everything",
    r"review\s+all\b",
    r"scan\s+all\b",
    r"go\s+through\s+everything",
    r"read\s+the\s+whole\b",
    r"review\s+everything",
)
_FULL_READ_RE = re.compile("|".join("(?:%s)" % p for p in _FULL_READ_PATTERNS), re.IGNORECASE)

# Walked at admission; anything under these dir names is not user content.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", "target", ".next", ".cache",
})

# ponytail: fixed inventory cap; beyond it the audit stays disarmed (fail-open).
# Raise only with evidence of real whole-folder requests above this size.
_MAX_AUDIT_FILES = 300
_MAX_NUDGE_NAMES = 20
#: Follow-up turns a refusal's audit stays armed for. The refusal invites a "continue",
#: which matches no completeness pattern; bounded so an abandoned folder cannot gate the
#: task forever (#124875).
_PENDING_TURNS = 3
#: Attribute carrying the refusal's inventory across turn start (which clears the audit).
_PENDING_ATTR = "_full_read_audit_pending"


def _flatten_user_text(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts = []
        for part in user_message:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if user_message is None else str(user_message)


def audit_requested(user_message: Any) -> bool:
    """True when the user text carries an explicit whole-folder read request."""
    try:
        return bool(_FULL_READ_RE.search(_flatten_user_text(user_message)))
    except Exception:
        return False


def _inventory_files(root: str) -> Optional[List[str]]:
    """Absolute paths of auditable files under *root*, or None to stay disarmed."""
    try:
        from tools.file_tools import get_read_block_error, has_binary_extension
        from tools.read_extract import is_extractable_document
    except Exception:
        logger.debug("full-read audit: file-tool helpers unavailable", exc_info=True)
        return None
    found: List[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if not os.path.isfile(path) or os.path.islink(path):
                        continue
                    # Binary files the read tool cannot serve are unauditable;
                    # extractable documents (.docx/...) count as full reads.
                    if has_binary_extension(path) and not is_extractable_document(path):
                        continue
                    if get_read_block_error(path):
                        continue
                except Exception:
                    continue
                found.append(os.path.abspath(path))
                if len(found) > _MAX_AUDIT_FILES:
                    logger.debug("full-read audit: inventory over cap (%d), disarmed", _MAX_AUDIT_FILES)
                    return None
    except Exception:
        logger.debug("full-read audit: inventory walk failed, disarmed", exc_info=True)
        return None
    return found


def arm_full_read_audit(agent: Any, user_message: Any, task_id: str) -> None:
    """Snapshot the inventory when the user asked for a whole-folder read.

    Always (re)sets ``agent._full_read_audit``: a dict when armed, else None. A refusal
    leaves its inventory behind in ``_PENDING_ATTR`` so the follow-up turn — which carries
    no completeness quantifier — is audited too, for at most ``_PENDING_TURNS`` turns.
    Fail-open: any error leaves the audit disarmed."""
    pending = getattr(agent, _PENDING_ATTR, None)
    if not isinstance(pending, dict) or pending.get("task_id") != task_id:
        pending = None
        setattr(agent, _PENDING_ATTR, None)
    agent._full_read_audit = None
    try:
        if not audit_requested(user_message):
            # ``finalizer_refusal`` ends with "Send `continue`": that follow-up matches no
            # completeness pattern, so the audit it refers to is re-armed here — otherwise
            # the very next turn delivers the synthesis that was just refused (#124875).
            if pending is None or pending.get("turns_left", 0) <= 0:
                setattr(agent, _PENDING_ATTR, None)
                return
            pending["turns_left"] = pending["turns_left"] - 1
            agent._full_read_audit = {
                "task_id": task_id,
                "root": pending.get("root") or "",
                "paths": list(pending.get("paths") or []),
            }
            logger.debug("full-read audit re-armed from refusal (%d turns left)",
                         pending["turns_left"])
            return
        setattr(agent, _PENDING_ATTR, None)
        from tools.file_tools_paths import _authoritative_workspace_root
        root = _authoritative_workspace_root(task_id)
        if not root or not os.path.isdir(root):
            return
        paths = _inventory_files(root)
        if not paths:
            return
        agent._full_read_audit = {"task_id": task_id, "root": root, "paths": paths}
        logger.debug("full-read audit armed: %d files under %s", len(paths), root)
    except Exception:
        logger.debug("full-read audit: arm failed, disarmed", exc_info=True)
        agent._full_read_audit = None


def disarm_full_read_audit(agent: Any) -> None:
    agent._full_read_audit = None


def missing_read_paths(agent: Any) -> List[str]:
    """Inventoried files the task has not read in full (empty when disarmed)."""
    state = getattr(agent, "_full_read_audit", None)
    if not isinstance(state, dict) or not state.get("paths"):
        return []
    try:
        from tools.file_tools_read_tracking import has_complete_read
    except Exception:
        return []
    task_id = state.get("task_id", "default")
    missing = []
    for path in state["paths"]:
        try:
            if not has_complete_read(path, task_id):
                missing.append(path)
        except Exception:
            missing.append(path)
    if not missing:
        # Satisfied: drop the cross-turn re-arm so later unrelated turns are not gated.
        setattr(agent, _PENDING_ATTR, None)
    return missing


def _display_paths(paths: List[str], root: str) -> List[str]:
    shown = []
    for path in paths[:_MAX_NUDGE_NAMES]:
        try:
            shown.append(os.path.relpath(path, root))
        except Exception:
            shown.append(os.path.basename(path))
    return shown


def build_full_read_nudge(agent: Any) -> Optional[str]:
    """Stop-gate nudge naming the unread files, or None when nothing is missing."""
    missing = missing_read_paths(agent)
    if not missing:
        return None
    state = getattr(agent, "_full_read_audit", None) or {}
    names = _display_paths(missing, state.get("root") or "")
    lines = "\n".join("- %s" % name for name in names)
    if len(missing) > len(names):
        lines += "\n- ... and %d more" % (len(missing) - len(names))
    return (
        "You said you would read everything, but these files have not been "
        "fully read yet:\n%s\n"
        "Read each file in full (page multi-page files through to the last "
        "line), then continue. Do not summarize until every file above has "
        "been read." % lines
    )


def finalizer_refusal(agent: Any) -> Optional[str]:
    """Backstop refusal for the finalizer, or None when the audit is satisfied."""
    missing = missing_read_paths(agent)
    if not missing:
        return None
    state = getattr(agent, "_full_read_audit", None) or {}
    # The refusal ends with "Send `continue`": leave the inventory behind so that follow-up
    # turn is audited too, instead of silently delivering this synthesis (#124875).
    setattr(agent, _PENDING_ATTR, {
        "task_id": state.get("task_id") or "default",
        "root": state.get("root") or "",
        "paths": list(state.get("paths") or []),
        "turns_left": _PENDING_TURNS,
    })
    names = _display_paths(missing, state.get("root") or "")
    lines = "\n".join("- %s" % name for name in names)
    if len(missing) > len(names):
        lines += "\n- ... and %d more" % (len(missing) - len(names))
    return (
        "I could not complete the requested full read: these files were not "
        "fully read before the turn ended:\n%s\n"
        "No synthesis is provided because it would be based on an incomplete "
        "read. Send `continue` to let me read the remaining files." % lines
    )
