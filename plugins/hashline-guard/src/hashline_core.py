"""hashline-guard core: strict-match pre-check on patch `old_string` anchors.

verify_anchor(file_text, old_string) -> ('ok', None) | ('block', reason)
context_hash(file_text, old_string, window=2) -> SHA-256 hex string

Fail-open: any IO/setup error is logged and skipped, never blocks.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_EMPTY_REASON = "old_string must be non-empty"


def verify_anchor(file_text: str, old_string: str) -> Tuple[str, Optional[str]]:
    """Return ('ok', None) when old_string occurs exactly once, else ('block', reason)."""
    if not old_string:
        return "block", _EMPTY_REASON

    count = file_text.count(old_string)
    if count == 0:
        return "block", "old_string not found in live file — anchor drifted"
    if count > 1:
        return "block", f"old_string is ambiguous: found {count} times in live file"
    return "ok", None


def context_hash(file_text: str, old_string: str, window: int = 2) -> str:
    """SHA-256 of old_string + up to `window` surrounding lines (foundation for full-hashline primitive)."""
    if not old_string:
        return hashlib.sha256(b"").hexdigest()

    lines = file_text.splitlines()
    idx = file_text.find(old_string)
    if idx == -1:
        # Anchor absent: hash old_string alone with no context
        return hashlib.sha256(old_string.encode("utf-8")).hexdigest()

    # Compute line index of the anchor's first line
    preceding = file_text[:idx]
    anchor_line = preceding.count("\n")
    start = max(0, anchor_line - window)
    end = min(len(lines), anchor_line + old_string.count("\n") + 1 + window)
    window_lines = lines[start:end]
    payload = "\n".join(window_lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
