"""Helpers for messaging approval shortcut replies."""

from __future__ import annotations

import dataclasses
from typing import Any


NUMERIC_APPROVAL_SHORTCUTS = {
    "1": "/approve",
    "2": "/approve session",
    "3": "/approve always",
    "4": "/deny",
}


def rewrite_numeric_approval_shortcut(event: Any, session_key: str) -> Any | None:
    """Return an event rewritten to /approve or /deny when an approval is live."""
    raw = (getattr(event, "text", "") or "").strip()
    replacement = NUMERIC_APPROVAL_SHORTCUTS.get(raw)
    if replacement is None:
        return None

    try:
        from tools.approval import has_blocking_approval
    except Exception:
        return None

    try:
        if not has_blocking_approval(session_key):
            return None
    except Exception:
        return None

    return dataclasses.replace(event, text=replacement)
