"""``STATUS_ANCHOR_MESSAGE_ID``: the inbound message a turn answers, carried in reply metadata.

Every platform's reply / progress metadata carries it, with no platform conditionals. An adapter
that shows a per-message status indicator (a turn answered top-level, with no thread to hang a
status on) anchors it there; routing never reads the key, so every other adapter ignores it.

``STATUS_OWNER_KEY`` rides beside it: the session key of the turn that owns the status. A
``/queue`` follow-up runs in the same session, so its status shares the opener's owner, and an
adapter clearing a finished turn's statuses takes exactly that session's, never a concurrent
session's in the same chat.
"""

from __future__ import annotations

from typing import Any, Optional

STATUS_ANCHOR_MESSAGE_ID = "status_anchor_message_id"
STATUS_OWNER_KEY = "status_owner_key"


def with_status_anchor(
    metadata: Optional[dict[str, Any]], message_id: Any, owner: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """``metadata`` plus the status anchor (and its owning session key, when known); unchanged
    (possibly None) when there is no message id."""
    if message_id is None or message_id == "":
        return metadata
    anchored = {**(metadata or {}), STATUS_ANCHOR_MESSAGE_ID: str(message_id)}
    if owner:
        anchored[STATUS_OWNER_KEY] = str(owner)
    return anchored


def without_status_anchor(metadata: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """``metadata`` minus the turn-scoped status keys, for metadata stored past the turn (a
    subscription row replayed later has no running turn to anchor a status on)."""
    if not metadata:
        return metadata
    return {k: v for k, v in metadata.items() if k not in (STATUS_ANCHOR_MESSAGE_ID, STATUS_OWNER_KEY)}
