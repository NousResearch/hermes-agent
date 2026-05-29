"""Per-session pending media queue for direct-enqueue tool delivery.

Some tools (e.g. ``browser_screenshot``) capture binary media files and want
to deliver them as native chat attachments without round-tripping a path
through the agent's text response. The agent process and the gateway send
loop live in the same Python process and share state via this module.

Pattern:
    1. Tool runs (inside the agent loop, in the gateway process). It saves
       the media file to a host path it controls and calls ``enqueue_media``
       with that path. No path is exposed to the agent.
    2. Tool returns a tiny success result to the agent. The agent generates
       its text response normally.
    3. After the gateway sends the agent's text response (and any explicit
       ``MEDIA:`` tags), it calls ``drain_media`` for the same session and
       sends every queued path as a native attachment.
    4. The queue is keyed by ``session_key`` (the same identifier used by
       ``tools.approval.set_current_session_key``), so concurrent sessions
       on different chats stay isolated.

Idempotency: ``drain_media`` atomically removes the entries before they are
sent, so a single response cycle flushes each queued item exactly once.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

_pending: Dict[str, List[str]] = {}
_lock = threading.Lock()


def enqueue_media(path: str, session_key: Optional[str] = None) -> None:
    """Append a media file path to the queue for the given session.

    If ``session_key`` is None the current session is looked up from the
    ``tools.approval`` ContextVar. Tools running inside the agent loop can
    therefore call ``enqueue_media(path)`` with no arguments.
    """
    if session_key is None:
        try:
            from tools.approval import get_current_session_key
            session_key = get_current_session_key()
        except Exception:
            session_key = "default"

    with _lock:
        _pending.setdefault(session_key, []).append(path)


def drain_media(session_key: str) -> List[str]:
    """Atomically remove and return all pending media paths for a session."""
    with _lock:
        return _pending.pop(session_key, [])


def peek_media(session_key: str) -> List[str]:
    """Return a copy of the pending paths for a session without removing them."""
    with _lock:
        return list(_pending.get(session_key, ()))
