"""Session-lifecycle helpers for gateway platform adapters.

Extracted verbatim from ``gateway/platforms/base.py`` (class
``BasePlatformAdapter``) — godfile decomposition shard s5.  Mixed into
``BasePlatformAdapter``; instance state (``_active_sessions``,
``_pending_messages``) resolves through the adapter via MRO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent


class SessionLifecycleMixin:
    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()
    
    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        return self._pending_messages.pop(session_key, None)
