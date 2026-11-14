"""SessionStore prompt pins: the durable snapshot of the system-prompt inputs internal turns reuse.

Internal events (kanban wakes, delegation completions, startup resume) reuse the last human turn's
session-context bytes and channel prompt so the system prompt stays byte-stable; the routing entry
carries that snapshot so a gateway restart does not flip it (#125793).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

PROMPT_PIN_VERSION = 1


def sanitize_prompt_pin(pin: Any) -> Optional[dict[str, Any]]:
    """Validated durable snapshot of the exact ephemeral inputs reused by internal turns.

    ``redact_pii`` is the ``privacy.redact_pii`` the context bytes were rendered under; a pin
    without it cannot prove which privacy policy produced them and is refused."""
    if not isinstance(pin, dict) or pin.get("version") != PROMPT_PIN_VERSION:
        return None
    context_key = pin.get("context_key")
    context_prompt = pin.get("context_prompt")
    redact_pii = pin.get("redact_pii")
    channel_prompt = pin.get("channel_prompt")
    parent_chat_id = pin.get("parent_chat_id")
    if not isinstance(context_key, str) or not context_key or not isinstance(context_prompt, str):
        return None
    if not isinstance(redact_pii, bool):
        return None
    if channel_prompt is not None and not isinstance(channel_prompt, str):
        return None
    if parent_chat_id is not None and not isinstance(parent_chat_id, str):
        return None
    return {
        "version": PROMPT_PIN_VERSION,
        "context_key": context_key,
        "context_prompt": context_prompt,
        "redact_pii": redact_pii,
        "channel_prompt": channel_prompt,
        "parent_chat_id": parent_chat_id,
    }


class SessionPromptPinMixin:
    """Fenced read/write of ``SessionEntry.prompt_pin`` on the routing index."""

    def set_prompt_pin(
        self, session_key: str, pin: dict[str, Any], *, expected_session_id: Optional[str] = None,
    ) -> bool:
        """Persist effective prompt inputs without letting a stale turn cross a boundary.

        The candidate is durable before the in-memory entry is published. The expected session id
        is the turn's launch identity: a concurrent reset or resume makes this a no-op instead of
        writing the old conversation's system bytes onto the new one.
        """
        cleaned = sanitize_prompt_pin(pin)
        if cleaned is None:
            return False
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or (expected_session_id is not None and entry.session_id != expected_session_id):
                return False
            if entry.prompt_pin != cleaned:
                self._save_entry(
                    session_key, entry_data=replace(entry, prompt_pin=cleaned).to_dict(), lock_held=True)
                entry.prompt_pin = cleaned
            return True

    def get_prompt_pin(
        self, session_key: str, *, expected_session_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Return the prompt pin only while the route still owns the caller's session."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            if expected_session_id is not None and entry.session_id != expected_session_id:
                return None
            return dict(entry.prompt_pin) if entry.prompt_pin else None
