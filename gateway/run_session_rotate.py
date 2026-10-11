"""Control-socket ``rotate-session``: rotate one channel's session on demand.

A channel's conversation could only ever start fresh in two ways: the time-based ``session_reset``
policy (``none|idle|daily|both`` — time is the only trigger, so it cannot react to an event) or a
human typing ``/new``. An external orchestrator that opens work for a channel — and wants that
conversation to start clean, after leaving a hand-off note — had no entry point at all.

Rotating it from outside the process is not an option either: :meth:`SessionStore.reset_session`
publishes a fresh routing entry into the gateway's IN-MEMORY index and only then ends the old row
(``_finish_route_transition(..., end_reason="session_reset")``). The gateway never re-reads
``state.db`` per message, so ending that row from another process rotates nothing while it lives.

So the verb asks the process that owns the index to do it, reusing what already exists: a channel's
identity is what the routing index already holds for it (``SessionEntry.origin``, the source every
ingress path built) and the rotation itself is ``reset_session``. Nothing is CREATED here — a
channel with no live session answers ``rotated: false``, which is exactly what a caller firing
before the channel's first message needs (no special-casing on their side).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Channel identity, matched against a routing entry's ``origin``: platform + chat are what a
# channel IS; the topic/thread and the participant narrow it down when the chat has them.
_REQUIRED_FIELDS = ("platform", "chat_id")
_OPTIONAL_FIELDS = ("thread_id", "user_id")


def _text(value: Any) -> Optional[str]:
    """Wire value as text (enums by their value, ints by their digits); None when absent or blank.

    Case is preserved here — a ``session_key`` is matched verbatim — and comparison of CHANNEL
    fields is case-insensitive in :func:`_matches`.
    """
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _matches(origin: Any, wanted: Dict[str, str]) -> bool:
    """True when a routing entry's ``origin`` describes the requested channel exactly.

    Compared case-insensitively: no id or platform name Hermes routes on is case-significant
    (Telegram/Discord ids are numeric, Slack's are upper-case, WhatsApp's lower).
    """
    return all((_text(getattr(origin, name, None)) or "").lower() == value.lower()
               for name, value in wanted.items())


def _rotate(store: Any, session_key: str) -> Dict[str, Any]:
    """Rotate *session_key* through the store's own counter, or say why it did not."""
    entry = store.lookup_by_session_key(session_key)
    if entry is None:
        return {"rotated": False, "reason": "no_session", "session_key": session_key}
    old_session_id = entry.session_id
    new_entry = store.reset_session(session_key)
    if new_entry is None:  # lost the race with a teardown that dropped the entry
        return {"rotated": False, "reason": "no_session", "session_key": session_key}
    logger.info("Session rotated on request: %s (%s -> %s)",
                session_key, old_session_id, new_entry.session_id)
    return {
        "rotated": True, "session_key": session_key, "old_session_id": old_session_id,
        "new_session_id": new_entry.session_id,
        # reset_session's own reason for closing the row (gateway/session.py); asserted against the
        # state.db row in tests/gateway/test_control_socket_rotate_session.py.
        "end_reason": "session_reset",
    }


def rotate_session_verb(runner: Any) -> Callable[..., Dict[str, Any]]:
    """Control-socket ``rotate-session``: ``{platform, chat_id[, thread_id][, user_id][, profile]}``
    or ``{session_key}`` -> ``{rotated, session_key, old_session_id, new_session_id, end_reason}``.

    Answers ``rotated: false`` with a ``reason`` when the channel has no live session
    (``no_session``) or when the identity it names matches several of them (``ambiguous``, plus the
    ``candidates`` found — pass one back as ``session_key`` to pick it explicitly). ``profile``
    narrows a chat served by more than one profile on a multiplexed gateway.

    Runs on the control socket's executor thread: the store is thread-safe and owns its own lock, so
    no hop onto the gateway loop is needed (unlike the verbs that touch live adapters).
    """

    def _handler(params: Optional[dict] = None) -> Dict[str, Any]:
        from gateway.session import _session_key_namespace

        params = params or {}
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"rotated": False, "error": "gateway has no session store"}

        explicit = _text(params.get("session_key"))
        if explicit is not None:
            return _rotate(store, explicit)

        wanted = {name: _text(params.get(name)) for name in _REQUIRED_FIELDS + _OPTIONAL_FIELDS}
        missing = [name for name in _REQUIRED_FIELDS if wanted[name] is None]
        if missing:
            return {"rotated": False, "error": f"{' and '.join(missing)} required (or pass session_key)"}
        wanted = {name: value for name, value in wanted.items() if value is not None}

        profile = _text(params.get("profile"))
        namespace = _session_key_namespace(profile) + ":" if profile else None
        candidates = sorted({
            entry.session_key for entry in store.list_sessions()
            if entry.origin is not None and _matches(entry.origin, wanted)
            and (namespace is None or str(entry.session_key).startswith(namespace))})
        if not candidates:
            return {"rotated": False, "reason": "no_session", "channel": wanted}
        if len(candidates) > 1:
            return {"rotated": False, "reason": "ambiguous", "channel": wanted,
                    "candidates": candidates}
        return _rotate(store, candidates[0])

    return _handler
