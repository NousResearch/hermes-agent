"""Own-DM scoping for session queries: what a non-admin Telegram Mini App caller may see.

Two shapes of the same rule, kept together so they cannot drift: a SQL ``WHERE`` clause for listing
and counting, and a row check for a single already-fetched session.
"""

from __future__ import annotations

from typing import Any


# Valid ``scope=`` values for list_sessions_rich/session_count.
_SESSION_LIST_SCOPES = ("own", "admin")


def _dm_own_scope_clause(requester_user_id: str) -> tuple[str, list[str]]:
    """WHERE clause restricting rows to *requester_user_id*'s own Telegram DMs.

    Used by the Telegram Mini App dashboard's non-admin tier (``scope="own"``
    on ``list_sessions_rich``/``session_count``) so a paired-but-non-admin
    caller only ever sees their own DM sessions, never another user's or a
    group/channel session. A Telegram DM's ``chat_id`` equals the
    participant's own user id, so matching ``chat_id`` doubles as a
    same-owner check without needing ``user_id`` at all. A row with a
    NULL/blank ``chat_id`` (legacy rows predating chat/thread capture) cannot
    prove ownership and is excluded — fails closed rather than guessing via
    ``user_id`` alone.
    """
    return (
        "s.source = 'telegram' AND s.chat_type = 'dm' "
        "AND s.chat_id IS NOT NULL AND s.chat_id != '' AND s.chat_id = ?",
        [str(requester_user_id)],
    )


def session_row_is_own_dm(session: dict[str, Any], requester_user_id: str) -> bool:
    """Row-match counterpart to :func:`_dm_own_scope_clause`.

    Same rule (``source == 'telegram'``, ``chat_type == 'dm'``, non-blank
    ``chat_id`` equal to ``requester_user_id``), evaluated against a single
    already-fetched session dict (e.g. ``SessionDB.get_session()``'s return
    value) instead of as a SQL WHERE filter. Used by the dashboard's
    single-session ownership check (``GET /api/sessions/{id}`` and
    ``.../messages``) where the caller already has the row in hand and a
    second query isn't needed — kept as a standalone function rather than
    inlined at the call site so the "what counts as ownership" rule has
    exactly one definition each for its two shapes (query-filter vs.
    row-match), not two independently-maintained copies that can drift.
    """
    if not requester_user_id:
        return False
    if session.get("source") != "telegram" or session.get("chat_type") != "dm":
        return False
    chat_id = session.get("chat_id")
    if not chat_id:
        return False
    return str(chat_id) == str(requester_user_id)
