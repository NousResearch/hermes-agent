"""Telegram sessions not yet linked to a topic."""


from __future__ import annotations
import sqlite3
from typing import Any, Optional
from persistence.sqlite.conversation._store.sql_common import _sql_preview_raw, _sql_session_last_active
from persistence.sqlite._holder import StoreHolder


def _normalize_telegram_topic_profile_name(profile_name: Optional[str] = None) -> str:
    """Empty/missing → ``"default"`` (single namespace for non-multiplexed gateways).
    Multiplexed callers must pass the *routed* profile (``source.profile``), never the
    process-global active profile."""
    name = str(profile_name or "").strip()
    return name if name else "default"


_UNLINKED_SELECT_HEAD = f"""
                    SELECT s.*,
                        COALESCE(sp.prompt, s.system_prompt)
                            AS _system_prompt_resolved,
                        {_sql_preview_raw()},
                        {_sql_session_last_active("s")} AS last_active
                    FROM sessions s
                    LEFT JOIN system_prompts sp
                      ON sp.hash = s.system_prompt_hash
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
"""


_UNLINKED_SELECT_TAIL = """                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """


_UNLINKED_SCOPE_CLAUSES = """                      AND COALESCE(NULLIF(TRIM(s.profile_name), ''), 'default') = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_dm_topic_bindings b
                          WHERE b.session_id = s.id
                      )
"""


def list_unlinked_telegram_sessions_for_user(
    holder: StoreHolder, *, chat_id: str, user_id: str, profile_name: str = "default", limit: int = 10
) -> list[dict[str, Any]]:
    """This user's Telegram sessions not bound to a topic. Read-only: if the bindings table
    is absent, every session is unlinked and the profile-unscoped query is used.
    Scoped by ``profile_name`` so multiplexed profiles do not surface each other.

    See #76423.
    """
    profile_name = _normalize_telegram_topic_profile_name(profile_name)
    with holder._read_ctx() as conn:
        try:
            rows = conn.execute(
                _UNLINKED_SELECT_HEAD + _UNLINKED_SCOPE_CLAUSES + _UNLINKED_SELECT_TAIL,
                (str(user_id), profile_name, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                _UNLINKED_SELECT_HEAD + _UNLINKED_SELECT_TAIL, (str(user_id), int(limit)),
            ).fetchall()
    return [_rich_row(row) for row in rows]


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.portability import _rich_row
