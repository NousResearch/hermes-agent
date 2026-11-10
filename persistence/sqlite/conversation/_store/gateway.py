"""Gateway session listing."""


from __future__ import annotations
from typing import Any, Optional
from persistence.sqlite.conversation._store.sql_common import _sql_session_last_active
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.core import _session_row_dict


def list_gateway_sessions(
    holder: StoreHolder, *, platform: Optional[str] = None, active_only: bool = True) -> list[dict[str, Any]]:
    """List gateway sessions (rows with a session_key): newest row per key, one live
    mapping per routing key. ``platform`` filters on ``source``."""
    # Full rows carry token/cost totals — drain queued async accounting deltas first.
    holder.flush_for_read()
    query = f"""
            SELECT sessions.*,
                   COALESCE(sp.prompt, sessions.system_prompt)
                       AS _system_prompt_resolved,
                   {_sql_session_last_active("sessions")} AS last_active
            FROM sessions
            LEFT JOIN system_prompts sp
              ON sp.hash = sessions.system_prompt_hash
            WHERE session_key IS NOT NULL
              AND started_at = (
                  SELECT MAX(s2.started_at) FROM sessions s2
                  WHERE s2.session_key = sessions.session_key
              )
        """
    params: list = [platform] if platform else []
    query += (" AND LOWER(source) = LOWER(?)" if platform else "") + (
        " AND ended_at IS NULL" if active_only else "") + " ORDER BY last_active DESC"
    return [_session_row_dict(r) for r in holder._read_all(query, params)]
