"""Build the bounded recent-session page before expensive row projection."""
from hermes_state_common import _RESET_CHILD_SQL, _sql_in_window, _sql_json_extract


def recent_page_sql(candidate_where: str, outer_where: str, select_head: str,
                    prompt_join: str, member_activity_sql: str, last_active_sql: str) -> str:
    """Keep exact chain ordering and filters inside a materialized ID page."""
    return f"""
        WITH RECURSIVE candidate_sessions(id, activity, started_at) AS MATERIALIZED (
            SELECT s.id, COALESCE(s.last_activity_at, s.started_at), s.started_at
            FROM sessions s {candidate_where}
        ),
        rooted_candidates(root_id, cur_id) AS (
            SELECT cs.id, cs.id FROM candidate_sessions cs
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions child
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE child.id = cs.id AND parent.end_reason = 'compression'
                  AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
                  AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
                  AND NOT ({_RESET_CHILD_SQL.format(a='child')})
                  AND COALESCE(child.source, '') != 'tool'
            )
            UNION
            SELECT rc.root_id, child.id
            FROM rooted_candidates rc
            JOIN sessions parent ON parent.id = rc.cur_id
            JOIN sessions child ON child.parent_session_id = rc.cur_id
            JOIN candidate_sessions cs ON cs.id = child.id
            WHERE parent.end_reason = 'compression'
              AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
              AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
              AND NOT ({_RESET_CHILD_SQL.format(a='child')})
              AND COALESCE(child.source, '') != 'tool'
        ),
        candidate_roots(root_id) AS (
            SELECT root_id FROM (
                SELECT rc.root_id, MAX(cs.activity) AS candidate_activity,
                       MAX(cs.started_at) AS candidate_started_at
                FROM rooted_candidates rc
                JOIN candidate_sessions cs ON cs.id = rc.cur_id
                GROUP BY rc.root_id
            )
            ORDER BY candidate_activity DESC, candidate_started_at DESC, root_id DESC
            LIMIT ?
        ),
        chain(root_id, cur_id) AS (
            SELECT root_id, root_id FROM candidate_roots
            UNION ALL
            SELECT c.root_id, child.id
            FROM chain c
            JOIN sessions parent ON parent.id = c.cur_id
            JOIN sessions child ON child.parent_session_id = c.cur_id
            WHERE parent.end_reason = 'compression'
              AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
              AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
              AND NOT ({_RESET_CHILD_SQL.format(a='child')})
              AND COALESCE(child.source, '') != 'tool'
        ),
        chain_max AS (
            SELECT
                root_id,
                MAX({member_activity_sql}) AS effective_last_active
            FROM chain
            GROUP BY root_id
        ),
        -- Fence preview/blob projection behind the exact page window.
        page AS MATERIALIZED (
            SELECT s.id,
                COALESCE(cm.effective_last_active, {_sql_in_window("s.started_at")}) AS _effective_last_active
            FROM sessions s
            JOIN candidate_roots cr ON cr.root_id = s.id
            LEFT JOIN chain_max cm ON cm.root_id = s.id
            {outer_where}
            ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
            LIMIT ? OFFSET ?
        )
        {select_head}{last_active_sql} AS last_active,
            page._effective_last_active
        FROM page
        JOIN sessions s ON s.id = page.id
        {prompt_join}
        ORDER BY page._effective_last_active DESC, s.started_at DESC, s.id DESC
    """
