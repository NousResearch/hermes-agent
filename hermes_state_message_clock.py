"""Project message-clock mixin for SessionDB: the strict last user/assistant message time
of each logical conversation (compression lineages included), used to order projects by
real conversation recency independently of the bounded sidebar selection."""

from __future__ import annotations

from typing import Any

from hermes_state_common import (
    _RESET_CHILD_SQL, _id_chunks, _placeholders as _session_ids_placeholders,
    _sql_in_window, _sql_json_extract, _sql_session_last_active,
)
from hermes_state_sessions import _session_filter_where, _where_sql


class SessionMessageClockMixin:
    def all_project_message_clock_rows(self, exclude_sources: list[str]) -> list[dict[str, Any]]:
        """Uncapped, light-weight project placement rows with complete compression lineages.

        Keep the same root admission as list_sessions_rich used by projects.tree, but
        do not fetch transcripts/previews or project every compressed root one by one.
        The preferred continuation uses the same ordering as get_compression_chain.
        This is independent of the overview's session/payload limit.
        """
        clauses, params = _session_filter_where(
            exclude_children=True, exclude_sources=exclude_sources,
            min_message_count=1, include_archived=False)
        clauses.append("s.hidden = 0")
        edge = f"""parent.end_reason = 'compression'
            AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
            AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
            AND NOT ({_RESET_CHILD_SQL.format(a='child')})
            AND COALESCE(child.source, '') != 'tool'"""
        rows = self._read_all(f"""
            WITH RECURSIVE
            eligible AS (SELECT s.id FROM sessions s {_where_sql(clauses)}),
            preferred AS (
                SELECT child.parent_session_id AS parent_id, child.id,
                    ROW_NUMBER() OVER (PARTITION BY child.parent_session_id ORDER BY
                        CASE WHEN child.end_reason = 'compression' THEN 0
                             WHEN child.ended_at IS NULL THEN 1 ELSE 2 END,
                        {_sql_session_last_active('child')} DESC,
                        child.started_at DESC, child.id DESC) AS priority
                FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE {edge}
            ),
            chain(root_id, cur_id) AS (
                SELECT id, id FROM eligible
                UNION
                SELECT chain.root_id, next.id FROM chain
                JOIN preferred next ON next.parent_id = chain.cur_id AND next.priority = 1
            )
            SELECT chain.root_id, chain.cur_id, tip.cwd, tip.git_branch, tip.git_repo_root,
                   next.id AS next_id
            FROM chain JOIN sessions tip ON tip.id = chain.cur_id
            LEFT JOIN preferred next ON next.parent_id = chain.cur_id AND next.priority = 1
        """, params)
        by_root: dict[str, dict[str, Any]] = {}
        for row in rows:
            root = by_root.setdefault(row['root_id'], {'segments': [], 'tip': None})
            root['segments'].append(row['cur_id'])
            if row['next_id'] is None:
                root['tip'] = row
        # Never present a partial/incorrect clock for malformed cyclic lineages.
        if any(group['tip'] is None for group in by_root.values()):
            raise ValueError('cyclic compression lineage in project message clock')
        projected = [
            {'id': group['tip']['cur_id'], '_lineage_ids': group['segments'],
             'cwd': group['tip']['cwd'], 'git_branch': group['tip']['git_branch'],
             'git_repo_root': group['tip']['git_repo_root']}
            for group in by_root.values()
        ]
        clocks = self.last_user_assistant_message_times(projected)
        for row in projected:
            row['last_message_at'] = clocks[row['id']]
        return projected

    def last_user_assistant_message_times(self, sessions: list[dict[str, Any]]) -> dict[str, float]:
        """Strict message clock for the selected logical conversations, keyed by projected row id.

        A compression-projected row names the tip but includes all predecessor ids in
        ``_lineage_ids``. Query just those selected segments in bounded batches, never one
        SELECT per tile. Inactive rewound rows are not conversation content; compacted
        predecessors remain visible history and count. No activity/start fallback.
        """
        by_segment: dict[str, list[str]] = {}
        for row in sessions:
            sid = row.get("id")
            if sid:
                for segment in row.get("_lineage_ids") or [sid]:
                    by_segment.setdefault(segment, []).append(sid)
        times = {row["id"]: 0.0 for row in sessions if row.get("id")}
        if not by_segment:
            return times
        for chunk in _id_chunks(list(by_segment)):
            rows = self._read_all(f"""
                SELECT m.session_id, MAX(m.timestamp) AS message_time
                FROM messages m INDEXED BY idx_messages_session
                WHERE m.session_id IN ({_session_ids_placeholders(chunk)})
                  AND m.role IN ('user', 'assistant')
                  AND (m.active = 1 OR m.compacted = 1)
                  AND {_sql_in_window('m.timestamp')} IS NOT NULL
                GROUP BY m.session_id
            """, chunk)
            for result in rows:
                value = float(result["message_time"])
                for sid in by_segment[result["session_id"]]:
                    times[sid] = max(times[sid], value)
        return times
