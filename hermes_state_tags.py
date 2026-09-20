"""Profile-local tag catalogue and compression-root assignments for SessionDB."""
from __future__ import annotations

import unicodedata

from hermes_state_common import _sql_json_extract


# Markers describe an edge, not ancestry: compression inherits model_config verbatim.
# Only markers pointing to the immediate parent block tag inheritance.
_TAG_PARENT_JOIN = f"""
    JOIN sessions parent ON parent.id = child.parent_session_id
      AND parent.end_reason = 'compression'
      AND COALESCE({_sql_json_extract('child.model_config', '$._branched_from')}, '') != parent.id
      AND COALESCE({_sql_json_extract('child.model_config', '$._delegate_from')}, '') != parent.id
      AND COALESCE({_sql_json_extract('child.model_config', '$._reset_from')}, '') != parent.id
      AND COALESCE(child.source, '') != 'tool'
"""


def normalize_session_tag(tag: str) -> str:
    """Keep display spelling/case; reject controls before trimming whitespace."""
    if not isinstance(tag, str) or any(unicodedata.category(c) == 'Cc' for c in tag):
        raise ValueError("tag must be a string without control characters")
    tag = tag.strip()
    if not tag or len(tag) > 64:
        raise ValueError("tag must contain 1 to 64 characters after trimming")
    return tag


def _tag_roots(conn, session_ids: list[str]) -> dict[str, str]:
    roots = {}
    # Bound SQL bind variables, not the number of sessions or tags.
    for start in range(0, len(session_ids), 400):
        ids = session_ids[start:start + 400]
        rows = conn.execute(f"""
            WITH RECURSIVE ancestors(session_id, id) AS (
                SELECT id, id FROM sessions WHERE id IN ({','.join('?' for _ in ids)})
                UNION
                SELECT a.session_id, parent.id FROM ancestors a
                JOIN sessions child ON child.id = a.id
                {_TAG_PARENT_JOIN}
            )
            SELECT a.session_id, a.id FROM ancestors a
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions child {_TAG_PARENT_JOIN} WHERE child.id = a.id
            )
        """, ids).fetchall()
        roots.update((row[0], row[1]) for row in rows)
    return roots


class SessionTagsMixin:
    """The DB path is the profile boundary; no process-global catalogue/cache."""

    def list_session_tags(self) -> list[str]:
        with self._read_ctx() as conn:
            # Read-only browsing of a profile not yet opened by this version cannot migrate it.
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'session_tag_catalog'").fetchone():
                return []
            return [row[0] for row in conn.execute("SELECT name FROM session_tag_catalog ORDER BY name")]

    def get_session_tags_batch(self, session_ids: list[str]) -> dict[str, list[str]]:
        result = {sid: [] for sid in session_ids}
        if not result:
            return result
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'session_tags'").fetchone():
                return result
            roots = _tag_roots(conn, list(result))
            by_root: dict[str, list[str]] = {root: [] for root in roots.values()}
            ids = list(by_root)
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                for root, tag in conn.execute(
                    f"SELECT session_id, tag FROM session_tags WHERE session_id IN ({','.join('?' for _ in chunk)}) ORDER BY tag",
                    chunk,
                ):
                    by_root[root].append(tag)
            for sid, root in roots.items():
                result[sid] = list(by_root[root])
        return result

    def get_session_tags(self, session_id: str) -> list[str]:
        return self.get_session_tags_batch([session_id])[session_id]

    def set_session_tag(self, session_id: str, tag: str, assigned: bool) -> list[str]:
        tag = normalize_session_tag(tag)
        if not isinstance(assigned, bool):
            raise ValueError("assigned must be a boolean")

        def _do(conn):
            # Resolve and mutate under the same write transaction as compression publication.
            root = _tag_roots(conn, [session_id]).get(session_id)
            if root is None:
                raise ValueError("session not found or invalid compression lineage")
            if assigned:
                conn.execute("INSERT OR IGNORE INTO session_tag_catalog(name) VALUES (?)", (tag,))
                conn.execute("INSERT OR IGNORE INTO session_tags(session_id, tag) VALUES (?, ?)", (root, tag))
            else:
                conn.execute("DELETE FROM session_tags WHERE session_id = ? AND tag = ?", (root, tag))
            return [row[0] for row in conn.execute(
                "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag", (root,))]
        return self._execute_write(_do)
