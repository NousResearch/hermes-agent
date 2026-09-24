"""PostgreSQL message search using the SessionDB result and visibility contracts."""

import re
from typing import Any, Collection, Dict, List, Optional

from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS
from hermes_state_postgres_schema import SEARCH_CONTENT_CHARS, SEARCH_TOOL_CALLS_CHARS, SEARCH_TOOL_NAME_CHARS
from hermes_state_search import _search_filter_clauses, _search_select_sql


_QUERY_TOKEN_RE = re.compile(r'"[^"]*"\*?|\S+')
_SEARCH_OVERFLOW_SQL = (
    f"((m.role <> 'tool' AND length(m.content) > {SEARCH_CONTENT_CHARS}) "
    f"OR length(m.tool_calls) > {SEARCH_TOOL_CALLS_CHARS} OR length(m.tool_name) > {SEARCH_TOOL_NAME_CHARS})"
)
_SEARCH_TEXT_SQL = (
    "left(COALESCE(m.content, ''), CASE WHEN m.role = 'tool' "
    f"THEN {FTS_TOOL_CONTENT_PREFIX_CHARS} ELSE {SEARCH_CONTENT_CHARS} END) || ' ' || "
    f"left(COALESCE(m.tool_name, ''), {SEARCH_TOOL_NAME_CHARS}) || ' ' || "
    f"left(COALESCE(m.tool_calls, ''), {SEARCH_TOOL_CALLS_CHARS})"
)


def _compile_tsquery(query: str) -> tuple[str, list[str]]:
    """Bind phrases separately so query text cannot become PostgreSQL query syntax."""
    groups: list[list[tuple[str, str, bool]]] = [[]]
    negate_next = False
    for token in _QUERY_TOKEN_RE.findall(query):
        operator = token.upper()
        if operator == "OR":
            if groups[-1]:
                groups.append([])
            negate_next = False
            continue
        if operator in {"AND", "NEAR"}:
            continue
        if operator == "NOT":
            negate_next = True
            continue
        term = token.rstrip("*").strip('"').strip()
        if not term:
            continue
        expression = "phraseto_tsquery('simple', ?)"
        if token.endswith("*"):
            expression = f"to_tsquery('simple', NULLIF({expression}::text, '') || ':*')"
        if negate_next:
            expression = f"!!({expression})"
        groups[-1].append((expression, term, negate_next))
        negate_next = False
    groups = [group for group in groups if any(not negated for _, _, negated in group)]
    sql = " || ".join(f"({' && '.join(expression for expression, _, _ in group)})" for group in groups)
    return sql, [term for group in groups for _, term, _ in group]


class SessionPostgresSearchMixin:
    """Replace SQLite's derived indexes while sharing result hydration and filters."""

    def _describe_search_path(self, query: str) -> str:
        return "postgres_like" if self._contains_cjk(query or "") else "postgres_fts"

    def _search_messages_impl(
        self, query: str, source_filter: List[str] = None, exclude_sources: List[str] = None,
        role_filter: List[str] = None, limit: int = 20, offset: int = 0, sort: str = None,
        include_inactive: bool = False, fields: Optional[Collection[str]] = None,
        after_ts: Optional[int] = None, before_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        result_fields = self._search_message_fields(fields)
        query = self._sanitize_fts5_query(query or "")
        if not query or limit == 0 or source_filter == []:
            return []
        filters = dict(include_inactive=include_inactive, source_filter=source_filter,
                       exclude_sources=exclude_sources or None, role_filter=role_filter,
                       after_ts=after_ts, before_ts=before_ts)
        route = dict(limit=None if limit < 0 else limit, offset=max(0, offset), sort=sort, **filters)
        if self._contains_cjk(query) or (role_filter and "tool" in role_filter):
            matches = self._search_messages_like_fallback(query, **route)
        else:
            matches = self._search_messages_postgres(query, **route)
            if not matches:
                matches = self._search_messages_like_fallback(query, **route)
        return self._finalize_search_matches(matches, result_fields=result_fields)

    def _postgres_like_query(self, query: str, *, full_tool_content: bool = False):
        predicate, params, snippet_term = self._compile_like_boolean_query(query)
        predicate = predicate.replace(" LIKE ?", " ILIKE ?")
        if not full_tool_content:
            content = ("CASE WHEN m.role = 'tool' "
                       f"THEN left(m.content, {FTS_TOOL_CONTENT_PREFIX_CHARS}) ELSE m.content END")
            predicate = predicate.replace("m.content", content)
        return predicate, params, snippet_term

    def _search_messages_postgres(self, query: str, *, limit, offset, sort, **filters):
        tsquery_sql, query_params = _compile_tsquery(query)
        if not tsquery_sql:
            return []
        overflow_predicate, overflow_params, snippet_term = self._postgres_like_query(query)
        if not overflow_predicate:
            return []
        where: list[str] = []
        filter_params: list = []
        _search_filter_clauses(where, filter_params, **filters)
        candidates_sql = f"""WITH search_query AS (SELECT {tsquery_sql} AS query),
            candidates AS (
                SELECT m.id FROM messages m CROSS JOIN search_query q
                WHERE m.search_vector @@ q.query AND NOT COALESCE({_SEARCH_OVERFLOW_SQL}, FALSE)
                UNION ALL
                SELECT m.id FROM messages m
                WHERE {_SEARCH_OVERFLOW_SQL} AND ({overflow_predicate})
            ) """
        rank_order = "ts_rank_cd(m.search_vector, q.query) DESC, m.id"
        normalized_sort = sort.strip().lower() if isinstance(sort, str) else None
        order_by = {"newest": "m.timestamp DESC, ", "oldest": "m.timestamp ASC, "}.get(normalized_sort, "")
        snippet_sql = (
            f"CASE WHEN {_SEARCH_OVERFLOW_SQL} "
            "THEN substr(m.content, GREATEST(1, strpos(lower(m.content), lower(?)) - 40), 120) "
            f"ELSE ts_headline('simple', {_SEARCH_TEXT_SQL}, q.query, "
            "'StartSel=>>>, StopSel=<<<, MaxWords=40, MinWords=10, MaxFragments=1') END AS snippet"
        )
        sql = candidates_sql + _search_select_sql(
            snippet_sql, "candidates c JOIN messages m ON m.id = c.id CROSS JOIN search_query q",
            where, f"ORDER BY {order_by}{rank_order}", "LIMIT ? OFFSET ?",
        )
        params = [*query_params, *overflow_params, snippet_term, *filter_params, limit, offset]
        return [dict(row) for row in self._read_all(sql, params)]

    def _search_messages_like_fallback(self, query: str, *, limit, offset, sort, **filters):
        predicate, params, snippet_term = self._postgres_like_query(
            query, full_tool_content="tool" in (filters.get("role_filter") or []))
        if not predicate or snippet_term is None:
            return []
        where = [f"({predicate})"]
        _search_filter_clauses(where, params, **filters)
        order = "ASC" if isinstance(sort, str) and sort.strip().lower() == "oldest" else "DESC"
        snippet_sql = "substr(m.content, GREATEST(1, strpos(lower(m.content), lower(?)) - 40), 120) AS snippet"
        sql = _search_select_sql(snippet_sql, "messages m", where,
                                 f"ORDER BY m.timestamp {order}, m.id {order}", "LIMIT ? OFFSET ?")
        return [dict(row) for row in self._read_all(sql, [snippet_term, *params, limit, offset])]

    def fts_rebuild_status(self):
        return None

    def fts_cjk_rebuild_status(self):
        return None

    def fts_rebuild_step(self) -> bool:
        return False

    def fts_cjk_rebuild_step(self) -> bool:
        return False

    def fts_optimize_available(self) -> bool:
        return False

    def optimize_fts(self) -> int:
        """Flush the GIN pending list; PostgreSQL owns vacuum and background maintenance."""
        if self.read_only:
            return 0
        self._execute_write(lambda conn: conn.execute(
            "SELECT pg_catalog.gin_clean_pending_list('idx_messages_search_vector'::regclass)"))
        return 1

    def rebuild_fts(self) -> int:
        """Rebuild the native index from stored vectors in one PostgreSQL transaction."""
        if self.read_only:
            return 0
        self._execute_write(lambda conn: conn.execute("REINDEX INDEX idx_messages_search_vector"))
        return 1

    def optimize_fts_storage(self, *, progress_cb=None, vacuum: bool = True) -> Dict[str, Any]:
        if self.read_only:
            return {"ok": False, "reason": "read_only"}
        self.optimize_fts()
        if progress_cb is not None:
            progress_cb({"phase": "done", "percent": 100, "indexed": 0, "total": 0})
        return {"ok": True, "vacuumed": None}
