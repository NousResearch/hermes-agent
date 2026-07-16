"""Native PostgreSQL SessionDB search operations."""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from state_store.postgres.search_ddl import (
    POSTGRES_SEARCH_CAPABILITY,
    SEARCH_DOCUMENT_EXPRESSION,
    SEARCH_REINDEX_STATEMENTS,
)


class PostgresSearchOperations:
    """Search mixin for a ``PostgresSessionDBBase`` composition."""

    _MAX_SEARCH_QUERY_CHARS = 4_096
    _MAX_SEARCH_OFFSET = 1_000_000
    _MAX_SNIPPET_CHARS = 512
    _MAX_CONTEXT_CHARS = 200

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Return whether text contains CJK, Japanese, or Korean characters."""

        return any(
            (
                0x4E00 <= ord(character) <= 0x9FFF
                or 0x3400 <= ord(character) <= 0x4DBF
                or 0x20000 <= ord(character) <= 0x2A6DF
                or 0x3000 <= ord(character) <= 0x303F
                or 0x3040 <= ord(character) <= 0x309F
                or 0x30A0 <= ord(character) <= 0x30FF
                or 0xAC00 <= ord(character) <= 0xD7AF
            )
            for character in text
        )

    @staticmethod
    def _like_pattern(value: str) -> str:
        """Escape wildcard syntax before passing a substring pattern as data."""

        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    def _search_page(self, limit: int, offset: int) -> Optional[tuple[int, int]]:
        """Validate and cap page inputs before they reach a database query."""

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or offset > self._MAX_SEARCH_OFFSET
        ):
            return None
        return min(limit, self._MAX_READ_ROWS), offset

    @staticmethod
    def _filter_values(values: Optional[List[str]]) -> Optional[list[str]]:
        """Copy list-like filters without interpolating their values into SQL."""

        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            return None
        return [str(value) for value in values]

    @staticmethod
    def _placeholders(values: Sequence[str]) -> str:
        return ", ".join("%s" for _ in values)

    def _message_filter_sql(
        self,
        *,
        source_filter: Optional[List[str]],
        exclude_sources: Optional[List[str]],
        role_filter: Optional[List[str]],
        include_inactive: bool,
    ) -> Optional[tuple[list[str], list[str]]]:
        """Build only fixed predicate templates and their bound values."""

        clauses: list[str] = []
        params: list[str] = []
        if not include_inactive:
            clauses.append("(m.active = 1 OR m.compacted = 1)")

        sources = self._filter_values(source_filter)
        if source_filter is not None:
            if not sources:
                return None
            clauses.append(f"s.source IN ({self._placeholders(sources)})")
            params.extend(sources)

        excluded = self._filter_values(exclude_sources)
        if excluded:
            clauses.append(f"s.source NOT IN ({self._placeholders(excluded)})")
            params.extend(excluded)

        roles = self._filter_values(role_filter)
        if roles:
            clauses.append(f"m.role IN ({self._placeholders(roles)})")
            params.extend(roles)
        return clauses, params

    @staticmethod
    def _message_sort(sort: Optional[str]) -> tuple[str, str]:
        """Return fixed safe ordering clauses for inner and outer queries."""

        normalized = sort.strip().lower() if isinstance(sort, str) else None
        if normalized == "newest":
            return (
                "m.timestamp DESC, rank DESC, m.id DESC",
                "matched.timestamp DESC, matched.rank DESC, matched.id DESC",
            )
        if normalized == "oldest":
            return (
                "m.timestamp ASC, rank DESC, m.id ASC",
                "matched.timestamp ASC, matched.rank DESC, matched.id ASC",
            )
        return (
            "rank DESC, m.timestamp DESC, m.id DESC",
            "matched.rank DESC, matched.timestamp DESC, matched.id DESC",
        )

    @classmethod
    def _context_select_sql(cls, outer_order: str) -> str:
        """Attach one bounded message before, at, and after every match."""

        return f"""
            SELECT
                matched.id,
                matched.session_id,
                matched.role,
                matched.snippet,
                matched.timestamp,
                matched.tool_name,
                matched.source,
                matched.model,
                matched.session_started,
                COALESCE(context_rows.context, '[]'::jsonb) AS context
            FROM matched
            LEFT JOIN LATERAL (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'role', context_message.role,
                        'content', LEFT(
                            COALESCE(context_message.content, ''),
                            {cls._MAX_CONTEXT_CHARS}
                        )
                    )
                    ORDER BY context_message.sequence
                ) AS context
                FROM (
                    SELECT *
                    FROM (
                        SELECT
                            prior.role,
                            prior.content,
                            0 AS sequence
                        FROM messages AS prior
                        WHERE prior.session_id = matched.session_id
                          AND (
                              prior.timestamp < matched.timestamp
                              OR (
                                  prior.timestamp = matched.timestamp
                                  AND prior.id < matched.id
                              )
                          )
                        ORDER BY prior.timestamp DESC, prior.id DESC
                        LIMIT 1
                    ) AS before_match
                    UNION ALL
                    SELECT
                        current_message.role,
                        current_message.content,
                        1 AS sequence
                    FROM messages AS current_message
                    WHERE current_message.id = matched.id
                    UNION ALL
                    SELECT *
                    FROM (
                        SELECT
                            following.role,
                            following.content,
                            2 AS sequence
                        FROM messages AS following
                        WHERE following.session_id = matched.session_id
                          AND (
                              following.timestamp > matched.timestamp
                              OR (
                                  following.timestamp = matched.timestamp
                                  AND following.id > matched.id
                              )
                          )
                        ORDER BY following.timestamp ASC, following.id ASC
                        LIMIT 1
                    ) AS after_match
                ) AS context_message
            ) AS context_rows ON TRUE
            ORDER BY {outer_order}
        """

    def _lexical_message_sql(self, *, where_sql: str, inner_order: str, outer_order: str) -> str:
        return f"""
            WITH parsed_input AS (
                SELECT
                    websearch_to_tsquery('simple'::regconfig, %s) AS web_query,
                    plainto_tsquery('simple'::regconfig, %s) AS plain_query
            ),
            parsed AS (
                SELECT CASE
                    WHEN numnode(web_query) > 0 THEN web_query
                    ELSE plain_query
                END AS query
                FROM parsed_input
            ),
            matched AS (
                SELECT
                    m.id,
                    m.session_id,
                    m.role,
                    LEFT(
                        ts_headline(
                            'simple'::regconfig,
                            COALESCE(m.content, ''),
                            parsed.query,
                            'StartSel=>>>, StopSel=<<<, MaxWords=40, MinWords=10, MaxFragments=1'
                        ),
                        {self._MAX_SNIPPET_CHARS}
                    ) AS snippet,
                    m.timestamp,
                    m.tool_name,
                    s.source,
                    s.model,
                    s.started_at AS session_started,
                    ts_rank(m.search_vector, parsed.query) AS rank
                FROM messages AS m
                JOIN sessions AS s ON s.id = m.session_id
                CROSS JOIN parsed
                WHERE m.search_vector @@ parsed.query
                  AND {where_sql}
                ORDER BY {inner_order}
                LIMIT %s OFFSET %s
            )
            {self._context_select_sql(outer_order)}
        """

    def _substring_message_sql(
        self,
        *,
        where_sql: str,
        inner_order: str,
        outer_order: str,
    ) -> str:
        return f"""
            WITH matched AS (
                SELECT
                    m.id,
                    m.session_id,
                    m.role,
                    LEFT(COALESCE(m.content, ''), {self._MAX_SNIPPET_CHARS}) AS snippet,
                    m.timestamp,
                    m.tool_name,
                    s.source,
                    s.model,
                    s.started_at AS session_started,
                    GREATEST(
                        similarity(COALESCE(m.content, ''), %s),
                        similarity(COALESCE(m.tool_name, ''), %s),
                        similarity(COALESCE(m.tool_calls, ''), %s)
                    ) AS rank
                FROM messages AS m
                JOIN sessions AS s ON s.id = m.session_id
                WHERE (
                    m.content ILIKE %s ESCAPE E'\\\\'
                    OR m.tool_name ILIKE %s ESCAPE E'\\\\'
                    OR m.tool_calls ILIKE %s ESCAPE E'\\\\'
                )
                  AND {where_sql}
                ORDER BY {inner_order}
                LIMIT %s OFFSET %s
            )
            {self._context_select_sql(outer_order)}
        """

    def _search_ready(self) -> bool:
        return bool(self.capabilities.get(POSTGRES_SEARCH_CAPABILITY, False))

    def _normalize_search_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Defensively enforce preview-only output at the Python boundary."""

        results: List[Dict[str, Any]] = []
        for row in rows:
            row.pop("content", None)
            snippet = row.get("snippet")
            row["snippet"] = "" if snippet is None else str(snippet)[:self._MAX_SNIPPET_CHARS]

            context = row.get("context", [])
            if isinstance(context, str):
                try:
                    context = self._json_loads(context, default=[])
                except (TypeError, ValueError):
                    context = []
            normalized_context: list[dict[str, str]] = []
            if isinstance(context, (list, tuple)):
                for item in context[:3]:
                    if not isinstance(item, Mapping):
                        continue
                    content = item.get("content")
                    normalized_context.append(
                        {
                            "role": str(item.get("role") or ""),
                            "content": "" if content is None else str(content)[:self._MAX_CONTEXT_CHARS],
                        }
                    )
            row["context"] = normalized_context
            results.append(row)
        return results

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search messages with native lexical or CJK substring indexes."""

        if (
            not self._search_ready()
            or not isinstance(query, str)
            or "\x00" in query
        ):
            return []
        needle = query.strip()
        if not needle or len(needle) > self._MAX_SEARCH_QUERY_CHARS:
            return []
        page = self._search_page(limit, offset)
        if page is None:
            return []
        filters = self._message_filter_sql(
            source_filter=source_filter,
            exclude_sources=exclude_sources,
            role_filter=role_filter,
            include_inactive=include_inactive,
        )
        if filters is None:
            return []
        clauses, filter_params = filters
        where_sql = " AND ".join(clauses) if clauses else "TRUE"
        inner_order, outer_order = self._message_sort(sort)
        page_limit, page_offset = page

        if self._contains_cjk(needle):
            pattern = self._like_pattern(needle)
            sql = self._substring_message_sql(
                where_sql=where_sql,
                inner_order=inner_order,
                outer_order=outer_order,
            )
            params: list[Any] = [
                needle,
                needle,
                needle,
                pattern,
                pattern,
                pattern,
                *filter_params,
                page_limit,
                page_offset,
            ]
        else:
            sql = self._lexical_message_sql(
                where_sql=where_sql,
                inner_order=inner_order,
                outer_order=outer_order,
            )
            params = [needle, needle, *filter_params, page_limit, page_offset]

        return self._normalize_search_rows(
            self._read_many(sql, tuple(params), limit=page_limit)
        )

    def search_sessions_by_id(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search session ids with exact, prefix, then substring ranking."""

        if not isinstance(query, str) or "\x00" in query:
            return []
        needle = query.strip()
        page = self._search_page(limit, 0)
        if not needle or page is None:
            return []
        substring = self._like_pattern(needle)
        prefix = f"{self._like_pattern(needle)[1:-1]}%"
        archived_clause = "" if include_archived else "AND s.archived = 0"
        sql = f"""
            SELECT
                s.*,
                COALESCE(
                    (
                        SELECT MAX(message.timestamp)
                        FROM messages AS message
                        WHERE message.session_id = s.id
                    ),
                    s.started_at
                ) AS last_active,
                CASE
                    WHEN lower(s.id) = lower(%s) THEN 0
                    WHEN s.id ILIKE %s ESCAPE E'\\\\' THEN 1
                    ELSE 2
                END AS _id_match_rank
            FROM sessions AS s
            WHERE s.id ILIKE %s ESCAPE E'\\\\'
              {archived_clause}
            ORDER BY _id_match_rank, last_active DESC, s.started_at DESC, s.id DESC
            LIMIT %s
        """
        rows = self._read_many(
            sql,
            (needle, prefix, substring, page[0]),
            limit=page[0],
        )
        for row in rows:
            row.pop("_id_match_rank", None)
        return rows

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions ordered by latest message activity."""

        page = self._search_page(limit, offset)
        if page is None:
            return []
        source_clause = "WHERE s.source = %s" if source else ""
        params: tuple[Any, ...]
        if source:
            params = (source, page[0], page[1])
        else:
            params = (page[0], page[1])
        sql = f"""
            SELECT
                s.*,
                COALESCE(
                    (
                        SELECT MAX(message.timestamp)
                        FROM messages AS message
                        WHERE message.session_id = s.id
                    ),
                    s.started_at
                ) AS last_active
            FROM sessions AS s
            {source_clause}
            ORDER BY last_active DESC, s.started_at DESC, s.id DESC
            LIMIT %s OFFSET %s
        """
        return self._read_many(sql, params, limit=page[0])

    def optimize_fts(self) -> int:
        """Rebuild derived PostgreSQL search indexes for writable stores."""

        if self.read_only or not self._search_ready():
            return 0

        def operation(connection: Any) -> int:
            for statement in SEARCH_REINDEX_STATEMENTS:
                connection.execute(statement)
            return len(SEARCH_REINDEX_STATEMENTS)

        return self._run(operation, read_only=False)
