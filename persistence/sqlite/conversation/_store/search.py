"""Message search over the FTS5, trigram and CJK indexes with the LIKE fallbacks, and the index rebuild."""


import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Collection, Optional
from persistence.sqlite._state.errors import is_malformed_db_error
from persistence.sqlite._state.common import fts_rebuild_admission
from sqlite_support import routed_sessions_setting
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema import _enter_fts_fail_open, _present_fts_tables
from persistence.sqlite._schema_ddl import (
    FTS_STALE_KEY, FTS_TOOL_CONTENT_PREFIX_CHARS, FTS_TRIGRAM_EXCLUDED_SOURCES, fts_trigram_session_sql,
)
from persistence.sqlite.conversation._store.sql_common import MAX_FTS5_QUERY_CHARS, escape_like as _escape_like


logger = logging.getLogger("hermes_state")


@dataclass
class _BackfillProgress:
    """How many messages a backfill pass (``backfill`` = base + trigram, ``backfill_cjk``) has
    indexed out of the messages at or below its high water. Counts rows, never ids: a pruned
    store's id gaps would otherwise read 0% then jump to 99%. The step engine adds the rows each
    window covered (:meth:`SessionSearchMixin._rebuild_step`)."""

    phase: str
    total: int = 0
    indexed: int = 0

    def info(self) -> dict[str, Any]:
        done, total = self.indexed, self.total
        return {"phase": self.phase, "percent": min(100, 100 * done // total) if total else 100,
                "indexed": min(done, total), "total": total}


def _search_slow_ms() -> float:
    """``sessions.search_slow_ms`` for the served profile (default 1000; 0 logs every call)."""
    value = routed_sessions_setting("search_slow_ms", "HERMES_SEARCH_SLOW_MS")
    try:
        return 1000.0 if value is None or str(value).strip() == "" else float(value)
    except (TypeError, ValueError):
        return 1000.0


_FTS5_SPECIAL_CHARS = '+{}():"^@/#&|~[]<>,;!?$=\\\''


_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")


_FTS_OPERATORS = frozenset({"AND", "OR", "NOT"})


_LIKE_SKIP_TOKENS = _FTS_OPERATORS | {"NEAR"}


_LIKE_TOKEN_RE = re.compile(r'"[^"]+"|\S+')


_QUOTED_PHRASE_RE = re.compile(r'"[^"]*"')


_SEARCH_SELECT_TAIL = "m.timestamp, m.tool_name, s.source, s.model, s.started_at AS session_started"


_LIKE_SNIPPET_SQL = "substr(m.content, max(1, instr(m.content, ?) - 40), 120) AS snippet"


_LIKE_ANY_COLUMN_SQL = (
    "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"
)


_LIKE_COALESCED_COLUMN_SQL = (
    "(COALESCE(m.content, '') LIKE ? ESCAPE '\\' OR "
    "COALESCE(m.tool_name, '') LIKE ? ESCAPE '\\' OR "
    "COALESCE(m.tool_calls, '') LIKE ? ESCAPE '\\')"
)


_FTS_ORDER_BY = {"newest": "ORDER BY m.timestamp DESC, rank", "oldest": "ORDER BY m.timestamp ASC, rank"}


_CONTEXT_WINDOW_SQL = """WITH target AS (
    SELECT session_id, timestamp, id FROM messages WHERE id IN ({ids})
)
SELECT t.id AS match_id, m.role, m.content
FROM target t JOIN messages m ON m.id IN (
    t.id,
    (SELECT p.id FROM messages p
     WHERE p.session_id = t.session_id AND (p.timestamp, p.id) < (t.timestamp, t.id)
     ORDER BY p.timestamp DESC, p.id DESC LIMIT 1),
    (SELECT n.id FROM messages n
     WHERE n.session_id = t.session_id AND (n.timestamp, n.id) > (t.timestamp, t.id)
     ORDER BY n.timestamp, n.id LIMIT 1)
)
ORDER BY t.id, m.timestamp, m.id"""


_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x3000, 0x303F), (0x3040, 0x309F),
    (0x30A0, 0x30FF), (0xAC00, 0xD7AF),
)


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _non_operator_tokens(raw_query: str) -> list[str]:
    return [t for t in raw_query.split() if t.upper() not in _FTS_OPERATORS]


def _quote_fts_tokens(raw_query: str) -> str:
    """Quote each non-operator token (neutralising FTS5 special characters), keeping AND/OR/NOT."""
    return " ".join(
        tok if tok.upper() in _FTS_OPERATORS else '"' + tok.replace('"', '""') + '"' for tok in raw_query.split()
    )


def _like_params(term: str) -> list[str]:
    """One ``%term%`` bind per column of ``_LIKE_ANY_COLUMN_SQL``."""
    return [f"%{_escape_like(term)}%"] * 3


def _strip_cjk_wildcards(raw_query: str) -> str:
    """Drop the trailing prefix wildcard callers append for ASCII ("nimb" -> "nimb*").

    None of the CJK routes can honour that star: the bigram and trigram routes
    quote every token before MATCH (so ``*`` matches a literal asterisk) and
    LIKE has no ``*`` wildcard at all (only ``%``/``_``). Left in place, every
    CJK search arriving from the web/desktop search box — which appends the
    star to each unquoted token so partial English words match — searches for
    a term ending in a literal ``*`` and returns nothing (#90636). Only
    TRAILING stars go: a star written inside a quoted phrase is the user's
    own text, and a token that is ALL stars keeps its original form so it
    cannot degrade to a match-everything empty term.
    """
    if "*" not in raw_query:
        return raw_query
    stripped: list[str] = []
    for token in raw_query.split():
        if token.upper() in _FTS_OPERATORS:
            stripped.append(token)
        else:
            stripped.append(token.rstrip("*") or token)
    return " ".join(stripped) or raw_query


def _flatten_text(decoded: Any) -> str:
    """Multimodal part list -> joined text (or the placeholder); str passes through; else ''."""
    if isinstance(decoded, list):
        parts = [p.get("text", "") for p in decoded if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(t for t in parts if t).strip() or "[multimodal content]"
    return decoded if isinstance(decoded, str) else ""


def _search_select_sql(snippet_sql: str, from_sql: str, where: list[str], order_by: str, limit_sql: str) -> str:
    """Result-row SELECT shared by the FTS and LIKE routes (SQL text is pinned)."""
    return f"""
            SELECT m.id, m.session_id, m.role,
                   {snippet_sql},
                   {_SEARCH_SELECT_TAIL}
            FROM {from_sql}
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by}
            {limit_sql}
        """


def _search_filter_clauses(
    where: list[str], params: list, *, include_inactive: bool, source_filter: Optional[list[str]],
    exclude_sources: Optional[list[str]], role_filter: Optional[list[str]],
    after_ts: Optional[int] = None, before_ts: Optional[int] = None) -> None:
    """Append the visibility/source/role/session-start predicates every search route shares. Live
    rows (active=1) AND compaction-archived rows (compacted=1) are discoverable; only
    rewind/undo rows (active=0, compacted=0) are hidden. ``after_ts``/``before_ts`` bound
    ``sessions.started_at`` (inclusive / exclusive) inside the query so LIMIT cannot be
    filled by out-of-window hits."""
    if not include_inactive:
        where.append("(m.active = 1 OR m.compacted = 1)")
    # display_kind="hidden" rows are model-facing scaffolding the person never saw; a hit would confuse.
    where.append("COALESCE(m.display_kind, '') <> 'hidden'")
    if source_filter is not None:
        where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
        params.extend(source_filter)
    if exclude_sources is not None:
        where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
        params.extend(exclude_sources)
    if role_filter:
        where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
        params.extend(role_filter)
    if after_ts is not None:
        where.append("s.started_at >= ?")
        params.append(int(after_ts))
    if before_ts is not None:
        where.append("s.started_at < ?")
        params.append(int(before_ts))


_SEARCH_MESSAGE_RESULT_FIELDS = (
    "id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model", "session_started", "context"
)


def _search_message_fields(fields: Optional[Collection[str]]) -> Optional[tuple[str, ...]]:
    """Validate and canonically order an optional result projection."""
    if fields is None:
        return None
    if isinstance(fields, str):
        raise TypeError("search fields must be a collection of field names, not a string")
    requested = set(fields)
    unknown = requested.difference(_SEARCH_MESSAGE_RESULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown search result field(s): {', '.join(sorted(unknown))}")
    return tuple(field for field in _SEARCH_MESSAGE_RESULT_FIELDS if field in requested)


def fts_rebuild_status(holder: StoreHolder) -> Optional[dict[str, Any]]:
    """Deferred-rebuild progress ``{"pending", "total", "indexed", "percent"}`` or None. Reads
    via the pooled reader (not get_meta/self._lock) so search never blocks on the writer."""
    return _rebuild_status(holder, "fts_rebuild")


def _rebuild_status(holder: StoreHolder, prefix: str) -> Optional[dict[str, Any]]:
    rows = holder._read_all("SELECT key, value FROM state_meta WHERE key IN (?, ?)",
                          (f"{prefix}_high_water", f"{prefix}_progress"))
    meta = {r["key"]: r["value"] for r in rows}
    high_water = meta.get(f"{prefix}_high_water")
    if high_water is None or int(high_water) <= 0:
        return None
    total, progress = int(high_water), int(meta.get(f"{prefix}_progress") or 0)
    return {"pending": True, "total": total, "indexed": progress, "percent": min(100, int(100 * progress / total))}


_BOUNDARY_SWEEP_SQL = (
    "INSERT INTO {table}(rowid, content, tool_name, tool_calls) "
    "SELECT m.id, m.content, m.tool_name, m.tool_calls FROM messages m WHERE m.id > ? AND m.id <= ? {extra}"
    "AND NOT EXISTS (SELECT 1 FROM {table}_docsize d WHERE d.id = m.id)"
)


_BASE_BOUNDARY_SWEEP_SQL = (
    "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
    "SELECT m.id, CASE WHEN m.role = 'tool' THEN substr(COALESCE(m.content, ''), 1, ?) "
    "ELSE m.content END, m.tool_name, m.tool_calls FROM messages m WHERE m.id > ? AND m.id <= ? "
    "AND NOT EXISTS (SELECT 1 FROM messages_fts_docsize d WHERE d.id = m.id)"
)


_TRIGRAM_BOUNDARY_SWEEP_SQL = (
    "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
    "SELECT m.id, m.content, m.tool_name FROM messages m JOIN sessions s ON s.id = m.session_id "
    f"WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' AND {fts_trigram_session_sql('s')} "
    "AND NOT EXISTS (SELECT 1 FROM messages_fts_trigram_docsize d WHERE d.id = m.id)"
)


_CHUNK_INSERT_SQL = (
    "INSERT INTO {table}(rowid, content, tool_name, tool_calls) "
    "SELECT id, CASE WHEN role = 'tool' "
    f"THEN substr(COALESCE(content, ''), 1, {FTS_TOOL_CONTENT_PREFIX_CHARS}) "
    "ELSE content END, tool_name, tool_calls FROM messages WHERE id > ? AND id <= ?{extra}"
)


_TRIGRAM_CHUNK_INSERT_SQL = (
    "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
    "SELECT m.id, m.content, m.tool_name FROM messages m JOIN sessions s ON s.id = m.session_id "
    f"WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' AND {fts_trigram_session_sql('s')}"
)


def _backfill_progress(holder: StoreHolder, phase: str, prefix: str) -> Optional[_BackfillProgress]:
    """Messages at or below *prefix*'s high water, and how many of them the resume marker already
    covers (one scan per pass); None when the pass has nothing pending."""
    st = _rebuild_status(holder, prefix)
    if st is None:
        return None
    with holder._read_ctx() as conn:
        # ``+id`` keeps the planner off the rowid range: counting from a covering index reads
        # far fewer pages (1.97M messages, cold: 73 s on the table b-tree, 1.9 s here).
        total, done = conn.execute("SELECT COUNT(*), COALESCE(SUM(id <= ?), 0) FROM messages WHERE +id <= ?",
                                   (st["indexed"], st["total"])).fetchone()
    return _BackfillProgress(phase=phase, total=int(total), indexed=int(done))


_OPTIMIZE_CHUNK_ATTEMPTS = 3


_VACUUM_TICK_S = 1.0


_HOLDER_SCAN_INTERVAL_S = 2.0


def get_anchored_view(
    holder: StoreHolder, session_id: str, around_message_id: int, window: int = 5, bookend: int = 3,
    keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> dict[str, Any]:
    """Anchored window (``get_messages_around``) plus session bookends, so one call yields the
    goal and the resolution of a long session. ``window`` is filtered to ``keep_roles``
    (None disables) EXCEPT the anchor; ``bookend_start`` / ``bookend_end`` are the
    first/last ``bookend`` non-empty-content messages with ids strictly outside the
    window (empty when it overlaps the head/tail). Empty result when the anchor isn't
    in the session."""
    bookend = max(bookend, 0)
    primitive = get_messages_around(holder, session_id, around_message_id, window=window)
    window_rows = primitive["window"]
    if not window_rows:
        return {"window": [], "messages_before": 0, "messages_after": 0, "bookend_start": [], "bookend_end": []}

    filtered_window = window_rows
    if keep_roles is not None:
        keep_set = set(keep_roles)
        filtered_window = [m for m in window_rows if m.get("id") == around_message_id or m.get("role") in keep_set]
    bookend_start_rows: list[Any] = []
    bookend_end_rows: list[Any] = []
    if bookend > 0:
        role_clause = "" if keep_roles is None else f" AND role IN ({','.join('?' for _ in keep_roles)})"
        role_params = [] if keep_roles is None else list(keep_roles)
        with holder._read_ctx() as conn:
            def _bookend(op: str, boundary_id: int, order: str):
                return conn.execute(
                    f"SELECT * FROM messages "
                    f"WHERE session_id = ? AND id {op} ?{role_clause} "
                    f"AND length(content) > 0 "
                    f"ORDER BY id {order} LIMIT ?",
                    (session_id, boundary_id, *role_params, bookend),
                ).fetchall()
            bookend_start_rows = _bookend("<", window_rows[0]["id"], "ASC")
            # End rows come back DESC for the LIMIT cap; flip to ASC.
            bookend_end_rows = list(reversed(_bookend(">", window_rows[-1]["id"], "DESC")))

    def _hydrate(row) -> dict[str, Any]:
        return _row_to_message_dict(holder, row, warn_context="get_anchored_view", summary_flag=False)
    return {
        "window": filtered_window, "messages_before": primitive["messages_before"],
        "messages_after": primitive["messages_after"],
        "bookend_start": [_hydrate(r) for r in bookend_start_rows],
        "bookend_end": [_hydrate(r) for r in bookend_end_rows],
    }


def _sanitize_fts5_query(query: str) -> str:
    """Sanitize user input for FTS5 MATCH (raw special characters raise): preserve paired
    quoted phrases, strip unmatched special characters, and quote hyphenated/dotted
    terms so FTS5 matches them as phrases (``chat-send``, ``P2.2``, ``my-app.config.ts``)."""
    # Cap before any regex processing so adversarial input stays bounded.
    query = query[:MAX_FTS5_QUERY_CHARS]

    # 1. Protect balanced quoted phrases via numbered placeholders (``"[^"]*"`` pairs
    # left-to-right without backtracking); a leftover unmatched quote becomes whitespace.
    _quoted_parts: list = []

    def _hold(m: re.Match[str]) -> str:
        _quoted_parts.append(m.group(0))
        return f"\x00Q{len(_quoted_parts) - 1}\x00"

    sanitized = _QUOTED_PHRASE_RE.sub(_hold, query).replace('"', " ")

    # 2. Strip FTS5-special characters (an unquoted ``TODO: fix`` parses as
    # ``column:term``). ``%`` is only spared for the CJK LIKE fallback.
    sanitized = _FTS5_SPECIAL_RE.sub(" ", sanitized)
    if "%" in sanitized and not _contains_cjk(sanitized):
        sanitized = sanitized.replace("%", " ")
    # 3. Collapse repeated * and drop leading * (prefix needs a char).
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
    # 4. Drop dangling boolean operators at start/end (syntax errors).
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
    sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())
    # 5. Quote dotted/hyphenated/underscored terms in ONE pass (sequential passes
    # double-quote ``my-app.config``).
    sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)
    # 6. Restore preserved quoted phrases.
    for i, quoted in enumerate(_quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)
    return sanitized.strip()


def _contains_cjk(text: str) -> bool:
    return any(_is_cjk(ord(ch)) for ch in text)


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if _is_cjk(ord(ch)))


def _has_lone_cjk_run(query: str) -> bool:
    """True when any maximal CJK run is a single char: the cjk-bigram index stores unigrams
    only for isolated chars, so such a term can't match inside longer runs — keep LIKE."""
    run = 0
    for ch in query:
        if _is_cjk(ord(ch)):
            run += 1
        else:
            if run == 1:
                return True
            run = 0
    return run == 1


def _or_relaxed_query(query: str) -> Optional[str]:
    """The sanitized implicit-AND query rewritten as an any-term OR query, or ``None`` when
    relaxation does not apply: fewer than two searchable units (a single term cannot relax)
    or explicit ``OR``/``NOT`` (the caller expressed exact semantics). Quoted phrases stay
    whole units: ``"docker networking" tls`` -> ``"docker networking" OR tls``."""
    units: list[str] = []
    for raw_token in _LIKE_TOKEN_RE.findall(query):
        upper = raw_token.upper()
        if upper in {"OR", "NOT"}:
            return None
        if upper != "AND":
            units.append(raw_token)
    return " OR ".join(units) if len(units) >= 2 else None


def _trigram_eligible_tokens(query: str) -> bool:
    """True when every non-operator token is >=3 chars: a shorter token produces no
    trigrams, and with FTS5's implicit AND one such token empties the whole MATCH."""
    tokens = _non_operator_tokens(query.strip('"').strip())
    return bool(tokens) and all(len(t) >= 3 for t in tokens)


def _has_short_cjk_token(raw_query: str) -> bool:
    """True when any non-operator CJK token has fewer than 3 CJK chars — the trigram
    tokenizer needs >=3 per token, so such a query must take the LIKE route."""
    return any(_count_cjk(t) < 3 for t in _non_operator_tokens(raw_query) if _contains_cjk(t))


def _trigram_route_ok(holder: StoreHolder, raw_query: str) -> bool:
    """Per-token CJK length gate for the trigram index: ``广西 OR 桂林 OR 漓江`` has 6
    CJK chars total but 2 per token, so trigram returns 0."""
    return (_count_cjk(raw_query) >= 3 and not _has_short_cjk_token(raw_query)
            and holder._trigram_available)


def _describe_search_path(holder: StoreHolder, query: str) -> str:
    """Best-effort name of the routing path a query takes (log-only)."""
    try:
        if holder._fts_stale:
            return "like_scan_fts_stale"
        sanitized = _sanitize_fts5_query(query or "")
        if not sanitized:
            return "empty"
        if not _contains_cjk(sanitized):
            return "fts5"
        raw = sanitized.strip('"').strip()
        if holder._fts_cjk_available and not _has_lone_cjk_run(raw):
            return "fts_cjk"
        return "trigram" if _trigram_route_ok(holder, raw) else "like_scan"
    except Exception:  # health: allow BLE001 -- the route is diagnostic; a failure reads unknown
        return "unknown"


def _fts_match_sql(table: str, match_query: str, order_by_sql: str, *, limit: int, offset: int,
                   **filters) -> tuple[str, list]:
    """MATCH query + params against one FTS5 index joined to messages/sessions."""
    where = [f"{table} MATCH ?"]
    params: list = [match_query]
    _search_filter_clauses(where, params, **filters)
    params.extend([limit, offset])
    sql = _search_select_sql(
        f"snippet({table}, -1, '>>>', '<<<', '...', 40) AS snippet",
        f"{table}\n            JOIN messages m ON m.id = {table}.rowid", where, order_by_sql, "LIMIT ? OFFSET ?",
    )
    return sql, params


def _match_rows(holder: StoreHolder, table: str, match_query: str, order_by_sql: str, *, fail_open: Optional[str] = None,
                operational_debug: Optional[str] = None, **kwargs) -> Optional[list[dict[str, Any]]]:
    """Run one MATCH against *table*; ``None`` when the query cannot execute (tokenizer /
    syntax) so the caller falls back. *fail_open* names the index for the
    substring-capable routes: a corruption-class ``DatabaseError`` there detaches the
    derived indexes (``_enter_fts_fail_open``) and answers from canonical rows — a live
    search never runs the unbounded rebuild. Other ``DatabaseError``s propagate."""
    sql, params = _fts_match_sql(table, match_query, order_by_sql, **kwargs)
    try:
        return [dict(row) for row in holder._read_all(sql, params)]
    except sqlite3.OperationalError:
        if operational_debug:
            logger.debug(operational_debug, exc_info=True)
        return None
    except sqlite3.DatabaseError as exc:
        if fail_open is None or not _enter_fts_fail_open(holder, exc):
            raise
        logger.warning(
            "%s FTS search hit a corruption error (%s); detached FTS and falling back to canonical LIKE.",
            fail_open, exc)
        return None


def _like_rows(holder: StoreHolder, where: list[str], params: list, *, order_by: str, limit_sql: str) -> list[dict[str, Any]]:
    """Canonical-table LIKE scan; ``params[0]`` is the snippet anchor term."""
    sql = _search_select_sql(_LIKE_SNIPPET_SQL, "messages m", where, order_by, limit_sql)
    return [dict(row) for row in holder._read_all(sql, params)]


def _compile_like_boolean_query(query: str) -> tuple[str, list[Any], Optional[str]]:
    """Compile the supported FTS boolean subset into LIKE predicates: terms within an OR
    group are ANDed (FTS5's implicit conjunction) and ``NOT`` negates the next term."""
    groups: list[list[tuple[str, bool]]] = [[]]
    negate_next = False
    for raw_token in _LIKE_TOKEN_RE.findall(query):
        operator = raw_token.upper()
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
        term = raw_token.strip('"').strip("*").strip()
        if term:
            groups[-1].append((term, negate_next))
            negate_next = False

    compiled_groups: list[str] = []
    params: list[Any] = []
    snippet_term: Optional[str] = None
    for group in groups:
        if not group or not any(not negated for _, negated in group):
            continue
        clauses: list[str] = []
        for term, negated in group:
            clauses.append(f"NOT {_LIKE_COALESCED_COLUMN_SQL}" if negated else _LIKE_COALESCED_COLUMN_SQL)
            params.extend(_like_params(term))
            if snippet_term is None and not negated:
                snippet_term = term
        compiled_groups.append(f"({' AND '.join(clauses)})")
    return " OR ".join(compiled_groups), params, snippet_term


def _search_messages_like_fallback(
    holder: StoreHolder, query: str, *, limit: int, offset: int, sort: Optional[str], **filters) -> list[dict[str, Any]]:
    """Search canonical messages while derived FTS state is stale."""
    predicate, params, snippet_term = _compile_like_boolean_query(query)
    if not predicate or snippet_term is None:
        return []
    where = [f"({predicate})"]
    _search_filter_clauses(where, params, **filters)
    order = "ASC" if isinstance(sort, str) and sort.strip().lower() == "oldest" else "DESC"
    return _like_rows(holder, where, [snippet_term, *params, limit, offset],
                           order_by=f"ORDER BY m.timestamp {order}, m.id {order}", limit_sql="LIMIT ? OFFSET ?")


def _refresh_fts_stale_state(holder: StoreHolder) -> None:
    """Observe fail-open initiated by another process sharing state.db."""
    if holder._fts_stale or not holder._fts_enabled:
        return
    try:
        stale = holder._read_one("SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_STALE_KEY,))
    except sqlite3.Error:
        return
    if stale is not None:
        holder._fts_stale = True
        holder._fts_enabled = holder._trigram_available = holder._fts_cjk_available = False


def _finalize_search_matches(
    holder: StoreHolder, matches: list[dict[str, Any]], result_fields: Optional[Collection[str]] = None) -> list[dict[str, Any]]:
    """Attach neighboring messages in bounded batches, only when context is requested."""
    if result_fields is None or "context" in result_fields:
        for start in range(0, len(matches), 500):
            batch = matches[start:start + 500]
            contexts = {match["id"]: [] for match in batch}
            try:
                sql = _CONTEXT_WINDOW_SQL.format(ids=",".join("?" for _ in contexts))
                with holder._read_ctx() as conn:
                    rows = conn.execute(sql, list(contexts)).fetchall()
                for row in rows:
                    contexts[row["match_id"]].append(row)
            except Exception:  # health: allow BLE001 -- context is optional; a failed read drops it
                contexts = {}
            for match in batch:
                try:
                    match["context"] = [
                        {"role": row["role"], "content": _flatten_text(_decode_content(row["content"]))[:200]}
                        for row in contexts.get(match["id"], [])]
                except Exception:  # health: allow BLE001 -- an undecodable context is empty
                    match["context"] = []
    # No route selects full content; the pop guards any future one that does.
    for match in matches:
        match.pop("content", None)
    if result_fields is not None:
        matches = [{field: match[field] for field in result_fields if field in match} for match in matches]
    return matches


def search_messages(
    holder: StoreHolder, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
    role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
    include_inactive: bool = False, fields: Optional[Collection[str]] = None,
    after_ts: Optional[int] = None, before_ts: Optional[int] = None,
) -> list[dict[str, Any]]:
    """:meth:`_search_messages_impl` plus one log line per slow search with the routing
    path taken. Threshold HERMES_SEARCH_SLOW_MS (default 1000; 0 logs every call)."""
    started = time.time()
    rows = None
    try:
        rows = _search_messages_impl(holder,
            query, source_filter=source_filter, exclude_sources=exclude_sources, role_filter=role_filter,
            limit=limit, offset=offset, sort=sort, include_inactive=include_inactive, fields=fields,
            after_ts=after_ts, before_ts=before_ts)
        return rows
    finally:
        elapsed_ms = (time.time() - started) * 1000.0
        if elapsed_ms >= _search_slow_ms():
            logger.info("slow session search: path=%s elapsed=%.0fms rows=%s query=%r",
                        _describe_search_path(holder, query), elapsed_ms, len(rows) if rows is not None else "err",
                        query[: 200])


def _search_messages_impl(
    holder: StoreHolder, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
    role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
    include_inactive: bool = False, fields: Optional[Collection[str]] = None,
    after_ts: Optional[int] = None, before_ts: Optional[int] = None,
) -> list[dict[str, Any]]:
    """FTS5 search across session messages (keywords, ``"phrases"``, AND/OR/NOT, ``prefix*``).
    Returns snippet + session metadata + 1-message context per hit; ``fields`` selects a
    projection. ``sort``: None = BM25 rank; "newest"/"oldest" = timestamp then rank (the
    CJK LIKE fallback ignores it). Rewound rows (``active=0, compacted=0``) are excluded
    by default; compaction-archived rows ARE included; ``include_inactive`` = every row.
    ``after_ts``/``before_ts`` bound ``sessions.started_at`` on every route (FTS5, CJK,
    trigram, LIKE fallback, unindexed-gap supplement)."""
    result_fields = _search_message_fields(fields)
    if not query or not query.strip():
        return []
    query = _sanitize_fts5_query(query)
    if not query:
        return []
    filters = dict(include_inactive=include_inactive, source_filter=source_filter,
                   exclude_sources=exclude_sources, role_filter=role_filter,
                   after_ts=after_ts, before_ts=before_ts)
    # New oversized tool results index only a bounded prefix; an explicit tool-role search is the
    # opt-in full-body path and scans canonical rows via LIKE.
    if role_filter and "tool" in role_filter:
        matches = _search_messages_like_fallback(holder, query, limit=limit, offset=offset, sort=sort, **filters)
        return _finalize_search_matches(holder, matches, result_fields=result_fields)
    _refresh_fts_stale_state(holder)
    if holder._fts_stale:
        matches = _search_messages_like_fallback(holder, query, limit=limit, offset=offset, sort=sort, **filters)
        return _finalize_search_matches(holder, matches, result_fields=result_fields)
    if not holder._fts_enabled:
        return []

    order_by_sql = _FTS_ORDER_BY.get(sort.strip().lower() if isinstance(sort, str) else None, "ORDER BY rank")
    route = dict(order_by_sql=order_by_sql, limit=limit, offset=offset, **filters)
    # Tool rows and FTS_TRIGRAM_EXCLUDED_SOURCES sessions are excluded from the trigram/cjk
    # indexes (see FTS_TRIGRAM_SQL); an explicit filter for them must scan the base table.
    wants_unindexed_rows = (bool(role_filter) and "tool" in role_filter) or (
        bool(source_filter) and any(src in FTS_TRIGRAM_EXCLUDED_SOURCES for src in source_filter))
    is_cjk = _contains_cjk(query)
    if is_cjk:
        matches = _search_cjk(holder, query, wants_unindexed_rows, route)
    else:
        sql, params = _fts_match_sql("messages_fts", query, **route)
        try:
            matches = [dict(row) for row in holder._read_all(sql, params)]
        except sqlite3.OperationalError:
            return []  # FTS5 syntax error despite sanitization
        except sqlite3.DatabaseError as exc:
            # Corruption parent class: detach the derived indexes and answer from
            # canonical rows; repair paths own the rebuild.
            # A corrupt FTS index raises the malformed / "fts5: corrupt structure record" class on the
            # MATCH read, the same class the write path handles (#66296). OperationalError (query
            # syntax) is a subclass caught above; this arm is the corruption parent. The existing
            # stale-open/repair paths retain rebuild ownership.
            if not _enter_fts_fail_open(holder, exc):
                raise
            matches = _search_messages_like_fallback(holder, query, limit=limit, offset=offset, sort=sort, **filters)

    # Deferred-rebuild supplement: while the backfill is pending the FTS indexes miss
    # the (progress, high_water] gap; top up with a bounded LIKE scan so old messages
    # never vanish mid-rebuild. Cost decays to zero as the backfill advances.
    if fts_rebuild_status(holder) is not None and len(matches) < limit:
        try:
            gap_matches = _search_unindexed_gap(holder, query, limit - len(matches), **filters)
            seen_ids = {m["id"] for m in matches}
            matches.extend(m for m in gap_matches if m["id"] not in seen_ids)
        except sqlite3.OperationalError as exc:
            logger.debug("Unindexed-gap supplement skipped: %s", exc)

    # unicode61 puts no boundary between Latin and adjacent CJK ("修改youer服务端" is
    # one token, so MATCH "youer" misses). On a zero-result Latin miss retry the
    # substring-capable indexes: cjk first (exact ranked match), then trigram (>=3-char
    # tokens). Gated on a miss so hits keep their ranking ("cat" may then match
    # "concatenate"). Skipped for role='tool' (both indexes exclude tool rows).
    if not matches and not is_cjk and not (bool(role_filter) and "tool" in role_filter):
        fb_query = _quote_fts_tokens(query.strip('"').strip())
        # ── CJK-bigram route (messages_fts_cjk, cjk_unicode61) ────── When the bigram index is
        # available it serves EVERY CJK query shape the legacy code split between trigram (>=3
        # chars/token) and LIKE full scans (1-2 char tokens) — the whole point of the index (PR #65544).
        # Exceptions stay on the legacy routes: - role_filter=['tool'] queries (tool rows aren't in the
        # cjk index, same exclusion as trigram), - queries containing a LONE 1-char CJK run: the index
        # stores bigrams for runs >=2, so a single-char term can only match isolated chars — LIKE
        # substring semantics are broader.
        if holder._fts_cjk_available:
            matches = _match_rows(holder, "messages_fts_cjk", fb_query, **route) or matches
        if not matches and holder._trigram_available and _trigram_eligible_tokens(query):
            matches = _match_rows(holder, "messages_fts_trigram", fb_query, **route) or matches

    # OR-relaxed retry: the implicit AND between terms means a paraphrased multi-word query
    # misses a stored sentence that lacks even ONE word ("when does Sarah like her standup
    # scheduled" vs "Sarah prefers the standup meeting scheduled ... Thursday mornings"). Once
    # the exact query and the substring fallbacks all miss, retry the unicode61 index matching
    # ANY term. The caller's ``sort`` still applies (``route`` carries order_by_sql): rank order
    # puts rows covering more terms first, newest/oldest keep their timestamp order. Gated on a
    # zero-result miss so hits keep exact-match semantics; explicit OR/NOT, single-term and
    # CJK-routed queries are left alone.
    if not matches and not is_cjk and not holder._fts_stale:
        relaxed = _or_relaxed_query(query)
        if relaxed is not None:
            matches = _match_rows(holder, "messages_fts", relaxed, fail_open="OR-relaxed",
                                       operational_debug="OR-relaxed FTS retry failed; keeping empty result",
                                       **route) or matches
    return _finalize_search_matches(holder, matches, result_fields=result_fields)


def _search_cjk(holder: StoreHolder, query: str, wants_unindexed_rows: bool, route: dict[str, Any]) -> list[dict[str, Any]]:
    """CJK routing: the unicode61 table splits CJK into single characters (false positives,
    missed phrases). cjk-bigram serves every shape except queries wanting rows the
    substring indexes exclude (role='tool', cron/subagent sources) and LONE
    1-char CJK runs (bigrams only exist for runs >=2 — LIKE is broader); then trigram
    (>=3 CJK chars per token); then a LIKE substring scan with one clause per
    non-operator token so "广西 OR 桂林 OR 漓江" matches each term."""
    raw_query = _strip_cjk_wildcards(query).strip('"').strip()
    match_query = _quote_fts_tokens(raw_query)
    if holder._fts_cjk_available and not wants_unindexed_rows and not _has_lone_cjk_run(raw_query):
        matches = _match_rows(holder,
            "messages_fts_cjk", match_query, fail_open="CJK-bigram",
            operational_debug="messages_fts_cjk query failed; falling back to trigram/LIKE", **route)
        if matches is not None:
            return matches
    if _trigram_route_ok(holder, raw_query) and not wants_unindexed_rows:
        matches = _match_rows(holder, "messages_fts_trigram", match_query, fail_open="Trigram", **route)
        if matches is not None:
            return matches
    non_op_tokens = _non_operator_tokens(raw_query) or [raw_query]
    like_params: list = [p for tok in non_op_tokens for p in _like_params(tok)]
    like_where = [f"({' OR '.join([_LIKE_ANY_COLUMN_SQL] * len(non_op_tokens))})"]
    filters = {k: route[k] for k in ("include_inactive", "source_filter", "exclude_sources", "role_filter",
                                     "after_ts", "before_ts")}
    _search_filter_clauses(like_where, like_params, **filters)
    # instr() for the snippet uses the first search token.
    return _like_rows(holder, like_where, [non_op_tokens[0], *like_params, route["limit"], route["offset"]],
                           order_by="ORDER BY m.timestamp DESC", limit_sql="LIMIT ? OFFSET ?")


def _search_unindexed_gap(holder: StoreHolder, fts_query: str, limit: int, **filters) -> list[dict[str, Any]]:
    """LIKE-scan ids in (fts_rebuild_progress, fts_rebuild_high_water] — rows the deferred
    rebuild hasn't indexed yet. The FTS query degrades to AND-joined substring terms
    (quoted phrases kept whole): recall-over-precision mid-rebuild."""
    status = fts_rebuild_status(holder)
    if status is None or limit <= 0:
        return []
    terms = [tok for tok in (t.strip('"').strip("*").strip() for t in _LIKE_TOKEN_RE.findall(fts_query))
             if tok and tok.upper() not in _LIKE_SKIP_TOKENS]
    if not terms:
        return []
    where = ["m.id > ? AND m.id <= ?", *([_LIKE_ANY_COLUMN_SQL] * len(terms))]
    params: list = [status["indexed"], status["total"], *(p for term in terms for p in _like_params(term))]
    _search_filter_clauses(where, params, **filters)
    return _like_rows(holder, where, [terms[0], *params, limit], order_by="ORDER BY m.timestamp DESC",
                           limit_sql="LIMIT ?")


def search_sessions_by_id(
    holder: StoreHolder, query: str, limit: int = 20, include_archived: bool = True, source: str | None = None,
    sources: list[str] | None = None, exclude_sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Search surfaced sessions by exact/prefix/substring session id. Also matches
    ``_lineage_root_id`` so an old compression root id resolves to the live continuation."""
    needle = (query or "").strip().lower()
    if not needle or limit <= 0:
        return []
    # list_sessions_rich pushes the id LIKE filter (own id + forward compression
    # chain) into SQL; over-fetch so the in-Python ranking has candidates.
    candidates = list_sessions_rich(holder,
        source=source, sources=sources, exclude_sources=exclude_sources, limit=max(limit * 4, limit),
        offset=0, include_archived=include_archived, order_by_last_active=True, id_query=needle)

    def score(row: dict[str, Any]) -> int:
        normalized = [v.lower() for v in (str(row.get("id") or ""), str(row.get("_lineage_root_id") or "")) if v]
        if any(value == needle for value in normalized):
            return 0
        return 1 if any(value.startswith(needle) for value in normalized) else 2
    ranked = sorted(enumerate(candidates), key=lambda item: (score(item[1]), item[0]))
    return [row for _, row in ranked[:limit]]


def rebuild_fts(holder: StoreHolder) -> int:
    """Rebuild FTS5 indexes from ``messages`` (``'rebuild'``) — the recovery for a corrupt index
    that rejects writes while reads succeed. Two processes rebuilding one state.db
    concurrently corrupted production DBs, so this admits through
    ``fts_rebuild_admission`` and FAILS CLOSED, returning 0 on deferral (callers treat 0
    as "no progress" and use the stale-FTS breadcrumb path). Returns indexes rebuilt.

    Uses the FTS5 ``'rebuild'`` command, which rewrites the internal b-tree segments from the content
    rows. Unlike ``optimize_fts`` (which merges existing segments), ``rebuild`` discards and recreates
    the index data entirely — the more destructive of the two, so it is quarantined the same way. See
    #50502.
    A full structural rebuild must never run concurrently in two processes sharing one state.db — that
    interleaving has structurally corrupted the database in production (PR #93200) — so this admits
    through the cross-process ``fts_rebuild_admission`` authority and FAILS CLOSED: if another process
    holds the rebuild lock beyond the bounded wait, this call defers (returns 0) rather than racing it.
    Callers already treat 0 as "rebuild made no progress" and fall back to the stale-FTS breadcrumb
    path, which retries in-process from the gateway housekeeping tick (``retry_deferred_fts_recovery``)
    and at next startup.
    """
    holder._raise_if_db_corrupt()
    rebuilt = 0
    with fts_rebuild_admission(holder.db_path) as admitted:
        if not admitted:
            logger.warning(
                "Deferred in-place FTS rebuild: another process holds the rebuild authority for this state.db.")
            return 0
        with holder._lock:
            holder._raise_if_db_replaced()
            if holder._conn is None:
                holder._reopen_after_close_locked(context="write")
            for tbl in _present_fts_tables(holder):
                try:
                    holder._conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')")
                    holder._conn.commit()
                    rebuilt += 1
                except sqlite3.DatabaseError as exc:  # SQLITE_CORRUPT is not an OperationalError (#133375)
                    holder._conn.rollback()
                    if is_malformed_db_error(exc):
                        logger.error(
                            "FTS rebuild failed with a corruption-class error for %s: %s; "
                            "run 'hermes sessions repair' (or 'hermes doctor --fix') to rebuild the index offline",
                            tbl, exc)
                    else:
                        logger.warning("FTS rebuild failed for %s: %s", tbl, exc)
    return rebuilt


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.messages import _decode_content, _row_to_message_dict, get_messages_around
from persistence.sqlite.conversation._store.sessions import list_sessions_rich
