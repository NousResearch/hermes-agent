"""In-memory reference ``SessionSearch`` repository (test fixture).

It reads the rows the sessions and messages references store, plus rows planted through the fixture seam ``seed_session`` / ``seed_message`` (sessions in this module's own ``search`` table, messages through the transcript store), and runs the store's query routes over them: the sanitizer, the word route with its substring and OR-relaxed retries, the ``tool`` role substring route and the CJK routes, with the store's filters, order, paging, snippets and context windows. Word matching models the store's tokenizer (letters, digits and marks; case and Latin diacritics folded); substring matching models its trigram index (one-to-one case folding) and its LIKE scans (ASCII case folding, text read up to the first NUL). Relevance is not modelled: equal-rank rows keep id order.

Index administration: there is no derived index, so an admitted ``rebuild_search_index`` reports 0 indexes; ``search_index_rebuild_status`` reads a marker planted with ``seed_deferred_rebuild`` and the rebuild leaves it alone; ``retry_deferred_search_recovery`` is always False because nothing is ever deferred.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

from persistence import MessageId, ProfileScope, RowVisibility, SessionId
from persistence.conversation.messages import CONTENT_JSON_PREFIX
from persistence.conversation.search import (
    FTS_TOOL_CONTENT_PREFIX_CHARS,
    MAX_SEARCH_QUERY_CHARS,
    SEARCH_RESULT_FIELDS,
    SearchContext,
    SearchHit,
    SearchIndexRebuildStatus,
    SearchSort,
)

from ..backend import InMemoryBackend, register_repository

# Bounded wait for the per-scope rebuild authority (contract: "bounded", the value is backend detail).
_REBUILD_ADMISSION_WAIT_SECONDS = 5.0
_TRIGRAM_EXCLUDED_SOURCES = ("cron", "subagent")
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
_OPERATORS = frozenset({"AND", "OR", "NOT"})
_SPECIAL_RE = re.compile(r"""[+{}():"^@/#&|~\[\]<>,;!?$=\\']""")
_QUOTED_RE = re.compile(r'"[^"]*"')
_UNIT_RE = re.compile(r'"[^"]+"|\S+')
_SNIPPET_TOKENS = 40


def _is_cjk(char: str) -> bool:
    return any(lo <= ord(char) <= hi for lo, hi in _CJK_RANGES)


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if _is_cjk(char))


def _non_operator_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token.upper() not in _OPERATORS]


def _quote_tokens(text: str) -> str:
    """Each non-operator word quoted, the operator words kept."""
    return " ".join(t if t.upper() in _OPERATORS else '"' + t.replace('"', '""') + '"' for t in text.split())


def _strip_cjk_wildcards(text: str) -> str:
    """Trailing ``*`` dropped from each non-operator word unless it is all stars (``_strip_cjk_wildcards``)."""
    if "*" not in text:
        return text
    return " ".join(t if t.upper() in _OPERATORS else (t.rstrip("*") or t) for t in text.split()) or text


def sanitize_query(query: str) -> str:
    """The store's query sanitizer (``SessionSearchMixin._sanitize_fts5_query``)."""
    query = query[:MAX_SEARCH_QUERY_CHARS]
    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00Q{len(held) - 1}\x00"

    text = _QUOTED_RE.sub(_hold, query).replace('"', " ")
    text = _SPECIAL_RE.sub(" ", text)
    if "%" in text and not any(_is_cjk(char) for char in text):
        text = text.replace("%", " ")
    text = re.sub(r"\*+", "*", text)
    text = re.sub(r"(^|\s)\*", r"\1", text)
    text = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", text.strip())
    text = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", text.strip())
    text = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', text)
    for index, quoted in enumerate(held):
        text = text.replace(f"\x00Q{index}\x00", quoted)
    return text.strip()


def _or_relaxed(query: str) -> Optional[str]:
    """The implicit-AND query with its units ORed, or None for fewer than two units or an explicit OR / NOT (``_or_relaxed_query``)."""
    units: list[str] = []
    for token in _UNIT_RE.findall(query):
        if token.upper() in {"OR", "NOT"}:
            return None
        if token.upper() != "AND":
            units.append(token)
    return " OR ".join(units) if len(units) >= 2 else None


def _word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LNM" or category == "Co"


def _fold_word(word: str) -> str:
    """One word as the word index stores it: combining marks dropped (also from characters that decompose into one), lowercased."""
    out = []
    for char in word:
        parts = unicodedata.normalize("NFD", char)
        if any(unicodedata.category(part) == "Mn" for part in parts):
            char = "".join(part for part in parts if unicodedata.category(part) != "Mn")
        out.append(char)
    return "".join(out).lower()


def words(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, folded)`` for each word of ``text`` (maximal runs of letters, digits and marks)."""
    out: list[tuple[int, int, str]] = []
    start: Optional[int] = None
    for index, char in enumerate(text + " "):
        if index < len(text) and _word_char(char):
            start = index if start is None else start
        elif start is not None:
            out.append((start, index, _fold_word(text[start:index])))
            start = None
    return [item for item in out if item[2]]


def fold_char(char: str) -> str:
    """Case folding one character to one character, as the substring index does."""
    mapped = char.casefold()
    if len(mapped) != 1:
        mapped = char.lower()
    return mapped if len(mapped) == 1 else char


def fold_chars(text: str) -> str:
    return "".join(fold_char(char) for char in text)


def like_text(text: Optional[str]) -> str:
    """A stored text as the LIKE scans read it: up to its first NUL, empty for NULL."""
    return (text or "").split("\x00", 1)[0]


def _ascii_lower(value: str) -> str:
    return re.sub(r"[A-Z]+", lambda match: match.group().lower(), value)


def like_contains(text: Optional[str], term: str) -> bool:
    """SQL ``LIKE '%term%'``: ASCII case folding only."""
    return _ascii_lower(term) in _ascii_lower(like_text(text))


class QuerySyntaxError(ValueError):
    """Sanitized text the store's query grammar still rejects."""


@dataclass(frozen=True)
class _Phrase:
    text: str
    prefix: bool


@dataclass(frozen=True)
class _Node:
    op: str
    left: Any
    right: Any


def _bareword_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_\x1a") or not char.isascii()


def _lex(text: str) -> list[Any]:
    """Phrases and operator words of a sanitized query; any other character is a syntax error."""
    out: list[Any] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in " \t\n\r":
            i += 1
            continue
        if char == '"':
            end, parts = i + 1, []
            while True:
                close = text.find('"', end)
                if close < 0:
                    raise QuerySyntaxError("unterminated phrase")
                parts.append(text[end:close])
                if text.startswith('""', close):
                    parts.append('"')
                    end = close + 2
                    continue
                i = close + 1
                break
            phrase = "".join(parts)
        elif _bareword_char(char):
            start = i
            while i < len(text) and _bareword_char(text[i]):
                i += 1
            phrase = text[start:i]
            if phrase in _OPERATORS and not text.startswith("*", i):
                out.append(phrase)
                continue
        else:
            raise QuerySyntaxError(f"unexpected {char!r}")
        prefix = text.startswith("*", i)
        i += 1 if prefix else 0
        out.append(_Phrase(phrase, prefix))
    return out


def parse_query(text: str) -> Any:
    """The query tree: adjacent phrases bind tightest, then ``NOT``, ``AND``, ``OR``; all left-associative."""
    items = _lex(text)
    position = 0

    def level(operator: Optional[str]) -> Any:
        nonlocal position
        if operator is None:
            node = None
            while position < len(items) and isinstance(items[position], _Phrase):
                node = items[position] if node is None else _Node("AND", node, items[position])
                position += 1
            if node is None:
                raise QuerySyntaxError("operand expected")
            return node
        lower = {"OR": "AND", "AND": "NOT", "NOT": None}[operator]
        node = level(lower)
        while position < len(items) and items[position] == operator:
            position += 1
            node = _Node(operator, node, level(lower))
        return node

    if not items:
        raise QuerySyntaxError("empty query")
    tree = level("OR")
    if position != len(items):
        raise QuerySyntaxError("operator out of place")
    return tree


def _phrase_words(phrase: _Phrase) -> list[str]:
    return [word for _, _, word in words(phrase.text)]


def _word_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Character spans of ``phrase`` in ``text`` under word matching (the last word as a prefix with ``*``)."""
    needle = _phrase_words(phrase)
    found = words(text)
    spans: list[tuple[int, int]] = []
    for i in range(len(found) - len(needle) + 1 if needle else 0):
        run = [word for _, _, word in found[i : i + len(needle)]]
        last_ok = run[-1].startswith(needle[-1]) if phrase.prefix else run[-1] == needle[-1]
        if run[:-1] == needle[:-1] and last_ok:
            spans.append((found[i][0], found[i + len(needle) - 1][1]))
    return spans


def _substring_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Spans of ``phrase`` in ``text`` as a case-folded substring; a phrase under three characters matches without a span."""
    if len(phrase.text) < 3:
        return [(0, 0)]
    needle, haystack = fold_chars(phrase.text), fold_chars(text)
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), haystack)]


def matched_phrases(tree: Any, fields: Sequence[str], hits: Callable[[_Phrase, str], list[tuple[int, int]]]) -> list[_Phrase]:
    """The phrases of the branches of ``tree`` a row with searchable columns ``fields`` matches (empty when the row does not match); a phrase matches inside one column."""
    if isinstance(tree, _Phrase):
        return [tree] if any(hits(tree, text) for text in fields) else []
    left = matched_phrases(tree.left, fields, hits)
    if tree.op == "NOT":
        return left if left and not matched_phrases(tree.right, fields, hits) else []
    right = matched_phrases(tree.right, fields, hits)
    if tree.op == "AND":
        return left + right if left and right else []
    return left + right


def _marked(text: str, spans: Sequence[tuple[int, int]], start: int, end: int) -> str:
    """``text[start:end]`` with every span inside it (overlaps merged) between ``>>>`` and ``<<<``; the store's snippet writer drops plain text from a NUL to the next match."""
    merged: list[list[int]] = []
    for lo, hi in sorted(span for span in spans if span[0] < span[1] and span[0] >= start and span[1] <= end):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    parts, cursor = [], start
    for lo, hi in merged:
        parts += [text[cursor:lo].split("\x00", 1)[0], ">>>", text[lo:hi], "<<<"]
        cursor = hi
    parts.append(text[cursor:end].split("\x00", 1)[0])
    return "".join(parts)


def word_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """An excerpt of the first column holding a match of ``phrases``: up to 40 words around the first one, matches marked, ``...`` where cut."""
    for text in columns:
        spans = [span for phrase in phrases for span in _word_hits(phrase, text)]
        if not spans:
            continue
        found = words(text)
        if len(found) <= _SNIPPET_TOKENS:
            return _marked(text, spans, 0, len(text))
        first = next(i for i, item in enumerate(found) if item[0] == min(spans)[0])
        lo = min(max(0, first - (_SNIPPET_TOKENS - 1) // 2), len(found) - _SNIPPET_TOKENS)
        hi = lo + _SNIPPET_TOKENS
        start = found[lo][0] if lo else 0
        end = found[hi - 1][1] if hi < len(found) else len(text)
        return ("..." if lo else "") + _marked(text, spans, start, end) + ("..." if hi < len(found) else "")
    return columns[0] if columns else ""


def substring_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """The first column holding a substring match of ``phrases``, matches marked; long text cut to about 40 characters around the first."""
    for text in columns:
        spans = [span for phrase in phrases for span in _substring_hits(phrase, text) if span[1] > span[0]]
        if not spans:
            continue
        if len(text) <= _SNIPPET_TOKENS + 2:
            return _marked(text, spans, 0, len(text))
        lo = max(0, min(min(spans)[0] - _SNIPPET_TOKENS // 2, len(text) - _SNIPPET_TOKENS))
        hi = lo + _SNIPPET_TOKENS
        return ("..." if lo else "") + _marked(text, spans, lo, hi) + ("..." if hi < len(text) else "")
    return columns[0] if columns else ""
def like_snippet(content: Optional[str], term: str) -> Optional[str]:
    """``substr(content, max(1, instr(content, term) - 40), 120)``."""
    if content is None:
        return None
    start = max(1, content.find(term) + 1 - 40)
    return content[start - 1 : start - 1 + 120]


def like_boolean(query: str) -> tuple[list[list[tuple[str, bool]]], Optional[str]]:
    """The ``tool`` role route's predicate: OR groups of (word, negated), groups without a positive word dropped, and the snippet word (``_compile_like_boolean_query``)."""
    groups: list[list[tuple[str, bool]]] = [[]]
    negate = False
    for token in _UNIT_RE.findall(query):
        operator = token.upper()
        if operator == "OR":
            if groups[-1]:
                groups.append([])
            negate = False
        elif operator == "NOT":
            negate = True
        elif operator not in {"AND", "NEAR"}:
            term = token.strip('"').strip("*").strip()
            if term:
                groups[-1].append((term, negate))
                negate = False
    kept = [group for group in groups if any(not negated for _, negated in group)]
    first = next((term for group in kept for term, negated in group if not negated), None)
    return kept, first


@dataclass
class _Session:
    session_id: str
    source: Optional[str]
    model: Optional[str]
    started_at: Optional[float]
    delegate_child: bool = False


@dataclass
class _Row:
    id: int
    session: _Session
    role: str
    content: Optional[str]
    decoded: Any
    timestamp: float
    tool_name: Optional[str]
    tool_calls: Optional[str]
    visibility: RowVisibility
    display_kind: Optional[str]

    def word_columns(self) -> list[str]:
        content = (self.content or "")[:FTS_TOOL_CONTENT_PREFIX_CHARS] if self.role == "tool" else self.content or ""
        return [content, self.tool_name or "", self.tool_calls or ""]

    def substring_indexed(self) -> bool:
        return self.role != "tool" and self.session.source not in _TRIGRAM_EXCLUDED_SOURCES and not self.session.delegate_child

    def substring_columns(self) -> list[str]:
        return [self.content or "", self.tool_name or ""]

    def like_columns(self) -> list[Optional[str]]:
        return [self.content, self.tool_name, self.tool_calls]


def _stored_text(content: Any) -> Optional[str]:
    """A stored payload's content column: text as kept, structured content as its ``\\x00json:`` encoding."""
    if content is None or isinstance(content, str):
        return content
    return CONTENT_JSON_PREFIX + json.dumps(content)


def _search_fields(fields: Optional[Collection[str]]) -> Optional[tuple[str, ...]]:
    """The requested projection in canonical order (``SessionSearchMixin._search_message_fields``)."""
    if fields is None:
        return None
    if isinstance(fields, str):
        raise TypeError("search fields must be a collection of field names, not a string")
    requested = set(fields)
    unknown = requested.difference(SEARCH_RESULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown search result field(s): {', '.join(sorted(unknown))}")
    return tuple(field for field in SEARCH_RESULT_FIELDS if field in requested)


def _page(rows: list[Any], limit: int, offset: int) -> list[Any]:
    """SQL ``LIMIT ? OFFSET ?``: a negative limit is no limit, a negative offset is zero."""
    start = max(0, offset)
    return rows[start:] if limit < 0 else rows[start : start + limit]


def _ranked(rows: list[_Row], sort: SearchSort) -> list[_Row]:
    """Word and substring route order: timestamp for NEWEST / OLDEST, equal rank (id order) otherwise and within a timestamp."""
    if sort is SearchSort.NEWEST:
        return sorted(rows, key=lambda r: (-r.timestamp, r.id))
    if sort is SearchSort.OLDEST:
        return sorted(rows, key=lambda r: (r.timestamp, r.id))
    return sorted(rows, key=lambda r: r.id)


def _scanned(rows: list[_Row], sort: SearchSort) -> list[_Row]:
    """``tool`` role route order: (timestamp, id) ascending for OLDEST, descending otherwise."""
    return sorted(rows, key=lambda r: (r.timestamp, r.id), reverse=sort is not SearchSort.OLDEST)


class InMemorySessionSearch:
    """Reference implementation of the ``SessionSearch`` protocol plus the fixture seeding seam."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend
        self._authorities: dict[str, threading.Lock] = {}
        self._authorities_guard = threading.Lock()

    def _tables(self, scope: ProfileScope) -> dict[str, Any]:
        tables: dict[str, Any] = self._backend.scope_store(scope).setdefault("search", {"sessions": {}})
        return tables

    def _transcript_rows(self, scope: ProfileScope) -> dict[str, list[_Row]]:
        """Every stored row by session, (timestamp, id) ordered, for sessions recorded by the sessions reference or the seam."""
        from .transcript_store import transcripts_of

        store = self._backend.scope_store(scope)
        sessions = dict(self._tables(scope)["sessions"])
        for sid, row in store.get("sessions", {}).items():
            started = row["started_at"]
            sessions[sid] = _Session(sid, row["source"], row["model"], started.timestamp() if started is not None else None,
                                     "_delegate_from" in (row.get("model_config") or {}))
        out: dict[str, list[_Row]] = {}
        for sid, transcript in transcripts_of(self._backend, scope).items():
            if sid not in sessions:
                continue
            out[sid] = sorted((_Row(
                stored.id, sessions[sid], stored.message.role, _stored_text(stored.message.content), stored.message.content,
                stored.epoch, stored.message.tool_name,
                json.dumps(list(stored.message.tool_calls)) if stored.message.tool_calls else None,
                stored.visibility, stored.message.display_kind) for stored in transcript.rows), key=lambda r: (r.timestamp, r.id))
        return out

    def _rebuild_authority(self, scope: ProfileScope) -> threading.Lock:
        """The per-store rebuild authority; kept off the scope store (the authority reference deep-copies those tables)."""
        with self._authorities_guard:
            return self._authorities.setdefault(scope.hermes_home_key, threading.Lock())

    # ── fixture seeding seam ─────────────────────────────────────────────────

    def seed_session(self, session_id: str, *, source: Optional[str] = None, model: Optional[str] = None,
                     started_at: Optional[float] = None, delegate_child: bool = False) -> SessionId:
        """Record a session for search only (no sessions-reference row), with the start time and source the filters read."""
        scope = self._backend.scope
        self._tables(scope)["sessions"][session_id] = _Session(session_id, source, model, started_at, delegate_child)
        return SessionId(session_id)

    def seed_message(self, session_id: str, role: str, content: Any, *, timestamp: float,
                     tool_name: Optional[str] = None, tool_calls: Sequence[Mapping[str, Any]] = (),
                     display_kind: Optional[str] = None, visibility: RowVisibility = RowVisibility.ACTIVE) -> MessageId:
        """Store one transcript row of a recorded session through the transcript store, in the given visibility."""
        scope = self._backend.scope
        from persistence.conversation.messages import NewMessage

        from .transcript_store import _messages_namespace, _Transcript, insert_rows

        if session_id not in self._tables(scope)["sessions"] and session_id not in self._backend.scope_store(scope).get("sessions", {}):
            raise ValueError(f"seed_message before seed_session for {session_id!r}")
        namespace = _messages_namespace(self._backend, scope)
        transcript = namespace["transcripts"].setdefault(session_id, _Transcript())
        message = NewMessage(role, content, float(timestamp), tool_name=tool_name, tool_calls=tuple(tool_calls), display_kind=display_kind)
        row = insert_rows(namespace, transcript, [message])[0]
        row.visibility = visibility
        return MessageId(row.id)

    def seed_deferred_rebuild(self, total: int, indexed: int) -> None:
        """Plant a deferred-rebuild progress marker (stand-in for a chunked backfill the reference never runs)."""
        scope = self._backend.scope
        self._tables(scope)["deferred_rebuild"] = {"total": int(total), "indexed": int(indexed)}

    # ── index administration ─────────────────────────────────────────────────

    def rebuild_search_index(self) -> int:
        scope = self._backend.scope
        self._tables(scope)
        authority = self._rebuild_authority(scope)
        if not authority.acquire(timeout=_REBUILD_ADMISSION_WAIT_SECONDS):
            return 0
        try:
            # Nothing derived exists; the stored rows are the index.
            return 0
        finally:
            authority.release()

    def retry_deferred_search_recovery(self) -> bool:
        scope = self._backend.scope
        self._tables(scope)
        return False

    def search_index_rebuild_status(self) -> Optional[SearchIndexRebuildStatus]:
        scope = self._backend.scope
        marker = self._tables(scope).get("deferred_rebuild")
        if marker is None or marker["total"] <= 0:
            return None
        return SearchIndexRebuildStatus(total=marker["total"], indexed=marker["indexed"])

    # ── search ───────────────────────────────────────────────────────────────

    def search_messages(self, query: str, *, source_filter: Optional[Sequence[str]] = None,
                        exclude_sources: Optional[Sequence[str]] = None, role_filter: Optional[Sequence[str]] = None,
                        limit: int = 20, offset: int = 0, sort: SearchSort = SearchSort.RELEVANCE, include_inactive: bool = False,
                        fields: Optional[Collection[str]] = None, after_ts: Optional[int] = None,
                        before_ts: Optional[int] = None) -> tuple[SearchHit, ...]:
        scope = self._backend.scope
        projection = _search_fields(fields)
        if not query or not query.strip():
            return ()
        text = sanitize_query(query)
        if not text:
            return ()
        transcripts = self._transcript_rows(scope)
        pool = [row for rows in transcripts.values() for row in rows if _eligible(
            row, include_inactive=include_inactive, source_filter=source_filter, exclude_sources=exclude_sources,
            role_filter=role_filter, after_ts=after_ts, before_ts=before_ts)]
        tool_route = bool(role_filter) and "tool" in (role_filter or ())
        if tool_route:
            found = _tool_route(pool, text, sort, limit, offset)
        elif any(_is_cjk(char) for char in text):
            unindexed = bool(source_filter) and any(src in _TRIGRAM_EXCLUDED_SOURCES for src in source_filter or ())
            found = _cjk_route(pool, text, unindexed, sort, limit, offset)
        else:
            try:
                found = _word_route(pool, parse_query(text), sort, limit, offset)
            except QuerySyntaxError:
                return ()
            if not found:
                found = _substring_retry(pool, text, sort, limit, offset)
            relaxed = _or_relaxed(text) if not found else None
            if relaxed is not None:
                try:
                    found = _word_route(pool, parse_query(relaxed), sort, limit, offset)
                except QuerySyntaxError:
                    found = []
        return tuple(_hit(row, snippet, transcripts, projection) for row, snippet in found)


def _eligible(row: _Row, *, include_inactive: bool, source_filter: Optional[Sequence[str]], exclude_sources: Optional[Sequence[str]],
              role_filter: Optional[Sequence[str]], after_ts: Optional[int], before_ts: Optional[int]) -> bool:
    """The filters every route shares (``_search_filter_clauses``); SQL NULL comparisons exclude a session without a source or start."""
    session = row.session
    checks = (
        include_inactive or row.visibility is not RowVisibility.REWOUND,
        (row.display_kind or "") != "hidden",
        source_filter is None or session.source in source_filter,
        exclude_sources is None or not exclude_sources or (session.source is not None and session.source not in exclude_sources),
        not role_filter or row.role in role_filter,
        after_ts is None or (session.started_at is not None and session.started_at >= int(after_ts)),
        before_ts is None or (session.started_at is not None and session.started_at < int(before_ts)),
    )
    return all(checks)


def _word_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: matched_phrases(tree, row.word_columns(), _word_hits) for row in pool}
    matched = [row for row in pool if phrases[row.id]]
    return [(row, word_snippet(row.word_columns(), phrases[row.id])) for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: matched_phrases(tree, row.substring_columns(), _substring_hits) for row in pool if row.substring_indexed()}
    matched = [row for row in pool if phrases.get(row.id)]
    return [(row, substring_snippet(row.substring_columns(), phrases[row.id])) for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_retry(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    """The zero-result retry over the substring index, for a query whose words all have three or more characters."""
    bare = text.strip('"').strip()
    tokens = _non_operator_tokens(bare)
    if not tokens or any(len(token) < 3 for token in tokens):
        return []
    try:
        return _substring_route(pool, parse_query(_quote_tokens(bare)), sort, limit, offset)
    except QuerySyntaxError:
        return []


def _tool_route(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, Optional[str]]]:
    groups, first = like_boolean(text)
    if first is None:
        return []

    def present(row: _Row, term: str) -> bool:
        return any(like_contains(column, term) for column in row.like_columns())

    matched = [row for row in pool if any(all(present(row, term) != negated for term, negated in group) for group in groups)]
    return [(row, like_snippet(row.content, first)) for row in _page(_scanned(matched, sort), limit, offset)]


def _cjk_route(pool: list[_Row], text: str, unindexed: bool, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, Optional[str]]]:
    """CJK text: the substring index when every CJK word has three or more CJK characters, else any word by LIKE, newest first."""
    raw = _strip_cjk_wildcards(text).strip('"').strip()
    tokens = _non_operator_tokens(raw)
    short = any(_count_cjk(token) < 3 for token in tokens if any(_is_cjk(char) for char in token))
    if _count_cjk(raw) >= 3 and not short and not unindexed:
        try:
            return _substring_route(pool, parse_query(_quote_tokens(raw)), sort, limit, offset)
        except QuerySyntaxError:
            pass
    terms = tokens or [raw]
    matched = [row for row in pool if any(like_contains(column, term) for term in terms for column in row.like_columns())]
    ordered = sorted(matched, key=lambda r: (r.timestamp, r.id), reverse=True)
    return [(row, like_snippet(row.content, terms[0])) for row in _page(ordered, limit, offset)]


def _context(row: _Row, transcripts: dict[str, list[_Row]]) -> tuple[SearchContext, ...]:
    """The previous row, the row and the next row of its session by (timestamp, id), whatever their visibility."""
    from ._turn_policy import flatten_search_text

    rows = transcripts[row.session.session_id]
    index = next(i for i, candidate in enumerate(rows) if candidate.id == row.id)
    return tuple(SearchContext(r.role, flatten_search_text(r.decoded)[:200]) for r in rows[max(0, index - 1) : index + 2])


def _hit(row: _Row, snippet: Optional[str], transcripts: dict[str, list[_Row]], projection: Optional[tuple[str, ...]]) -> SearchHit:
    values: dict[str, Any] = {
        "id": MessageId(row.id), "session_id": SessionId(row.session.session_id), "role": row.role, "snippet": snippet,
        "timestamp": row.timestamp, "tool_name": row.tool_name, "source": row.session.source, "model": row.session.model,
        "session_started": row.session.started_at,
    }
    wanted = SEARCH_RESULT_FIELDS if projection is None else projection
    if "context" in wanted:
        values["context"] = _context(row, transcripts)
    names = {"id": "message_id"}
    return SearchHit(**{names.get(name, name): values[name] for name in wanted})


register_repository(
    "search",
    InMemorySessionSearch,
    {"full_text_search": True, "substring_search": True},
)
