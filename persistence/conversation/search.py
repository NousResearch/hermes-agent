"""Message search over one scope's transcripts, and the recovery of the index that serves it.

The query is the text a user typed, in the store's search syntax: words (implicitly ANDed), ``"quoted phrases"``, uppercase ``AND`` / ``OR`` / ``NOT`` (tightest first: adjacent terms, ``NOT``, ``AND``, ``OR``) and ``prefix*``. The backend sanitizes it before matching: text past ``MAX_SEARCH_QUERY_CHARS`` is cut, balanced quoted phrases are kept as written and a stray quote is whitespace, the characters ``+ { } ( ) : ^ @ / # & | ~ [ ] < > , ; ! ? $ = \\ '`` are whitespace (``%`` too unless the text holds CJK), repeated ``*`` collapse and a leading ``*`` is dropped, one leading and one trailing operator word of any case are dropped, and a word joined by ``.``, ``_`` or ``-`` (``chat-send``, ``P2.2``, ``my-app.config.ts``) becomes a quoted phrase. Blank text, or text that sanitizes to nothing, matches nothing; so does text the sanitized syntax still rejects (``a OR NOT b``, ``-quick``, ``quick.``, a NUL), with no retry.

Which rows match, by route:

- A word matches whole words of a message's content, tool name or tool-call JSON, case-insensitively and without Latin diacritics (``cafe`` matches ``Café``); ``_`` splits words. Tool content counts only up to its first ``FTS_TOOL_CONTENT_PREFIX_CHARS`` characters. Structured content is searched as its stored ``\\x00json:`` JSON text, so a CJK string inside it is searchable only by its ``\\uXXXX`` escape.
- When that finds nothing (and the query holds no CJK and no ``tool`` role filter), a query whose words all have three or more characters is retried with each word as a case-folded substring of the content or tool name of rows that are not tool rows and not in ``cron`` or ``subagent`` sessions or delegate children (``cat`` then matches ``concatenate``); when that also finds nothing, an implicit-AND query of two or more units without ``OR`` / ``NOT`` is retried with its units ORed. Each retry keeps the filters, sort, limit and offset, so an offset past the end of a smaller result reaches the next retry.
- A ``tool`` role filter reads the stored rows by substring: words of one ``OR`` group must all appear (``AND`` ignored, ``NOT`` excludes the next word, quotes and ``*`` stripped from each word), matched ASCII-case-insensitively against content, tool name and tool-call JSON, where structured content never matches (its stored text starts with NUL).
- CJK text: with three or more CJK characters in every CJK word and no filter reaching rows the substring index skips (``tool`` role, ``cron`` / ``subagent`` source), the words match as case-folded substrings of non-tool rows outside ``cron`` / ``subagent`` sessions and delegate children, combined by the operators (a word under three characters matches every row); otherwise any one word appearing as an ASCII-case-insensitive substring of content, tool name or tool-call JSON matches (operators ignored, trailing ``*`` dropped).

Filters apply on every route: rewound rows (neither active nor compaction-archived) are hidden unless ``include_inactive``; ``display_kind == "hidden"`` rows never match; ``source_filter`` keeps sessions of those sources (an empty sequence keeps none), ``exclude_sources`` drops them, a non-empty ``role_filter`` keeps those roles; ``after_ts`` (inclusive) and ``before_ts`` (exclusive) bound the owning session's ``started_at``.

Order: ``NEWEST`` / ``OLDEST`` order by message timestamp; equal timestamps are ordered by relevance on the word and substring routes, by message id in the same direction on the ``tool`` role route, and not at all on the CJK any-word route. ``RELEVANCE`` order is the backend's. The ``tool`` role route orders newest first for ``RELEVANCE``; the CJK any-word route always orders newest first. ``limit`` and ``offset`` are applied after ordering (a negative limit is no limit, a negative offset is zero).

A hit carries the row id, session id, role, snippet, timestamp, tool name, the session's source, model and start time, and ``context``: the previous row, the row itself and the next row of its session by (timestamp, id), whatever their visibility, each as role and text (structured parts flattened, cut to 200 characters). A word or substring route snippet is an excerpt of the matched field with each match between ``>>>`` and ``<<<`` and ``...`` where text was cut; the composition of the excerpt is the backend's. An any-word or ``tool`` role snippet is 120 characters of content starting 40 characters before the first case-sensitive occurrence of the first matching word (from the start when it does not occur), without markers. ``fields`` projects the hit to the named fields (``id``, ``session_id``, ``role``, ``snippet``, ``timestamp``, ``tool_name``, ``source``, ``model``, ``session_started``, ``context``); the others are ``None`` (``()`` for context) and context is not read unless named.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Collection, Optional, Protocol, Sequence, runtime_checkable

from ..core.identity import MessageId, SessionId

__all__ = [
    "FTS_TOOL_CONTENT_PREFIX_CHARS",
    "MAX_SEARCH_QUERY_CHARS",
    "SEARCH_RESULT_FIELDS",
    "SearchContext",
    "SearchHit",
    "SearchIndexRebuildStatus",
    "SearchSort",
    "SessionSearch",
]

#: Longest query text honoured; the rest is cut before sanitizing.
MAX_SEARCH_QUERY_CHARS = 2_048
#: Characters of a tool row's content the word index covers.
FTS_TOOL_CONTENT_PREFIX_CHARS = 8_192
#: The hit fields ``fields`` may name, in their canonical order.
SEARCH_RESULT_FIELDS = ("id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model", "session_started", "context")


class SearchSort(Enum):
    """Hit order: the backend's relevance, or message timestamp newest or oldest first."""

    RELEVANCE = "relevance"
    NEWEST = "newest"
    OLDEST = "oldest"


@dataclass(frozen=True, slots=True)
class SearchContext:
    """One row of a hit's context window: its role and its text cut to 200 characters."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One matching row; a field ``fields`` did not name is ``None`` (``context`` is ``()``). Times are epoch-second floats."""

    message_id: Optional[MessageId] = None
    session_id: Optional[SessionId] = None
    role: Optional[str] = None
    snippet: Optional[str] = None
    timestamp: Optional[float] = None
    tool_name: Optional[str] = None
    source: Optional[str] = None
    model: Optional[str] = None
    session_started: Optional[float] = None
    context: tuple[SearchContext, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchIndexRebuildStatus:
    """Progress of a pending deferred index rebuild: ``total`` rows to cover (positive), ``indexed`` as the stored progress reads (not clamped)."""

    total: int
    indexed: int

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("SearchIndexRebuildStatus.total must be positive; no rebuild is pending otherwise")

    @property
    def percent(self) -> int:
        """``min(100, int(100 * indexed / total))``."""
        return min(100, int(100 * self.indexed / self.total))


@runtime_checkable
class SessionSearch(Protocol):
    """Message search over one scope plus the recovery of its index (module docstring: query syntax, routes, filters, order, hit shape)."""

    def rebuild_search_index(self) -> int:
        """Rebuild every search index of the store from the stored rows; returns the number rebuilt. Two processes never rebuild one store at once: admission waits a bounded time and returns 0 without touching the index when denied, and 0 also when another process holds the store open. Each index rebuilds in its own commit; one that fails is logged and not counted, so a read-only store returns 0. A pending deferred rebuild stays pending. A corrupt or replaced store raises ``CorruptStoreError`` / ``StoreReplaced``."""
        ...

    def retry_deferred_search_recovery(self) -> bool:
        """One non-blocking attempt to finish the index recovery this handle deferred at open; True only when the index was rebuilt and serves again. False when nothing was deferred, inside the backoff window (restarted when the set of other processes holding the store changes), when admission is denied, on a read-only or quarantined store, or when the attempt fails. Never raises."""
        ...

    def search_messages(self, query: str, *, source_filter: Optional[Sequence[str]] = None, exclude_sources: Optional[Sequence[str]] = None, role_filter: Optional[Sequence[str]] = None, limit: int = 20, offset: int = 0, sort: SearchSort = SearchSort.RELEVANCE, include_inactive: bool = False, fields: Optional[Collection[str]] = None, after_ts: Optional[int] = None, before_ts: Optional[int] = None) -> tuple[SearchHit, ...]:
        """The hits of ``query`` under the filters, ordered by ``sort`` and paged by ``limit`` / ``offset`` (module docstring). ``fields`` given as a string raises ``TypeError`` and naming an unknown field ``ValueError``, both before the query is read; a query matching nothing returns ``()``."""
        ...

    def search_index_rebuild_status(self) -> Optional[SearchIndexRebuildStatus]:
        """Progress of the deferred rebuild, or ``None`` when none is pending; read without waiting on writers, and served by a read-only store."""
        ...
