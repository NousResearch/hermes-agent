"""``SessionSearch`` on SQLite: message search over the FTS5, trigram and CJK indexes and the recovery of those
indexes, through the search store functions and the holder's schema recovery."""

from __future__ import annotations

import logging
from typing import Any, Collection, Optional, Sequence

from persistence import MessageId, SessionId
from persistence.conversation.search import SearchContext, SearchHit, SearchIndexRebuildStatus, SearchSort

from .._backend import SQLiteStore, register_repository
from .._schema import retry_deferred_fts_recovery
from ._store.search import fts_rebuild_status, rebuild_fts, search_messages
from ._errors import refusals

__all__ = ["SQLiteSessionSearch"]

logger = logging.getLogger("gateway.session")

# The store's sort argument per contract order; relevance is its default rank order.
_SORT = {SearchSort.RELEVANCE: None, SearchSort.NEWEST: "newest", SearchSort.OLDEST: "oldest"}


def _optional_list(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    return None if values is None else list(values)


def _hit(match: dict[str, Any]) -> SearchHit:
    """One store match dict as a hit; keys the projection left out stay unset."""
    message_id = match.get("id")
    session_id = match.get("session_id")
    return SearchHit(
        message_id=MessageId(int(message_id)) if message_id is not None else None,
        session_id=SessionId(session_id) if session_id is not None else None,
        role=match.get("role"), snippet=match.get("snippet"), timestamp=match.get("timestamp"),
        tool_name=match.get("tool_name"), source=match.get("source"), model=match.get("model"),
        session_started=match.get("session_started"),
        context=tuple(SearchContext(row["role"], row["content"]) for row in match.get("context") or ()),
    )


class SQLiteSessionSearch:
    """Message search and index recovery over the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def rebuild_search_index(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            foreign_holders = holder._foreign_state_db_holders()
            if foreign_holders:
                logger.warning(
                    "Skipping Session DB FTS rebuild while foreign processes hold the database or "
                    "WAL sidecars (%s); canonical transcript writes remain available.", foreign_holders)
                return 0
            return rebuild_fts(holder)

    def retry_deferred_search_recovery(self) -> bool:
        holder = self._store.holder()
        with refusals(holder):
            return retry_deferred_fts_recovery(holder)

    def search_messages(
        self, query: str, *, source_filter: Optional[Sequence[str]] = None,
        exclude_sources: Optional[Sequence[str]] = None, role_filter: Optional[Sequence[str]] = None,
        limit: int = 20, offset: int = 0, sort: SearchSort = SearchSort.RELEVANCE, include_inactive: bool = False,
        fields: Optional[Collection[str]] = None, after_ts: Optional[int] = None, before_ts: Optional[int] = None,
    ) -> tuple[SearchHit, ...]:
        holder = self._store.holder()
        with refusals(holder):
            matches = search_messages(holder,
                query, source_filter=_optional_list(source_filter), exclude_sources=_optional_list(exclude_sources),
                role_filter=_optional_list(role_filter), limit=limit, offset=offset, sort=_SORT[sort],
                include_inactive=include_inactive, fields=fields, after_ts=after_ts, before_ts=before_ts)
        return tuple(_hit(match) for match in matches)

    def search_index_rebuild_status(self) -> Optional[SearchIndexRebuildStatus]:
        holder = self._store.holder()
        with refusals(holder):
            status = fts_rebuild_status(holder)
        return None if status is None else SearchIndexRebuildStatus(total=status["total"], indexed=status["indexed"])


register_repository("search", SQLiteSessionSearch, {"full_text_search": True, "substring_search": True})
