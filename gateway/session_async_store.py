"""Async calls into the synchronous, thread-safe gateway SessionStore."""

from __future__ import annotations

import asyncio
from typing import Any


class AsyncSessionStore:
    """Async boundary for the synchronous, thread-safe SessionStore."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str):
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        async def _offloaded(*args, **kwargs) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _offloaded

    async def append_to_transcript(self, session_id: str, message: dict[str, Any], skip_db: bool = False) -> None:
        """Skip executor dispatch when the sync store's documented skip_db path is a no-op."""
        if skip_db:
            return None
        return await asyncio.to_thread(self._store.append_to_transcript, session_id, message, skip_db)
