"""Keep Hermes's own auto-thread title rename out of the pinned session-context prompt.

``chat_name`` keys the pinned prompt, and the title lane renames a new auto-thread between its
first and second turns. While the thread carries Hermes's title, its chat name keeps the name the
title replaced. Any other observed name retires the record: equal text is not edit provenance, so
a moderator who later restores Hermes's title sees it as itself. Records live in memory; after a
restart each renamed thread re-renders once.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    import discord
except ImportError:  # the adapter does not start without discord.py
    discord = None

_MAX_RECORDS = 2000


@dataclass
class _TitleRename:
    replaced: str
    title: str
    last_event_name: str
    title_seen: bool = False


class SemanticThreadRenames:
    """Hermes's latest guarded title rename per thread id."""

    def __init__(self) -> None:
        self._records: dict[str, _TitleRename] = {}

    @contextmanager
    def attempt(self, thread_id: str, replaced: str, title: str) -> Iterator[None]:
        """Record a rename before its REST edit, which gateway events can overtake; drop it if the edit fails."""
        record = _TitleRename(replaced, title, last_event_name=replaced)
        self._records.pop(thread_id, None)  # re-insert last: eviction is oldest first
        self._records[thread_id] = record
        while len(self._records) > _MAX_RECORDS:
            del self._records[next(iter(self._records))]
        try:
            yield
        except BaseException:
            # Cancellation included. A newer attempt owns a different record and keeps it.
            if self._records.get(thread_id) is record:
                del self._records[thread_id]
            raise

    def observe_event(self, thread_id: str, name: str) -> None:
        """Apply one gateway rename event; only Hermes's own replaced -> title step keeps the record."""
        record = self._records.get(thread_id)
        if record is None or name == record.last_event_name:
            return
        if (record.last_event_name, name) == (record.replaced, record.title):
            record.last_event_name = name
        else:
            del self._records[thread_id]

    def display_name(self, thread_id: str, name: str) -> str:
        """The name to render for a thread currently called *name*."""
        record = self._records.get(thread_id)
        if record is None:
            return name
        if name == record.title:
            record.title_seen = True
            return record.replaced
        # Until the title has been seen, the replaced name may come from a cache that lags the edit.
        if record.title_seen or name != record.replaced:
            del self._records[thread_id]
        return name


class DiscordThreadTitlesMixin:
    _semantic_thread_renames: SemanticThreadRenames
    _is_forum_parent: Callable[[Any], bool]
    _get_effective_topic: Callable[..., str | None]

    async def _on_platform_raw_thread_update(self, payload: Any) -> None:
        # discord.py updates the cached thread before cached callbacks run, so batched updates
        # would all report the final name; each raw payload keeps its own.
        name = payload.data.get("name")
        if isinstance(name, str):
            self._semantic_thread_renames.observe_event(str(payload.thread_id), name)

    def _format_thread_chat_name(self, thread: Any) -> str:
        """Build a readable chat name for thread-like Discord channels, including forum context when available."""
        thread_name = getattr(thread, "name", None) or str(getattr(thread, "id", "thread"))
        thread_name = self._semantic_thread_renames.display_name(str(getattr(thread, "id", "")), thread_name)
        parent = getattr(thread, "parent", None)
        guild = getattr(thread, "guild", None) or getattr(parent, "guild", None)
        guild_name = getattr(guild, "name", None)
        parent_name = getattr(parent, "name", None)
        if self._is_forum_parent(parent) and guild_name and parent_name:
            return f"{guild_name} / {parent_name} / {thread_name}"
        if parent_name and guild_name:
            return f"{guild_name} / #{parent_name} / {thread_name}"
        if parent_name:
            return f"{parent_name} / {thread_name}"
        return thread_name

    def _guild_channel_labels(self, channel: Any) -> tuple[str, str | None]:
        """``(chat_name, chat_topic)`` for a guild channel or thread, as a message posted there gets them."""
        if isinstance(channel, discord.Thread):
            return self._format_thread_chat_name(channel), self._get_effective_topic(channel, is_thread=True)
        name, guild = getattr(channel, "name", str(channel.id)), getattr(channel, "guild", None)
        return (f"{guild.name} / #{name}" if guild else name), self._get_effective_topic(channel)
