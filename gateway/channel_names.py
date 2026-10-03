"""Receiving-adapter-owned name snapshots. No disk or network I/O during matching."""

from __future__ import annotations

import time
from dataclasses import dataclass

from gateway.channel_matching import ChannelNames, NameResolver, lazy_names

DIRECTORY_TTL = 300.0


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class _Name:
    canonical: str | None = None
    alias: str | None = None

    def values(self, live=None) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v for v in (_text(live) or self.canonical, self.alias) if v))


class ChannelNameDirectory:
    """Indexes use platform scope + real chat/thread IDs, never split display IDs."""

    def __init__(self, entries, *, now=None, platform=None):
        self.platform = platform
        self.expires_at = (time.monotonic() if now is None else now) + DIRECTORY_TTL
        self.chats: dict[tuple, _Name] = {}
        self.threads: dict[tuple, _Name] = {}
        self.guilds: dict[tuple, _Name] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            scope = _text(entry.get("scope_id") or entry.get("guild_id"))
            thread = _text(entry.get("thread_id"))
            chat = _text(entry.get("chat_id")) or (None if thread else _text(entry.get("id")))
            if not chat:
                continue
            chat_name = _text(entry.get("chat_name")) or (None if thread else _text(entry.get("name")))
            alias = _text(entry.get("alias"))
            self._add(self.chats, (scope, chat), chat_name, alias if not thread or chat == thread else None)
            if thread:
                self._add(self.threads, (scope, chat, thread), _text(entry.get("thread_name")), alias)
            parent = _text(entry.get("parent_chat_id"))
            if parent:
                self._add(self.chats, (scope, parent), _text(entry.get("parent_chat_name")))
            if scope:
                self._add(self.guilds, (scope,), _text(entry.get("guild")))

    @staticmethod
    def _add(index, key, name, alias=None):
        old = index.get(key, _Name())
        # Platform enumeration comes before historical session labels.
        index[key] = _Name(old.canonical or name, alias or old.alias)

    @staticmethod
    def _lookup(index, scope, *ids) -> _Name:
        unscoped = index.get((None, *ids), _Name())
        if scope is None:
            return unscoped
        scoped = index.get((scope, *ids), _Name())
        # Legacy labels with no workspace must not identify a workspace-local ID.
        # Explicit aliases use the existing flat, platform-wide ID namespace.
        return _Name(scoped.canonical, scoped.alias or unscoped.alias)

    def resolve(self, source) -> ChannelNames:
        scope = _text(getattr(source, "scope_id", None) or getattr(source, "guild_id", None))
        chat = str(source.chat_id)
        thread = _text(getattr(source, "thread_id", None))
        parent = _text(getattr(source, "parent_chat_id", None))
        return ChannelNames(
            self._lookup(self.chats, scope, chat).values(getattr(source, "channel_name", None)),
            self._lookup(self.threads, scope, chat, thread).values(getattr(source, "thread_name", None)) if thread else (),
            self._lookup(self.chats, scope, parent).values(getattr(source, "parent_chat_name", None)) if parent else (),
            self.guilds.get((scope,), _Name()).values(getattr(source, "guild_name", None)) if scope else (),
        )


def publish_adapter_directory(adapter, entries, *, platform=None) -> None:
    adapter._routing_channel_directory = ChannelNameDirectory(entries, platform=platform)


def clear_adapter_directory(adapter) -> None:
    # Invalidate any refresh already awaiting platform enumeration.
    adapter._routing_directory_generation = getattr(adapter, "_routing_directory_generation", 0) + 1
    adapter._routing_channel_directory = None


def name_resolver_for_source(source, runner=None) -> NameResolver:
    """Capture the RECEIVING transport before a route changes the runtime profile."""
    cached = getattr(source, "_channel_name_resolver", None)
    if cached is not None:
        return cached
    ref = getattr(source, "_transport_adapter_ref", None)
    adapter = ref() if callable(ref) else None
    if adapter is None and runner is not None:
        from gateway.session_identity import identity_of
        if identity_of(source) is not None:
            adapter = runner._delivery_adapter_for(source)

    def resolve() -> ChannelNames:
        snapshot = getattr(adapter, "_routing_channel_directory", None)
        platform = getattr(source.platform, "value", source.platform)
        if (isinstance(snapshot, ChannelNameDirectory) and snapshot.platform in (None, platform)
                and snapshot.expires_at > time.monotonic()):
            return snapshot.resolve(source)
        # Live metadata remains useful on a cold cache, or before the next directory refresh.
        def one(field):
            value = _text(getattr(source, field, None))
            return (value,) if value else ()
        return ChannelNames(one("channel_name"), one("thread_name"), one("parent_chat_name"), one("guild_name"))

    source._channel_name_resolver = lazy_names(resolve)
    return source._channel_name_resolver
