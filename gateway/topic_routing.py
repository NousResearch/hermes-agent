"""Strict exact-match topic routing for owner-private forum-topic gateways.

This module is the generic core of "topic mode": a deterministic, exact
``(owner_chat_id, message_thread_id)`` route registry that is evaluated
BEFORE session creation, text parsing, media download, callback lookup, or
tool access. It is platform-agnostic and deliberately tiny:

- Exact-match only. There is NO default route, NO "lobby" behavior, NO
  last-active-topic recovery, and NO fallback of any kind. A missing,
  zero, foreign, unknown, deleted, or unregistered thread id fails closed
  with a stable value-free reason code.
- The private "General" topic of a Telegram forum-mode DM is the
  non-deletable thread id ``1`` (:data:`GENERAL_TOPIC_THREAD_ID`).
- Each registered route carries a stable per-profile namespace so prompt /
  tool identity stays constant for the lifetime of a session (prompt-cache
  friendly: routing never swaps a session's prompt or toolset).
- Outbound actions are origin-bound: :class:`RouteOrigin` freezes the
  ``(bot_id, owner_chat_id, thread_id, update_id, message_id)`` tuple of
  the inbound update so replies, edits, media, callbacks, and receipts can
  be pinned to the registered origin thread and nothing else.

Privacy: reason codes and route/profile names are the only strings this
module ever emits; no message content, captions, or media metadata pass
through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "GENERAL_TOPIC_THREAD_ID",
    "RouteDenied",
    "TopicRoute",
    "RouteOrigin",
    "TopicRouteRegistry",
]

#: Telegram's non-deletable private "General" forum topic id. In the Sol
#: deployment this topic is renamed "Sol"; the id is what routing keys on.
GENERAL_TOPIC_THREAD_ID = 1

# Stable, value-free denial reason codes (safe to log).
REASON_MISSING_THREAD = "topic_route_missing_thread"
REASON_INVALID_THREAD = "topic_route_invalid_thread"
REASON_FOREIGN_CHAT = "topic_route_foreign_chat"
REASON_UNKNOWN_THREAD = "topic_route_unknown_thread"


class RouteDenied(Exception):
    """Raised when an update cannot be routed. Always fail-closed.

    ``reason_code`` is a stable value-free string suitable for logs.
    """

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class TopicRoute:
    """One registered destination: an exact (owner chat, thread) pair.

    ``profile`` is the stable per-topic namespace (system prompt identity,
    session key component, tool allowlist root). It must never change for
    a live route; sessions rely on it for prompt-cache stability.
    """

    owner_chat_id: str
    thread_id: int
    profile: str

    @property
    def key(self) -> Tuple[str, int]:
        return (self.owner_chat_id, self.thread_id)


@dataclass(frozen=True)
class RouteOrigin:
    """Frozen origin tuple of one inbound update.

    Every outbound action produced while handling this update (send, edit,
    typing, upload, retry, acknowledgement, callback answer, receipt) must
    be bound to exactly this tuple. There is no fallback destination.
    """

    bot_id: str
    owner_chat_id: str
    thread_id: int
    update_id: int
    message_id: int


def _coerce_thread_id(raw: Any) -> Optional[int]:
    """Return a positive int thread id, or None if absent/invalid.

    Note: ``0`` and negative values are invalid (Telegram thread ids are
    positive; General is 1) and are treated as *invalid*, not missing.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is an int subclass; reject explicitly
        raise RouteDenied(REASON_INVALID_THREAD)
    if isinstance(raw, int):
        if raw < 1:
            raise RouteDenied(REASON_INVALID_THREAD)
        return raw
    if isinstance(raw, str) and raw.isdigit():
        value = int(raw)
        if value < 1:
            raise RouteDenied(REASON_INVALID_THREAD)
        return value
    raise RouteDenied(REASON_INVALID_THREAD)


class TopicRouteRegistry:
    """Immutable exact-match route registry.

    Built once from validated configuration entries; resolution is a pure
    dictionary lookup with fail-closed semantics. The registry never grows
    at runtime and never consults history, session state, or "last active"
    anything.
    """

    def __init__(self, routes: Iterable[TopicRoute]) -> None:
        table: Dict[Tuple[str, int], TopicRoute] = {}
        profiles: Dict[str, TopicRoute] = {}
        chats: set[str] = set()
        for route in routes:
            if not route.owner_chat_id or not isinstance(route.owner_chat_id, str):
                raise ValueError("topic route requires a non-empty owner_chat_id string")
            if not isinstance(route.thread_id, int) or isinstance(route.thread_id, bool) or route.thread_id < 1:
                raise ValueError("topic route requires a positive integer thread_id")
            if not route.profile or not isinstance(route.profile, str):
                raise ValueError("topic route requires a non-empty profile name")
            if route.key in table:
                raise ValueError("duplicate topic route key")
            if route.profile in profiles:
                raise ValueError("duplicate topic route profile")
            table[route.key] = route
            profiles[route.profile] = route
            chats.add(route.owner_chat_id)
        if not table:
            raise ValueError("topic route registry requires at least one route")
        self._table = table
        self._profiles = profiles
        self._chats = frozenset(chats)

    @classmethod
    def from_config(cls, entries: Iterable[Mapping[str, Any]]) -> "TopicRouteRegistry":
        """Build from config dicts: ``{chat_id, thread_id, profile}``.

        Strict: unknown value shapes raise ``ValueError`` at startup so a
        misconfigured registry can never silently route.
        """
        routes: List[TopicRoute] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("topic route entry must be a mapping")
            unknown = set(entry.keys()) - {"chat_id", "thread_id", "profile"}
            if unknown:
                raise ValueError("topic route entry has unknown keys")
            chat_id = entry.get("chat_id")
            thread_id = entry.get("thread_id")
            profile = entry.get("profile")
            if chat_id is None or thread_id is None or profile is None:
                raise ValueError("topic route entry requires chat_id, thread_id, profile")
            if isinstance(thread_id, bool) or not isinstance(thread_id, int):
                raise ValueError("topic route thread_id must be an integer")
            routes.append(
                TopicRoute(
                    owner_chat_id=str(chat_id),
                    thread_id=thread_id,
                    profile=str(profile),
                )
            )
        return cls(routes)

    def resolve(self, chat_id: Any, thread_id: Any) -> TopicRoute:
        """Resolve an inbound (chat, thread) to its registered route.

        Fail-closed contract:

        - ``thread_id`` absent            -> RouteDenied(missing_thread)
        - ``thread_id`` malformed / < 1   -> RouteDenied(invalid_thread)
        - chat not in the registry        -> RouteDenied(foreign_chat)
        - (chat, thread) not registered   -> RouteDenied(unknown_thread)

        There is intentionally no way to obtain a route without an exact
        registered match.
        """
        chat_key = str(chat_id) if chat_id is not None else ""
        if chat_key not in self._chats:
            raise RouteDenied(REASON_FOREIGN_CHAT)
        thread = _coerce_thread_id(thread_id)
        if thread is None:
            raise RouteDenied(REASON_MISSING_THREAD)
        route = self._table.get((chat_key, thread))
        if route is None:
            raise RouteDenied(REASON_UNKNOWN_THREAD)
        return route

    def route_for_profile(self, profile: str) -> Optional[TopicRoute]:
        return self._profiles.get(profile)

    def profiles(self) -> Tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def __len__(self) -> int:
        return len(self._table)
