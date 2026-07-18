"""Strict owner-private Telegram chat routing.

This is the unthreaded counterpart to :mod:`gateway.topic_routing`.  It is
intentionally small: one gateway process is bound to one Telegram bot
identity, one private chat, one owner user, and one stable profile.  There is
no default profile, pairing fallback, group admission, or thread recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from gateway.topic_routing import RouteDenied

__all__ = ["PrivateChatRoute", "PrivateChatRouteRegistry"]

REASON_FOREIGN_CHAT = "private_route_foreign_chat"
REASON_FOREIGN_USER = "private_route_foreign_user"
REASON_NOT_PRIVATE = "private_route_not_private"
REASON_THREADED = "private_route_unexpected_thread"


@dataclass(frozen=True)
class PrivateChatRoute:
    """One exact unthreaded owner-private destination."""

    owner_chat_id: str
    owner_user_id: str
    profile: str
    expected_bot_id: str
    expected_bot_username: str

    @property
    def thread_id(self) -> None:
        """Private chats deliberately have no synthetic thread identifier."""

        return None


class PrivateChatRouteRegistry:
    """Immutable single-route registry for a dedicated private-chat bot."""

    def __init__(self, route: PrivateChatRoute) -> None:
        for name in (
            "owner_chat_id",
            "owner_user_id",
            "profile",
            "expected_bot_id",
            "expected_bot_username",
        ):
            value = getattr(route, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"private chat route requires non-empty {name}")
        if not route.owner_chat_id.lstrip("-").isdigit():
            raise ValueError("private chat owner_chat_id must be an integer string")
        if not route.owner_user_id.isdigit():
            raise ValueError("private chat owner_user_id must be a positive integer string")
        if not route.expected_bot_id.isdigit():
            raise ValueError("private chat expected_bot_id must be a positive integer string")
        username = route.expected_bot_username
        if username.startswith("@") or username.casefold() != username:
            raise ValueError(
                "private chat expected_bot_username must be lowercase without @"
            )
        self._route = route

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "PrivateChatRouteRegistry":
        if not isinstance(raw, Mapping):
            raise ValueError("private_chat_routing must be a mapping")
        expected_keys = {
            "mode",
            "chat_id",
            "user_id",
            "profile",
            "expected_bot_id",
            "expected_bot_username",
            "hooks",
        }
        if set(raw) - expected_keys:
            raise ValueError("private_chat_routing has unknown keys")
        if raw.get("mode") != "strict":
            raise ValueError("private_chat_routing.mode must be 'strict'")
        required = expected_keys - {"hooks"}
        if any(raw.get(key) is None for key in required):
            raise ValueError("private_chat_routing is missing a required key")
        return cls(
            PrivateChatRoute(
                owner_chat_id=str(raw["chat_id"]),
                owner_user_id=str(raw["user_id"]),
                profile=str(raw["profile"]),
                expected_bot_id=str(raw["expected_bot_id"]),
                expected_bot_username=str(raw["expected_bot_username"]),
            )
        )

    @property
    def route(self) -> PrivateChatRoute:
        return self._route

    def route_for_profile(self, profile: str) -> Optional[PrivateChatRoute]:
        return self._route if profile == self._route.profile else None

    def resolve(
        self,
        *,
        chat_id: Any,
        chat_type: Any,
        user_id: Any,
        thread_id: Any,
    ) -> PrivateChatRoute:
        if str(chat_type or "").casefold() != "private":
            raise RouteDenied(REASON_NOT_PRIVATE)
        if thread_id is not None:
            raise RouteDenied(REASON_THREADED)
        if str(chat_id or "") != self._route.owner_chat_id:
            raise RouteDenied(REASON_FOREIGN_CHAT)
        if str(user_id or "") != self._route.owner_user_id:
            raise RouteDenied(REASON_FOREIGN_USER)
        return self._route

    def resolve_destination(self, *, chat_id: Any, thread_id: Any) -> PrivateChatRoute:
        """Validate an outbound destination without inventing caller identity."""

        if thread_id is not None:
            raise RouteDenied(REASON_THREADED)
        if str(chat_id or "") != self._route.owner_chat_id:
            raise RouteDenied(REASON_FOREIGN_CHAT)
        return self._route

    def validate_bot_identity(self, *, bot_id: Any, username: Any) -> None:
        """Fail startup closed when a token belongs to the wrong named bot."""

        if str(bot_id or "") != self._route.expected_bot_id:
            raise ValueError("private chat Telegram bot id does not match sealed identity")
        normalized = str(username or "").removeprefix("@").casefold()
        if normalized != self._route.expected_bot_username:
            raise ValueError(
                "private chat Telegram bot username does not match sealed identity"
            )
