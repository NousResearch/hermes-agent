"""Authenticated post-route plugin seams for strict topic routing.

These hooks are the ONLY extension points strict topic mode offers, and
they run at exactly two moments:

1. after the inbound update has been owner-authenticated AND resolved to a
   registered :class:`~gateway.topic_routing.TopicRoute` (post-route), and
2. for media, additionally BEFORE any byte of the file is downloaded
   (pre-download).

Design rules (mirrored by the dispatch code below):

- A hook is bound to exactly one route profile at registration time. It is
  never consulted for updates routed to any other profile, so per-profile
  behavior (e.g. a food workflow on one topic) cannot leak across topics.
- Hooks cannot change, widen, or re-target the resolved route. Dispatch
  passes frozen dataclasses only.
- Hook outcomes are a closed enum: CONTINUE (generic pipeline proceeds),
  CONSUME (the hook owns the interaction; the generic pipeline stops), and
  DENY (fail closed; the generic pipeline stops and nothing downloads or
  executes).
- A raising hook is a DENY. Errors are logged as stable reason codes only;
  no update content, captions, labels, callback data, or paths are logged.
- Replies a hook wants to send go through the origin-bound ``reply``
  callable provided by the platform adapter, which pins delivery to the
  origin ``(owner_chat_id, thread_id)``; hooks never pick destinations.
- Callback routing is prefix-based and stateless in the seam itself, so a
  process restart cannot orphan a callback: whichever hook owns the prefix
  re-resolves the token against its own durable state.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple

from gateway.topic_routing import RouteOrigin, TopicRoute

logger = logging.getLogger(__name__)

__all__ = [
    "HookDecision",
    "MediaDescriptor",
    "TopicPluginHook",
    "TopicHookRegistry",
]

# Stable value-free reason codes.
REASON_HOOK_ERROR = "topic_hook_error"
REASON_HOOK_DENIED = "topic_hook_denied"

#: Origin-bound reply callable supplied by the platform adapter. The
#: adapter constructs it already pinned to the origin chat + thread (and,
#: where supported, as a reply to the origin message); hooks cannot select
#: any other destination.
ReplyFn = Callable[[str], Awaitable[None]]


class HookDecision(enum.Enum):
    CONTINUE = "continue"
    CONSUME = "consume"
    DENY = "deny"


@dataclass(frozen=True)
class MediaDescriptor:
    """Pre-download metadata for one inbound media update.

    Everything here comes from the platform's update object, not from file
    bytes: hooks use it to fail closed BEFORE any download happens.
    ``media_group_id`` is non-None when the update is part of an album.
    Unknown fields are None rather than guessed.
    """

    kind: str  # "photo", "video", "document", "audio", "sticker", ...
    file_size: Optional[int]
    width: Optional[int]
    height: Optional[int]
    media_group_id: Optional[str]
    mime_type: Optional[str] = None
    caption_length: int = 0


class TopicPluginHook:
    """Base class for per-profile topic hooks.

    Subclasses set :attr:`profile` and :attr:`callback_prefixes` and
    override any of the three ``on_*`` methods. Defaults are pass-through
    (CONTINUE), so a hook only owns what it explicitly implements.
    """

    #: Route profile this hook is bound to (must exist in the registry).
    profile: str = ""

    #: Callback-data prefixes owned by this hook (e.g. ``("sf1:",)``).
    callback_prefixes: Tuple[str, ...] = ()

    async def on_message(
        self, route: TopicRoute, origin: RouteOrigin, text: str, reply: ReplyFn
    ) -> HookDecision:
        return HookDecision.CONTINUE

    async def on_media_pre_download(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        media: MediaDescriptor,
        reply: ReplyFn,
    ) -> HookDecision:
        return HookDecision.CONTINUE

    async def on_callback(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        callback_data: str,
        reply: ReplyFn,
    ) -> HookDecision:
        return HookDecision.CONTINUE


class TopicHookRegistry:
    """Registry + fail-closed dispatcher for :class:`TopicPluginHook`.

    One hook per profile; one owner per callback prefix. Dispatch requires
    an already-resolved route (there is no way to reach a hook without
    routing first, which is what makes the seam "authenticated
    post-route").
    """

    def __init__(self) -> None:
        self._by_profile: Dict[str, TopicPluginHook] = {}
        self._by_prefix: Dict[str, TopicPluginHook] = {}

    def register(self, hook: TopicPluginHook) -> None:
        if not hook.profile or not isinstance(hook.profile, str):
            raise ValueError("topic hook requires a non-empty profile binding")
        if hook.profile in self._by_profile:
            raise ValueError("duplicate topic hook profile")
        for prefix in hook.callback_prefixes:
            if not prefix or not isinstance(prefix, str):
                raise ValueError("topic hook callback prefix must be a non-empty string")
            if prefix in self._by_prefix:
                raise ValueError("duplicate topic hook callback prefix")
        self._by_profile[hook.profile] = hook
        for prefix in hook.callback_prefixes:
            self._by_prefix[prefix] = hook

    def hook_for(self, route: TopicRoute) -> Optional[TopicPluginHook]:
        return self._by_profile.get(route.profile)

    def hook_for_callback(self, callback_data: str) -> Optional[TopicPluginHook]:
        for prefix, hook in self._by_prefix.items():
            if callback_data.startswith(prefix):
                return hook
        return None

    def owns_callback(self, callback_data: str) -> bool:
        return self.hook_for_callback(callback_data) is not None

    async def dispatch_message(
        self, route: TopicRoute, origin: RouteOrigin, text: str, reply: ReplyFn
    ) -> HookDecision:
        hook = self._by_profile.get(route.profile)
        if hook is None:
            return HookDecision.CONTINUE
        return await self._guarded(hook.on_message(route, origin, text, reply))

    async def dispatch_media_pre_download(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        media: MediaDescriptor,
        reply: ReplyFn,
    ) -> HookDecision:
        hook = self._by_profile.get(route.profile)
        if hook is None:
            return HookDecision.CONTINUE
        return await self._guarded(
            hook.on_media_pre_download(route, origin, media, reply)
        )

    async def dispatch_callback(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        callback_data: str,
        reply: ReplyFn,
    ) -> HookDecision:
        hook = self.hook_for_callback(callback_data)
        if hook is None:
            return HookDecision.CONTINUE
        if hook.profile != route.profile:
            # A callback prefix owned by another topic's hook is a foreign
            # action against this route: fail closed, never cross-dispatch.
            logger.warning("[topic-hooks] %s", REASON_HOOK_DENIED)
            return HookDecision.DENY
        return await self._guarded(
            hook.on_callback(route, origin, callback_data, reply)
        )

    @staticmethod
    async def _guarded(awaitable: Awaitable[HookDecision]) -> HookDecision:
        try:
            decision = await awaitable
        except Exception:
            # Fail closed. Reason code only — never hook exception content
            # (it could embed user text or paths).
            logger.warning("[topic-hooks] %s", REASON_HOOK_ERROR)
            return HookDecision.DENY
        if not isinstance(decision, HookDecision):
            logger.warning("[topic-hooks] %s", REASON_HOOK_ERROR)
            return HookDecision.DENY
        return decision
