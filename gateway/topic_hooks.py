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
    "register_topic_hook_factory",
    "clear_topic_hook_factories",
    "create_topic_hook",
]

# Stable value-free reason codes.
REASON_HOOK_ERROR = "topic_hook_error"
REASON_HOOK_DENIED = "topic_hook_denied"

#: Origin-bound reply callable supplied by the platform adapter. The
#: adapter constructs it already pinned to the origin chat + thread (and,
#: where supported, as a reply to the origin message); hooks cannot select
#: any other destination.
ReplyFn = Callable[[str], Awaitable[None]]
TopicHookFactory = Callable[[], "TopicPluginHook"]


@dataclass(frozen=True)
class _FactoryRegistration:
    owner: str
    factory: TopicHookFactory


# Process-global factory declarations are inert.  A factory is called only
# when an operator explicitly lists its profile in
# ``platforms.telegram.extra.topic_routing.hooks``.  This lets bundled plugins
# advertise a topic integration without reading credentials, creating state,
# or changing legacy gateways during plugin discovery.
_TOPIC_HOOK_FACTORIES: Dict[str, _FactoryRegistration] = {}


def clear_topic_hook_factories() -> None:
    """Clear lazy declarations before an explicit full plugin rescan."""
    _TOPIC_HOOK_FACTORIES.clear()


def register_topic_hook_factory(
    profile: str,
    factory: TopicHookFactory,
    *,
    owner: str,
) -> None:
    """Register one lazy topic-hook factory for ``profile``.

    A plugin owner may replace its own registration during an explicit plugin
    rescan.  A different owner cannot claim the same profile.  Registration is
    side-effect free: the factory is not called here.
    """
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("topic hook factory requires a non-empty profile")
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("topic hook factory requires a non-empty owner")
    if not callable(factory):
        raise ValueError("topic hook factory must be callable")
    profile = profile.strip()
    owner = owner.strip()
    existing = _TOPIC_HOOK_FACTORIES.get(profile)
    if existing is not None and existing.owner != owner:
        raise ValueError("duplicate topic hook factory profile")
    _TOPIC_HOOK_FACTORIES[profile] = _FactoryRegistration(owner, factory)


def create_topic_hook(profile: str, *, owner: str) -> "TopicPluginHook":
    """Instantiate one explicitly enabled hook, failing closed.

    The returned hook must bind to the requested profile.  Factories cannot
    silently redirect an operator's configured namespace.
    """
    registration = _TOPIC_HOOK_FACTORIES.get(profile)
    if registration is None:
        raise ValueError("configured topic hook profile has no registered factory")
    if registration.owner != owner:
        raise ValueError("configured topic hook factory owner mismatch")
    hook = registration.factory()
    if not isinstance(hook, TopicPluginHook):
        raise ValueError("topic hook factory returned an invalid hook")
    if hook.profile != profile:
        raise ValueError("topic hook factory profile mismatch")
    return hook


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
    override any of the ``on_*`` methods. Defaults are pass-through
    (CONTINUE), so a hook only owns what it explicitly implements.
    """

    #: Route profile this hook is bound to (must exist in the registry).
    profile: str = ""

    #: Callback-data prefixes owned by this hook (e.g. ``("sf1:",)``).
    callback_prefixes: Tuple[str, ...] = ()

    async def start(self) -> None:
        """Start background work after the owning adapter is connected."""

    async def stop(self) -> None:
        """Stop and await background work before the adapter disconnects."""

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

    async def on_media_downloaded(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        media: MediaDescriptor,
        content: bytes,
        caption: Optional[str],
        reply: ReplyFn,
    ) -> HookDecision:
        """Inspect already bounded bytes after the pre-download gate."""
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
        self._started = False

    def register(self, hook: TopicPluginHook) -> None:
        if self._started:
            raise RuntimeError("cannot register topic hook after lifecycle start")
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

    async def start(self) -> None:
        """Start each registered hook once, in registration order."""
        if self._started:
            return
        started: List[TopicPluginHook] = []
        try:
            for hook in self._by_profile.values():
                await hook.start()
                started.append(hook)
        except Exception:
            for hook in reversed(started):
                try:
                    await hook.stop()
                except Exception:
                    logger.warning("[topic-hooks] %s", REASON_HOOK_ERROR)
            raise
        self._started = True

    async def stop(self) -> None:
        """Stop each started hook once, in reverse registration order."""
        if not self._started:
            return
        self._started = False
        for hook in reversed(tuple(self._by_profile.values())):
            try:
                await hook.stop()
            except Exception:
                logger.warning("[topic-hooks] %s", REASON_HOOK_ERROR)

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

    async def dispatch_media_downloaded(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        media: MediaDescriptor,
        content: bytes,
        caption: Optional[str],
        reply: ReplyFn,
    ) -> HookDecision:
        hook = self._by_profile.get(route.profile)
        if hook is None:
            return HookDecision.CONTINUE
        return await self._guarded(
            hook.on_media_downloaded(route, origin, media, content, caption, reply)
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
