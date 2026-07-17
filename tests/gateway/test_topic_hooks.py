"""Topic plugin hook seams: authenticated post-route, fail-closed."""

import pytest

from gateway.topic_hooks import (
    HookDecision,
    MediaDescriptor,
    TopicHookRegistry,
    TopicPluginHook,
)
from gateway.topic_routing import RouteOrigin, TopicRoute

SOL = TopicRoute("208214988", 1, "sol")
ATLAS = TopicRoute("208214988", 77, "atlas")
ORIGIN = RouteOrigin("bot1", "208214988", 1, 1000, 42)


async def _noop_reply(text: str) -> None:
    return None


class RecordingHook(TopicPluginHook):
    profile = "sol"
    callback_prefixes = ("sf1:",)

    def __init__(self, decision=HookDecision.CONSUME):
        self.decision = decision
        self.calls = []

    async def on_message(self, route, origin, text, reply):
        self.calls.append(("message", route.profile))
        return self.decision

    async def on_media_pre_download(self, route, origin, media, reply):
        self.calls.append(("media", route.profile))
        return self.decision

    async def on_callback(self, route, origin, callback_data, reply):
        self.calls.append(("callback", route.profile))
        return self.decision


class TestRegistration:
    def test_requires_profile(self):
        hook = RecordingHook()
        hook.profile = ""
        with pytest.raises(ValueError):
            TopicHookRegistry().register(hook)

    def test_duplicate_profile_rejected(self):
        registry = TopicHookRegistry()
        registry.register(RecordingHook())
        with pytest.raises(ValueError):
            registry.register(RecordingHook())

    def test_duplicate_prefix_rejected(self):
        registry = TopicHookRegistry()
        registry.register(RecordingHook())
        other = RecordingHook()
        other.profile = "atlas"
        with pytest.raises(ValueError):
            registry.register(other)


class TestDispatch:
    @pytest.mark.asyncio
    async def test_message_dispatches_to_bound_profile_only(self):
        registry = TopicHookRegistry()
        hook = RecordingHook()
        registry.register(hook)
        decision = await registry.dispatch_message(SOL, ORIGIN, "hi", _noop_reply)
        assert decision is HookDecision.CONSUME
        # A different profile's route does not reach this hook.
        decision = await registry.dispatch_message(ATLAS, ORIGIN, "hi", _noop_reply)
        assert decision is HookDecision.CONTINUE
        assert hook.calls == [("message", "sol")]

    @pytest.mark.asyncio
    async def test_media_dispatch(self):
        registry = TopicHookRegistry()
        hook = RecordingHook(HookDecision.DENY)
        registry.register(hook)
        media = MediaDescriptor(
            kind="photo", file_size=1, width=1, height=1, media_group_id=None
        )
        decision = await registry.dispatch_media_pre_download(
            SOL, ORIGIN, media, _noop_reply
        )
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_callback_prefix_ownership(self):
        registry = TopicHookRegistry()
        hook = RecordingHook()
        registry.register(hook)
        assert registry.owns_callback("sf1:AAAAAAAAAAAAAAAAAAAAAA")
        assert not registry.owns_callback("mp:whatever")
        decision = await registry.dispatch_callback(
            SOL, ORIGIN, "sf1:AAAAAAAAAAAAAAAAAAAAAA", _noop_reply
        )
        assert decision is HookDecision.CONSUME

    @pytest.mark.asyncio
    async def test_foreign_route_callback_denied(self):
        # sf1 token clicked on a message routed to another profile: DENY,
        # never cross-dispatch.
        registry = TopicHookRegistry()
        registry.register(RecordingHook())
        decision = await registry.dispatch_callback(
            ATLAS, ORIGIN, "sf1:AAAAAAAAAAAAAAAAAAAAAA", _noop_reply
        )
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_unowned_callback_continues(self):
        registry = TopicHookRegistry()
        registry.register(RecordingHook())
        decision = await registry.dispatch_callback(
            SOL, ORIGIN, "ea:once:5", _noop_reply
        )
        assert decision is HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_raising_hook_fails_closed(self):
        class BoomHook(RecordingHook):
            async def on_message(self, route, origin, text, reply):
                raise RuntimeError("secret content must not leak")

        registry = TopicHookRegistry()
        registry.register(BoomHook())
        decision = await registry.dispatch_message(SOL, ORIGIN, "hi", _noop_reply)
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_non_decision_return_fails_closed(self):
        class WeirdHook(RecordingHook):
            async def on_message(self, route, origin, text, reply):
                return "continue"  # not a HookDecision

        registry = TopicHookRegistry()
        registry.register(WeirdHook())
        decision = await registry.dispatch_message(SOL, ORIGIN, "hi", _noop_reply)
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_no_hook_means_continue(self):
        registry = TopicHookRegistry()
        decision = await registry.dispatch_message(SOL, ORIGIN, "hi", _noop_reply)
        assert decision is HookDecision.CONTINUE
