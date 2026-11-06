"""Tests for ``BasePlatformAdapter.register_post_delivery_callback`` chaining.

When two features want to run after the final response lands on the same
session (e.g. background-review release + temporary-progress cleanup), the
registration API chains them rather than clobbering. Per-callback
exceptions are swallowed so one bad callback can't sabotage the others.
Stale-generation registrations are rejected.

The chained wrapper is ``async`` so it transparently supports sync or async
callbacks — the outer invoker in ``_handle_message`` awaits awaitable
callbacks, and a sync wrapper would silently drop coroutine results from
async callbacks chained behind it.
"""
import asyncio
import inspect

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _MinAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture
def adapter():
    return _MinAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)


def _invoke(cb):
    """Invoke a popped callback, awaiting if it returns a coroutine.

    Single-registration callbacks are returned as the raw user callable
    (sync). Chained callbacks (two or more registrations on the same
    session) are wrapped in an async helper. Tests use this helper so
    they don't have to care which case they're exercising.
    """
    result = cb()
    if inspect.isawaitable(result):
        asyncio.run(result)


class TestPostDeliveryCallbackChaining:
    def test_single_callback_fires(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A"]

    def test_two_callbacks_chain_in_order(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        adapter.register_post_delivery_callback("s", lambda: fired.append("B"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A", "B"]

    def test_three_callbacks_chain_in_order(self, adapter):
        """Chain composes over an already-chained callback."""
        fired = []
        for label in ("A", "B", "C"):
            adapter.register_post_delivery_callback(
                "s", lambda x=label: fired.append(x)
            )
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A", "B", "C"]


class TestPostDeliveryCallbackAsyncChaining:
    """When an async callback is chained, the wrapper must await it.

    Regression test for a bug where the sync ``_chained`` wrapper called
    async callbacks without awaiting, silently dropping the returned
    coroutine. This broke ``/goal`` continuations (Discord etc.) where
    the continuation injection is an async ``_deliver()`` coroutine.
    """

    def test_async_callback_in_chain_is_awaited(self, adapter):
        fired = []

        async def async_cb():
            await asyncio.sleep(0)
            fired.append("async")

        adapter.register_post_delivery_callback("s", lambda: fired.append("sync"))
        adapter.register_post_delivery_callback("s", async_cb)
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["sync", "async"]



class TestCarryOverAcrossGenerations:
    """A turn's progress-bubble cleanup must survive a newer turn taking the session's slot
    before the older reply is delivered; ordinary stale callbacks stay dropped."""

    def test_older_cleanup_fires_after_newer_generation_overwrites(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("old-release"), generation=1)
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("old-cleanup"), generation=1, carry_over=True)
        adapter.register_post_delivery_callback("s", lambda: fired.append("new-release"), generation=2)
        assert adapter.pop_post_delivery_callback("s", generation=1) is None
        _invoke(adapter.pop_post_delivery_callback("s", generation=2))
        assert fired == ["old-cleanup", "new-release"]

    def test_stale_cleanup_registered_after_newer_generation_still_fires(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("new-release"), generation=2)
        adapter.register_post_delivery_callback("s", lambda: fired.append("stale"), generation=1)
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("old-cleanup"), generation=1, carry_over=True)
        _invoke(adapter.pop_post_delivery_callback("s", generation=2))
        assert fired == ["new-release", "old-cleanup"]

    def test_carried_cleanup_fires_once_across_two_supersessions(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("c1"), generation=1, carry_over=True)
        adapter.register_post_delivery_callback("s", lambda: fired.append("c2"), generation=2, carry_over=True)
        adapter.register_post_delivery_callback("s", lambda: fired.append("r3"), generation=3)
        _invoke(adapter.pop_post_delivery_callback("s", generation=3))
        assert fired == ["c1", "c2", "r3"]
        fired.clear()
        adapter.register_post_delivery_callback("s", lambda: fired.append("r4"), generation=4)
        _invoke(adapter.pop_post_delivery_callback("s", generation=4))
        assert fired == ["r4"]
