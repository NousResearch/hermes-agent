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

    def test_exception_in_one_callback_does_not_block_next(self, adapter):
        fired = []

        def boom():
            raise ValueError("boom")

        adapter.register_post_delivery_callback("s", boom)
        adapter.register_post_delivery_callback("s", lambda: fired.append("survived"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["survived"]

    def test_same_generation_chains(self, adapter):
        fired = []
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("A"), generation=5
        )
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("B"), generation=5
        )
        cb = adapter.pop_post_delivery_callback("s", generation=5)
        _invoke(cb)
        assert fired == ["A", "B"]

    def test_stale_generation_registration_rejected(self, adapter):
        """A registration with an older generation than the existing
        entry is rejected — it doesn't clobber the newer run's slot."""
        fired = []
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("gen7"), generation=7
        )
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("stale_gen3"), generation=3
        )
        cb = adapter.pop_post_delivery_callback("s", generation=7)
        _invoke(cb)
        assert fired == ["gen7"]

    def test_pop_at_wrong_generation_returns_none(self, adapter):
        adapter.register_post_delivery_callback(
            "s", lambda: None, generation=5
        )
        assert adapter.pop_post_delivery_callback("s", generation=99) is None
        # Correct generation still finds it.
        assert adapter.pop_post_delivery_callback("s", generation=5) is not None

    def test_empty_session_key_is_noop(self, adapter):
        adapter.register_post_delivery_callback("", lambda: None)
        assert adapter._post_delivery_callbacks == {}

    def test_non_callable_is_noop(self, adapter):
        adapter.register_post_delivery_callback("s", "not-callable")  # type: ignore[arg-type]
        assert adapter._post_delivery_callbacks == {}


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

    def test_two_async_callbacks_both_awaited(self, adapter):
        fired = []

        def make(label):
            async def _cb():
                await asyncio.sleep(0)
                fired.append(label)

            return _cb

        adapter.register_post_delivery_callback("s", make("A"))
        adapter.register_post_delivery_callback("s", make("B"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A", "B"]
