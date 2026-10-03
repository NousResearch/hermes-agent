"""Google Chat streams must not consume shared executor capacity (#127018)."""

import asyncio
import contextvars
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.google_chat import adapter as google_chat


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    monkeypatch.setattr(google_chat, "pubsub_v1", SimpleNamespace(
        types=SimpleNamespace(FlowControl=lambda **kwargs: kwargs)))
    monkeypatch.setattr(google_chat, "gax_exceptions", SimpleNamespace(
        Unauthenticated=type("Unauthenticated", (Exception,), {}),
        PermissionDenied=type("PermissionDenied", (Exception,), {})))
    return google_chat.GoogleChatAdapter(PlatformConfig(enabled=True))


def test_live_stream_leaves_default_executor_available(adapter):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        entered = asyncio.Event()
        scope = contextvars.ContextVar("scope", default="unbound")
        observed = []

        class Stream(Future):
            def result(self, timeout=None):
                observed.append(scope.get())
                loop.call_soon_threadsafe(entered.set)
                return super().result(timeout)

        stream = Stream()
        adapter._subscriber = Mock(subscribe=Mock(return_value=stream))
        scope.set("owning-profile")
        task = asyncio.create_task(adapter._run_supervisor())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            probe = asyncio.create_task(asyncio.to_thread(lambda: "available"))
            done, _ = await asyncio.wait({probe}, timeout=2)
            assert probe in done, "live Pub/Sub stream starved the default executor"
            assert probe.result() == "available"
            assert observed == ["owning-profile"]
        finally:
            adapter._shutting_down = True
            stream.set_result(None)
            await asyncio.wait_for(task, 3)

    asyncio.run(scenario())


def test_supervisor_preserves_retry_fatal_and_disconnect(adapter):
    async def scenario():
        loop = asyncio.get_running_loop()
        subscribed = asyncio.Event()
        transient = Future()
        transient.set_exception(RuntimeError("transport failed"))
        class StreamingPull(Future):
            # The SDK cancel requests shutdown; completion signals that shutdown
            # has finished (it does not mark the concurrent Future cancelled).
            def cancel(self):
                self.set_result(None)
                return True

        live = StreamingPull()
        streams = iter([transient, live])

        def subscribe(*args, **kwargs):
            stream = next(streams)
            if stream is live:
                subscribed.set()
            return stream

        adapter._RECONNECT_BASE_DELAY = 0
        subscriber = Mock(subscribe=Mock(side_effect=subscribe))
        adapter._subscriber = subscriber
        adapter._supervisor_task = asyncio.create_task(adapter._run_supervisor())
        try:
            await asyncio.wait_for(subscribed.wait(), 3)
            await asyncio.wait_for(adapter.disconnect(), 3)
            assert live.done()
            subscriber.close.assert_called_once()
            assert adapter._supervisor_task.done()
            assert await asyncio.to_thread(lambda: True)
        finally:
            if not live.done():
                live.set_result(None)
            await asyncio.wait_for(adapter._supervisor_task, 3)

        # A fresh connection still classifies permanent auth failure, not retry.
        adapter._shutting_down = False
        fatal = Future()
        fatal.set_exception(google_chat.gax_exceptions.PermissionDenied("denied"))
        adapter._subscriber = Mock(subscribe=Mock(return_value=fatal))
        await asyncio.wait_for(adapter._run_supervisor(), 3)
        assert adapter.fatal_error_code == "pubsub_permission"
        assert adapter._subscriber.subscribe.call_count == 1

    asyncio.run(scenario())
