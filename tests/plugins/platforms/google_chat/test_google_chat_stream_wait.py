"""Regression for #127018: a live Pub/Sub stream must not hold a default-executor worker.

The supervisor waits on its StreamingPullFuture for the whole connection. Waiting on a
default-executor thread pins one worker per served profile, so six google_chat profiles
exhaust a 2-vCPU host's pool and every later ``asyncio.to_thread`` call queues forever.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.google_chat import adapter as google_chat


class _Unauthenticated(Exception):
    pass


class _PermissionDenied(Exception):
    pass


class _Manager:
    """The StreamingPullManager calls StreamingPullFuture makes; close() shuts down on a thread, as the real one does."""

    def __init__(self) -> None:
        self.close_calls = 0
        self._callbacks: list = []

    def add_close_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def close(self, reason=None) -> None:
        self.close_calls += 1
        threading.Thread(target=self._fire, args=(reason,), daemon=True).start()

    def _fire(self, reason) -> None:
        for callback in self._callbacks:
            callback(self, reason)


class _StreamingPull(concurrent.futures.Future):
    """StreamingPullFuture's contract (google-cloud-pubsub >= 2.4.1), for hosts without the extra.

    cancel() only starts the manager's shutdown; the future resolves later, on the shutdown
    thread, with True or with the error that ended the stream.
    """

    def __init__(self, manager: _Manager) -> None:
        super().__init__()
        self._manager = manager
        self._cancel_requested = False
        manager.add_close_callback(self._on_close)

    def _on_close(self, _manager, reason) -> None:
        if self.done():
            return
        if reason is None:
            self.set_result(True)
        else:
            self.set_exception(reason)

    def cancel(self) -> bool:
        self._cancel_requested = True
        self._manager.close()
        return True

    def cancelled(self) -> bool:
        return self._cancel_requested


def _real_streaming_pull(manager: _Manager):
    futures = pytest.importorskip("google.cloud.pubsub_v1.subscriber.futures")
    return futures.StreamingPullFuture(manager)


@pytest.fixture(params=[_StreamingPull, _real_streaming_pull], ids=["contract", "google-cloud-pubsub"])
def make_stream(request):
    return request.param


@pytest.fixture
def adapter(monkeypatch):
    # The google stack is an optional extra; the supervisor only reads these two names.
    monkeypatch.setattr(google_chat, "pubsub_v1", SimpleNamespace(types=SimpleNamespace(FlowControl=lambda **kw: kw)))
    monkeypatch.setattr(
        google_chat, "gax_exceptions",
        SimpleNamespace(Unauthenticated=_Unauthenticated, PermissionDenied=_PermissionDenied),
    )
    a = google_chat.GoogleChatAdapter(PlatformConfig(enabled=True))
    a._subscription_path = "projects/p/subscriptions/s"
    return a


@pytest.mark.asyncio
async def test_live_stream_leaves_the_default_executor_free(adapter):
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    stream = _StreamingPull(_Manager())
    subscribed = asyncio.Event()

    def subscribe(*_args, **_kwargs):
        subscribed.set()
        return stream

    adapter._subscriber = Mock(subscribe=subscribe)
    supervisor = asyncio.create_task(adapter._run_supervisor())
    probe = None
    try:
        await asyncio.wait_for(subscribed.wait(), 5)
        probe = asyncio.ensure_future(asyncio.to_thread(threading.get_ident))
        done, _ = await asyncio.wait({probe}, timeout=5)
        assert probe in done, "a live Pub/Sub stream is holding the only default-executor worker"
    finally:
        adapter._shutting_down = True
        stream.set_result(True)
        await asyncio.wait_for(supervisor, 5)
        if probe is not None:
            await asyncio.wait_for(probe, 5)


@pytest.mark.asyncio
async def test_stream_outcomes_still_drive_resubscribe_retry_and_fatal(adapter, make_stream):
    """A clean end re-subscribes, a transient error backs off and retries, a permission error is fatal."""
    adapter._RECONNECT_BASE_DELAY = 0
    endings = [None, RuntimeError("stream reset by peer"), _PermissionDenied("missing roles/pubsub.subscriber")]
    managers: list[_Manager] = []

    def subscribe(*_args, **_kwargs):
        manager = _Manager()
        stream = make_stream(manager)
        manager.close(endings[len(managers)])  # the stream ends on the SDK's shutdown thread
        managers.append(manager)
        return stream

    adapter._subscriber = Mock(subscribe=subscribe)
    await asyncio.wait_for(adapter._run_supervisor(), 5)

    assert len(managers) == 3
    assert adapter.fatal_error_code == "pubsub_permission"


@pytest.mark.asyncio
async def test_disconnect_stops_a_live_stream(adapter, make_stream):
    manager = _Manager()
    stream = make_stream(manager)
    subscribed = asyncio.Event()

    def subscribe(*_args, **_kwargs):
        subscribed.set()
        return stream

    subscriber = Mock(subscribe=subscribe)
    adapter._subscriber = subscriber
    adapter._supervisor_task = supervisor = asyncio.create_task(adapter._run_supervisor())
    await asyncio.wait_for(subscribed.wait(), 5)

    await asyncio.wait_for(adapter.disconnect(), 5)

    assert supervisor.done()
    assert manager.close_calls >= 1
    assert stream.result(timeout=0) is True
    subscriber.close.assert_called_once()
