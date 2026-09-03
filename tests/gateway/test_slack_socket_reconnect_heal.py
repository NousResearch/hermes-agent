"""
Tests for Slack Socket Mode teardown (issue #46990).

slack_sdk's SocketModeClient.connect() is an unconditional retry loop that
swallows connection errors and never checks the client's ``closed`` flag. If a
task is still inside that loop when the client's shared aiohttp session is
closed, it keeps retrying forever and logs
``Failed to connect (error: Session is closed); Retrying...`` against a session
that can never work again.

These tests pin the ordering and cleanup that keep old-client background work
from outliving a teardown.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock the slack-bolt package if it's not installed
# ---------------------------------------------------------------------------


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal stand-ins for the slack_sdk objects involved in teardown
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for the ``aiohttp.ClientSession`` SocketModeClient holds."""

    def __init__(self, client=None) -> None:
        self.closed = False
        self.reachable = False
        self.ws_connect_after_close = 0
        self._client = client
        self.live_tasks_at_close: list = []

    async def ws_connect(self):
        if self.closed:
            # This is the exact failure recorded in #46990.
            self.ws_connect_after_close += 1
            raise RuntimeError("Session is closed")
        if not self.reachable:
            raise ConnectionError("connection refused")
        return object()

    async def close(self) -> None:
        # Record which client tasks were still alive at the instant the shared
        # session went away. Anything listed here could be inside connect().
        if self._client is not None:
            self.live_tasks_at_close = self._client.live_task_names()
        self.closed = True
        # Closing a real session performs I/O and yields control back to the
        # loop, which is what gives a surviving retry task a chance to run.
        await asyncio.sleep(0.01)


class _FakeSocketModeClient:
    """Mirrors the parts of SocketModeClient that matter during teardown."""

    _TASK_ATTRS = ("message_processor", "current_session_monitor", "message_receiver")

    def __init__(self) -> None:
        self.aiohttp_client_session = _FakeSession(self)
        self.closed = False
        self.close_should_raise = False
        self.message_processor: asyncio.Task | None = None
        self.current_session_monitor: asyncio.Task | None = None
        self.message_receiver: asyncio.Task | None = None

    def live_task_names(self) -> list:
        return [
            attr
            for attr in self._TASK_ATTRS
            if getattr(self, attr) is not None and not getattr(self, attr).done()
        ]

    async def connect_to_new_endpoint(self) -> None:
        # monitor_current_session() (on staleness) and receive_messages() (on a
        # CLOSE frame) both reach connect() through here, independently.
        await self.connect()

    async def monitor_current_session(self) -> None:
        while not self.closed:
            await asyncio.sleep(0.001)
            await self.connect_to_new_endpoint()

    async def connect(self) -> None:
        # Mirrors SocketModeClient.connect(): ``while True`` with a broad
        # ``except Exception``, so neither the closed flag nor a closed session
        # ends the loop.
        while True:
            try:
                await self.aiohttp_client_session.ws_connect()
                return
            except Exception:
                await asyncio.sleep(0.001)

    async def close(self) -> None:
        self.closed = True
        if self.close_should_raise:
            # SocketModeClient.close() calls disconnect() before it cancels its
            # background tasks. A broken session makes disconnect() raise, so
            # the SDK never reaches those cancel() calls at all.
            raise RuntimeError("Session is closed")
        for task in (
            self.message_processor,
            self.current_session_monitor,
            self.message_receiver,
        ):
            if task is not None:
                # The SDK requests cancellation but never awaits it.
                task.cancel()
        await self.aiohttp_client_session.close()


class _FakeHandler:
    """Stands in for AsyncSocketModeHandler."""

    def __init__(self) -> None:
        self.client = _FakeSocketModeClient()

    async def start_async(self) -> None:
        await self.client.connect()
        await asyncio.sleep(float("inf"))

    async def close_async(self) -> None:
        await self.client.close()


async def _spin() -> None:
    while True:
        await asyncio.sleep(0.001)


async def _resist_cancellation_until(release: asyncio.Event) -> None:
    """Keep running across cancellation until the test explicitly releases it."""
    while not release.is_set():
        try:
            await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            continue


async def _replace_client_task_on_cancel(
    client: _FakeSocketModeClient,
    release: asyncio.Event,
) -> None:
    """Model an SDK owner that replaces its task while cancellation runs."""
    try:
        await _spin()
    except asyncio.CancelledError:
        client.message_processor = asyncio.create_task(
            _resist_cancellation_until(release)
        )


async def _replace_client_task_with_cooperative_owner_on_cancel(
    client: _FakeSocketModeClient,
) -> None:
    """Publish a normal cancellable owner while the first snapshot settles."""
    try:
        await _spin()
    except asyncio.CancelledError:
        client.message_processor = asyncio.create_task(_spin())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app_token = "xapp-fake"
    a._proxy_url = None
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _attach(adapter, handler):
    """Wire a handler into the adapter the way _start_socket_mode_handler does."""
    adapter._handler = handler
    task = asyncio.create_task(handler.start_async())
    adapter._socket_mode_task = task
    return task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSocketModeTeardown:
    @pytest.mark.asyncio
    async def test_socket_task_stops_before_session_is_closed(self, adapter):
        """The socket task must be stopped before close_async() kills the session.

        The task is parked in the SDK's connect() retry loop, which is the state
        #46990 describes. If teardown closes the shared session first, that loop
        wakes up and retries against a session that is already gone.
        """
        handler = _FakeHandler()
        task = _attach(adapter, handler)
        # Let the task settle into the retry loop.
        await asyncio.sleep(0.01)

        await adapter._stop_socket_mode_handler()
        # Give anything that survived a chance to make itself known.
        await asyncio.sleep(0.03)

        session = handler.client.aiohttp_client_session
        assert session.ws_connect_after_close == 0, (
            "the old socket task retried against a closed session "
            f"{session.ws_connect_after_close} time(s) after close_async()"
        )
        assert task.done(), "the old socket task outlived teardown"

    @pytest.mark.asyncio
    async def test_client_tasks_are_dead_before_the_session_closes(self, adapter):
        """Nothing may still be inside connect() when the shared session closes.

        monitor_current_session() and receive_messages() each reach
        connect_to_new_endpoint() on their own, and connect() rebinds
        current_session_monitor and message_receiver to fresh tasks on success.
        The live task set therefore changes across the awaits inside
        SocketModeClient.close(), so cancelling from a snapshot taken partway
        through races a moving target. Everything has to be stopped before the
        session is closed. See slackapi/python-slack-sdk#1913.
        """
        handler = _FakeHandler()
        client = handler.client
        client.message_processor = asyncio.create_task(_spin())
        client.current_session_monitor = asyncio.create_task(
            client.monitor_current_session()
        )
        client.message_receiver = asyncio.create_task(client.monitor_current_session())

        _attach(adapter, handler)
        # Let both reconnect loops settle inside connect().
        await asyncio.sleep(0.01)

        await adapter._stop_socket_mode_handler()
        await asyncio.sleep(0.03)

        session = client.aiohttp_client_session
        assert session.live_tasks_at_close == [], (
            "client tasks were still running when the shared session was closed: "
            f"{session.live_tasks_at_close}"
        )
        assert session.ws_connect_after_close == 0, (
            "a client task retried against a closed session "
            f"{session.ws_connect_after_close} time(s)"
        )


class TestSocketModeRestart:
    @pytest.mark.asyncio
    async def test_restart_defers_close_for_task_created_during_cancellation(
        self, adapter
    ):
        """A post-snapshot SDK task must keep the old session open."""
        handler = _FakeHandler()
        client = handler.client
        release = asyncio.Event()
        client.message_processor = asyncio.create_task(
            _replace_client_task_on_cancel(client, release)
        )
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        started: list[str] = []

        try:
            with (
                patch.object(_slack_mod, "_SOCKET_TASK_CANCEL_TIMEOUT_S", 0.01),
                patch.object(
                    adapter,
                    "_start_socket_mode_handler",
                    side_effect=lambda: started.append("started"),
                ),
            ):
                await adapter._restart_socket_mode("transport disconnected")

                replacement = client.message_processor
                assert replacement is not None
                assert replacement.done() is False
                assert client.aiohttp_client_session.closed is False
                assert adapter._handler is handler
                assert started == []

                release.set()
                await asyncio.wait_for(replacement, timeout=0.1)
                await adapter._restart_socket_mode("deferred teardown retry")

            assert client.aiohttp_client_session.closed is True
            assert started == ["started"]
        finally:
            release.set()
            replacement = client.message_processor
            if replacement is not None and not replacement.done():
                await asyncio.wait_for(replacement, timeout=0.1)

    @pytest.mark.parametrize(
        "owner_attr",
        ["outer", *_FakeSocketModeClient._TASK_ATTRS],
    )
    @pytest.mark.asyncio
    async def test_restart_defers_close_while_owner_resists_cancellation(
        self, adapter, owner_attr
    ):
        """Reconnect must not close a session that an owner can still use."""
        handler = _FakeHandler()
        client = handler.client
        release = asyncio.Event()
        stubborn_task = asyncio.create_task(_resist_cancellation_until(release))
        if owner_attr == "outer":
            adapter._handler = handler
            adapter._socket_mode_task = stubborn_task
            old_task = stubborn_task
        else:
            setattr(client, owner_attr, stubborn_task)
            old_task = _attach(adapter, handler)
        old_task.add_done_callback(adapter._on_socket_mode_task_done)
        await asyncio.sleep(0.01)

        started: list[str] = []

        try:
            with (
                patch.object(_slack_mod, "_SOCKET_TASK_CANCEL_TIMEOUT_S", 0.01),
                patch.object(
                    adapter,
                    "_start_socket_mode_handler",
                    side_effect=lambda: started.append("started"),
                ),
            ):
                await adapter._restart_socket_mode("transport disconnected")

                session = client.aiohttp_client_session
                assert session.closed is False
                assert session.ws_connect_after_close == 0
                assert adapter._handler is handler
                assert adapter._socket_mode_task is old_task
                assert started == []

                release.set()
                await asyncio.wait_for(stubborn_task, timeout=0.1)
                await adapter._restart_socket_mode("deferred teardown retry")
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            assert session.closed is True
            assert started == ["started"]
        finally:
            release.set()
            if not stubborn_task.done():
                await asyncio.wait_for(stubborn_task, timeout=0.1)

    @pytest.mark.asyncio
    async def test_disconnect_closes_handler_when_watchdog_teardown_is_cancelled(
        self, adapter
    ):
        """Cancelling the watchdog mid-stop must restore ownership for shutdown."""
        handler = _FakeHandler()
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        entered_cancel = asyncio.Event()
        original_cancel = _slack_mod._cancel_socket_tasks
        cancel_calls = 0

        async def _block_first_cancel(tasks):
            nonlocal cancel_calls
            cancel_calls += 1
            if cancel_calls == 1:
                entered_cancel.set()
                await asyncio.Event().wait()
            return await original_cancel(tasks)

        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        with (
            patch.object(_slack_mod, "_cancel_socket_tasks", _block_first_cancel),
            patch.object(adapter, "_start_socket_mode_handler") as start_handler,
        ):
            adapter._socket_watchdog_task = asyncio.create_task(
                adapter._restart_socket_mode("transport disconnected")
            )
            await asyncio.wait_for(entered_cancel.wait(), timeout=0.1)
            assert adapter._handler is None

            await adapter.disconnect()

        assert handler.client.aiohttp_client_session.closed is True
        assert adapter._handler is None
        assert adapter._socket_mode_task is None
        assert adapter._socket_watchdog_task is None
        assert adapter._app is None
        start_handler.assert_not_called()
        adapter._close_workspace_clients.assert_awaited_once()
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancellation_during_stream_seal_waits_for_cleanup(
        self, adapter
    ):
        """Early stream teardown cancellation must not bypass final cleanup."""
        handler = _FakeHandler()
        client = handler.client
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        seal_entered = asyncio.Event()
        seal_release = asyncio.Event()

        async def _blocked_seal_stream(chat_id, stream) -> None:
            seal_entered.set()
            await seal_release.wait()

        adapter._active_streams = {"C123": object()}
        adapter._seal_stream = _blocked_seal_stream
        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        disconnect_task = asyncio.create_task(adapter.disconnect())
        await asyncio.wait_for(seal_entered.wait(), timeout=0.1)
        disconnect_task.cancel()
        await asyncio.sleep(0)
        assert disconnect_task.done() is False

        seal_release.set()
        with pytest.raises(asyncio.CancelledError):
            await disconnect_task

        assert client.aiohttp_client_session.closed is True
        assert adapter._active_streams == {}
        assert adapter._handler is None
        assert adapter._socket_mode_task is None
        assert adapter._app is None
        assert adapter._socket_reconnect_lock.locked() is False
        adapter._close_workspace_clients.assert_awaited_once()
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancellation_waits_for_final_cleanup(self, adapter):
        """Caller cancellation must not bypass final close or lock release."""
        handler = _FakeHandler()
        client = handler.client
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        close_entered = asyncio.Event()
        close_release = asyncio.Event()

        async def _blocked_close_async() -> None:
            close_entered.set()
            await close_release.wait()
            await client.close()

        handler.close_async = _blocked_close_async
        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        disconnect_task = asyncio.create_task(adapter.disconnect())
        await asyncio.wait_for(close_entered.wait(), timeout=0.1)
        disconnect_task.cancel()
        await asyncio.sleep(0)
        assert disconnect_task.done() is False

        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await disconnect_task

        assert client.aiohttp_client_session.closed is True
        assert adapter._handler is None
        assert adapter._socket_mode_task is None
        assert adapter._app is None
        assert adapter._socket_reconnect_lock.locked() is False
        adapter._close_workspace_clients.assert_awaited_once()
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_awaits_task_created_during_cancellation(self, adapter):
        """Full shutdown must await a cooperative post-snapshot SDK owner."""
        handler = _FakeHandler()
        client = handler.client
        client.message_processor = asyncio.create_task(
            _replace_client_task_with_cooperative_owner_on_cancel(client)
        )
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        await adapter.disconnect()

        replacement = client.message_processor
        assert replacement is not None
        assert replacement.done() is True
        assert client.aiohttp_client_session.live_tasks_at_close == []
        assert client.aiohttp_client_session.closed is True
        assert adapter._handler is None
        adapter._close_workspace_clients.assert_awaited_once()
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_disconnect_cannot_restore_deferred_handler(self, adapter):
        """Shutdown must win a race with reconnect teardown and close its handler."""
        handler = _FakeHandler()
        release_owner = asyncio.Event()
        stubborn_task = asyncio.create_task(_resist_cancellation_until(release_owner))
        handler.client.message_processor = stubborn_task
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        entered_cancel = asyncio.Event()
        continue_cancel = asyncio.Event()
        original_cancel = _slack_mod._cancel_socket_tasks
        cancel_calls = 0

        async def _pause_first_cancel(tasks):
            nonlocal cancel_calls
            cancel_calls += 1
            if cancel_calls == 1:
                entered_cancel.set()
                await continue_cancel.wait()
            return await original_cancel(tasks)

        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        try:
            with (
                patch.object(_slack_mod, "_SOCKET_TASK_CANCEL_TIMEOUT_S", 0.01),
                patch.object(_slack_mod, "_cancel_socket_tasks", _pause_first_cancel),
                patch.object(adapter, "_start_socket_mode_handler") as start_handler,
            ):
                reconnect = asyncio.create_task(
                    adapter._restart_socket_mode("transport disconnected")
                )
                await asyncio.wait_for(entered_cancel.wait(), timeout=0.1)
                assert adapter._handler is None

                shutdown = asyncio.create_task(adapter.disconnect())
                await asyncio.sleep(0.01)
                continue_cancel.set()
                await asyncio.gather(reconnect, shutdown)

            assert handler.client.aiohttp_client_session.closed is True
            assert adapter._handler is None
            assert adapter._socket_mode_task is None
            assert adapter._app is None
            start_handler.assert_not_called()
            adapter._close_workspace_clients.assert_awaited_once()
            adapter._release_platform_lock.assert_called_once()
        finally:
            continue_cancel.set()
            release_owner.set()
            if not stubborn_task.done():
                await asyncio.wait_for(stubborn_task, timeout=0.1)

    @pytest.mark.asyncio
    async def test_full_disconnect_cleans_up_after_cancellation_timeout(self, adapter):
        """Fail-closed reconnect state must not strand full gateway shutdown."""
        handler = _FakeHandler()
        release = asyncio.Event()
        stubborn_task = asyncio.create_task(_resist_cancellation_until(release))
        handler.client.message_processor = stubborn_task
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        adapter._close_workspace_clients = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        try:
            with (
                patch.object(_slack_mod, "_SOCKET_TASK_CANCEL_TIMEOUT_S", 0.01),
                patch.object(adapter, "_start_socket_mode_handler") as start_handler,
            ):
                await adapter._restart_socket_mode("transport disconnected")
                assert handler.client.aiohttp_client_session.closed is False
                assert adapter._handler is handler
                start_handler.assert_not_called()

                await adapter.disconnect()

            assert handler.client.aiohttp_client_session.closed is True
            assert adapter._handler is None
            assert adapter._socket_mode_task is None
            assert adapter._app is None
            adapter._close_workspace_clients.assert_awaited_once()
            adapter._release_platform_lock.assert_called_once()
        finally:
            release.set()
            if not stubborn_task.done():
                await asyncio.wait_for(stubborn_task, timeout=0.1)

    @pytest.mark.asyncio
    async def test_watchdog_restarts_when_transport_disconnected(self, adapter):
        """A transport that reports itself down still triggers a reconnect."""
        live_task = MagicMock()
        live_task.done.return_value = False
        adapter._socket_mode_task = live_task
        adapter._handler = MagicMock()

        reasons: list[str] = []

        async def _fake_restart(reason: str) -> None:
            reasons.append(reason)
            adapter._running = False

        adapter._restart_socket_mode = _fake_restart
        adapter._socket_transport_connected = AsyncMock(return_value=False)
        adapter._socket_watchdog_interval_s = 0.01

        await adapter._socket_watchdog_loop()

        assert reasons == ["transport disconnected"]
