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
        self.message_processor = None
        self.current_session_monitor = None
        self.message_receiver = None

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
    async def test_sdk_background_tasks_do_not_outlive_teardown(self, adapter):
        """Old-client background tasks must be cancelled even if close_async() fails.

        SocketModeClient.close() cancels message_processor,
        current_session_monitor and message_receiver only after disconnect()
        returns, so a raising disconnect() leaves all three running. The adapter
        logs and moves on, so it has to clean them up itself.
        """
        handler = _FakeHandler()
        client = handler.client
        client.close_should_raise = True
        client.message_processor = asyncio.create_task(_spin())
        client.current_session_monitor = asyncio.create_task(_spin())
        client.message_receiver = asyncio.create_task(_spin())

        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        await adapter._stop_socket_mode_handler()
        await asyncio.sleep(0.03)

        for name in (
            "message_processor",
            "current_session_monitor",
            "message_receiver",
        ):
            assert getattr(client, name).done(), f"{name} outlived teardown"

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

    @pytest.mark.asyncio
    async def test_stop_clears_adapter_state(self, adapter):
        """Teardown always drops its references, even when close_async() raises."""
        handler = _FakeHandler()
        handler.client.close_should_raise = True
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        await adapter._stop_socket_mode_handler()

        assert adapter._handler is None
        assert adapter._socket_mode_task is None


class TestSocketModeRestart:
    @pytest.mark.asyncio
    async def test_restart_stops_old_handler_before_starting_new_one(self, adapter):
        """A reconnect must fully retire the old handler before replacing it."""
        old = _FakeHandler()
        old_task = _attach(adapter, old)
        await asyncio.sleep(0.01)

        started: list[str] = []

        def _fake_start() -> None:
            started.append("started")
            assert old_task.done(), (
                "the replacement handler was created while the old socket task "
                "was still running"
            )

        with patch.object(adapter, "_start_socket_mode_handler", _fake_start):
            await adapter._restart_socket_mode("transport disconnected")

        await asyncio.sleep(0.03)

        assert started == ["started"]
        assert old.client.aiohttp_client_session.ws_connect_after_close == 0

    @pytest.mark.asyncio
    async def test_watchdog_restarts_when_socket_task_stops(self, adapter):
        """The existing watchdog triggers still fire after the teardown change."""
        done_task = MagicMock()
        done_task.done.return_value = True
        adapter._socket_mode_task = done_task

        reasons: list[str] = []

        async def _fake_restart(reason: str) -> None:
            reasons.append(reason)
            adapter._running = False

        adapter._restart_socket_mode = _fake_restart
        adapter._socket_transport_connected = AsyncMock(return_value=None)
        adapter._socket_watchdog_interval_s = 0.01

        await adapter._socket_watchdog_loop()

        assert reasons == ["socket task stopped"]

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
