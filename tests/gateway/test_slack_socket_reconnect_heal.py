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


class TestSocketModeAppRebuild:
    """Regression for #85574: _restart_socket_mode must rebuild the AsyncApp.

    When the aiohttp ClientSession inside the AsyncApp's AsyncWebClient is
    closed, simply stopping/starting the Socket Mode handler reuses the same
    dead client. The restart path must drop the old app/clients and build fresh
    ones from the stored tokens.
    """

    @pytest.mark.asyncio
    async def test_restart_rebuilds_app_and_clients(self, adapter):
        """_restart_socket_mode creates a new AsyncApp and per-workspace clients."""
        old_app = adapter._app
        old_team_clients = {"T1": MagicMock(), "T2": MagicMock()}
        adapter._team_clients = dict(old_team_clients)
        adapter._bot_tokens = ["xoxb-primary", "xoxb-second"]
        adapter._team_tokens = {"T1": "xoxb-primary", "T2": "xoxb-second"}
        adapter._app_token = "xapp-fake"
        adapter._proxy_url = None

        created_apps: list = []
        created_clients: list = []

        def _fake_async_app(*, token: str, client):
            app = MagicMock()
            app.token = token
            app.client = client
            created_apps.append((token, client))
            return app

        def _fake_async_web_client(*, token: str, user_agent_prefix: str):
            client = MagicMock()
            client.token = token
            client.user_agent_prefix = user_agent_prefix
            created_clients.append((token, user_agent_prefix))
            return client

        # Avoid actually starting a socket task in the test.
        start_calls: list = []

        def _fake_start():
            start_calls.append(adapter._app)
            adapter._socket_mode_task = MagicMock()
            adapter._socket_mode_task.done.return_value = False

        adapter._stop_socket_mode_handler = AsyncMock()
        adapter._close_workspace_clients = AsyncMock()
        adapter._register_app_event_handlers = MagicMock()
        adapter._start_socket_mode_handler = _fake_start

        with patch.object(_slack_mod, "AsyncApp", side_effect=_fake_async_app):
            with patch.object(_slack_mod, "AsyncWebClient", side_effect=_fake_async_web_client):
                await adapter._restart_socket_mode("ping/pong stale")

        # The old app and team clients should have been discarded.
        assert adapter._app is not old_app, "AsyncApp was not rebuilt"
        assert adapter._team_clients is not old_team_clients, "team_clients dict not replaced"
        assert set(adapter._team_clients.keys()) == {"T1", "T2"}, "team workspaces lost"

        # One AsyncApp plus one AsyncWebClient per bot token were created
        # (the primary is the first bot token, which is also the T1 client).
        assert len(created_apps) == 1
        assert len(created_clients) == 3  # primary + T1 + T2
        # Primary token is the first bot token.
        app_token, app_client = created_apps[0]
        assert app_token == "xoxb-primary"
        assert app_client.user_agent_prefix == _slack_mod._HERMES_SLACK_USER_AGENT_PREFIX

        # Per-workspace clients use the stored team_id → token mapping.
        client_tokens = {c[0] for c in created_clients}
        assert client_tokens == {"xoxb-primary", "xoxb-second"}

        # Clean-up and handler wiring were performed in order.
        adapter._stop_socket_mode_handler.assert_awaited_once()
        adapter._close_workspace_clients.assert_awaited_once()
        adapter._register_app_event_handlers.assert_called_once()
        assert len(start_calls) == 1
        assert start_calls[0] is adapter._app

    @pytest.mark.asyncio
    async def test_restart_skips_without_stored_tokens(self, adapter):
        """Empty token cache leaves the live app/handler alone and warns once."""
        adapter._bot_tokens = []
        adapter._app_token = "xapp-fake"
        old_app = adapter._app

        adapter._stop_socket_mode_handler = AsyncMock()
        adapter._close_workspace_clients = AsyncMock()
        adapter._start_socket_mode_handler = MagicMock()

        with patch.object(_slack_mod.logger, "warning") as warn:
            await adapter._restart_socket_mode("ping/pong stale")
            await adapter._restart_socket_mode("ping/pong stale")

        adapter._stop_socket_mode_handler.assert_not_awaited()
        adapter._close_workspace_clients.assert_not_awaited()
        adapter._start_socket_mode_handler.assert_not_called()
        assert adapter._app is old_app
        assert adapter._socket_rebuild_skipped_no_tokens is True
        assert warn.call_count == 1

    @pytest.mark.asyncio
    async def test_restart_keeps_app_when_construct_fails(self, adapter):
        """A failed AsyncApp construct must not null _app or stop the handler."""
        old_app = adapter._app
        adapter._bot_tokens = ["xoxb-primary"]
        adapter._team_tokens = {"T1": "xoxb-primary"}
        adapter._app_token = "xapp-fake"

        adapter._stop_socket_mode_handler = AsyncMock()
        adapter._close_workspace_clients = AsyncMock()
        adapter._start_socket_mode_handler = MagicMock()

        with patch.object(_slack_mod, "AsyncApp", side_effect=RuntimeError("boom")):
            with patch.object(_slack_mod, "AsyncWebClient", return_value=MagicMock()):
                await adapter._restart_socket_mode("ping/pong stale")

        assert adapter._app is old_app
        adapter._stop_socket_mode_handler.assert_not_awaited()
        adapter._close_workspace_clients.assert_not_awaited()
        adapter._start_socket_mode_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_clears_stored_tokens(self, adapter):
        """disconnect() drops the token cache so a later restart cannot build a stale app."""
        adapter._bot_tokens = ["xoxb-fake"]
        adapter._team_tokens = {"T1": "xoxb-fake"}

        # Avoid stopping/cleanup side effects and platform lock release.
        adapter._stop_socket_mode_handler = AsyncMock()
        adapter._close_workspace_clients = AsyncMock()
        adapter._seal_stream = AsyncMock()
        adapter._stop_native_task_card_stream = AsyncMock()
        adapter._release_platform_lock = MagicMock()

        # _release_platform_lock normally checks _running; make it a no-op.
        await adapter.disconnect()

        assert adapter._bot_tokens == []
        assert adapter._team_tokens == {}
