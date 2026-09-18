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


class _RebindingClient(_FakeSocketModeClient):
    """A client that rebinds a live task *during* close(), as the SDK does.

    ``connect()`` rebinds ``current_session_monitor`` / ``message_receiver`` on
    success, and those rebinds happen across the awaits inside ``close()`` --
    the moving target the adapter's teardown comment warns about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rebound_task = None

    async def close(self) -> None:
        self.closed = True
        self.rebound_task = asyncio.create_task(self.connect_to_new_endpoint())
        self.message_receiver = self.rebound_task
        await self.aiohttp_client_session.close()


class _RebindingHandler(_FakeHandler):
    """Stands in for an AsyncSocketModeHandler whose client rebinds on close."""

    def __init__(self) -> None:
        super().__init__()
        self.client = _RebindingClient()


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

    @pytest.mark.asyncio
    async def test_watchdog_reaps_retired_generations(self, adapter):
        """Every watchdog tick gives retired generations a chance to be reaped.

        Nothing else in the watchdog can see them: the transport and ping/pong
        probes only ever look at the current handler.
        """
        live_task = MagicMock()
        live_task.done.return_value = False
        adapter._socket_mode_task = live_task
        adapter._handler = MagicMock()

        reaped: list[int] = []

        async def _fake_reap() -> None:
            reaped.append(1)

        async def _transport() -> bool:
            # End the loop after this iteration -- the reaper must not be what
            # stops the watchdog, or a pre-fix run would hang instead of fail.
            adapter._running = False
            return True

        adapter._reap_retired_socket_generations = _fake_reap
        adapter._socket_transport_connected = _transport
        adapter._socket_ping_pong_stale = MagicMock(return_value=False)
        adapter._socket_watchdog_interval_s = 0.01

        await adapter._socket_watchdog_loop()

        assert reaped == [1]


class TestRetiredSocketGenerationReaping:
    """A task that outlives teardown must be reaped, and reported as a leak.

    ``SocketModeClient.connect()`` rebinds the client's task attributes across
    the awaits inside ``close()``, so cancelling a single snapshot taken before
    the close leaves survivors retrying ``connect()`` against a session that is
    already gone (#83662). Those orphans belong to a *retired* generation, which
    is why the watchdog -- looking only at the current handler -- never sees
    them (#85574).
    """

    @pytest.mark.asyncio
    async def test_rebound_task_after_teardown_snapshot_is_drained(self, adapter):
        """A task bound during close_async() must not outlive teardown.

        The SDK's rebind happens *inside* the close, so a few retries against
        the dying session are unavoidable; what matters is that nothing is still
        retrying once teardown has returned. Without the drain the task survives
        and keeps logging "Session is closed" forever (#83662).
        """
        handler = _RebindingHandler()
        _attach(adapter, handler)
        await asyncio.sleep(0.01)

        await adapter._stop_socket_mode_handler()
        session = handler.client.aiohttp_client_session
        retries_at_teardown = session.ws_connect_after_close

        # Give a survivor every chance to make itself known.
        await asyncio.sleep(0.05)

        rebound = handler.client.rebound_task
        try:
            assert rebound is not None, "the fake never rebound a task"
            assert rebound.done(), (
                "a task rebound during close_async() outlived teardown"
            )
            assert session.ws_connect_after_close == retries_at_teardown, (
                "a task rebound during close_async() kept retrying after "
                f"teardown: {session.ws_connect_after_close - retries_at_teardown} "
                "more attempt(s)"
            )
        finally:
            if rebound is not None and not rebound.done():
                rebound.cancel()

    @pytest.mark.asyncio
    async def test_reap_cancels_orphaned_task_and_warns_once(self, adapter, caplog):
        """An orphan no attribute points at is still found, cancelled, and reported."""
        handler = _FakeHandler()
        _attach(adapter, handler)
        await asyncio.sleep(0.01)
        await adapter._stop_socket_mode_handler()

        client = handler.client
        session = client.aiohttp_client_session
        # A survivor can end up reachable from no attribute at all once the SDK
        # has rebound them. Model that: a bare task inside connect(), with the
        # client referenced only by the task's live frame.
        orphan = asyncio.create_task(client.connect())
        await asyncio.sleep(0.01)
        client.message_receiver = None
        client.current_session_monitor = None

        retries_before = session.ws_connect_after_close

        try:
            with caplog.at_level("WARNING"):
                await adapter._reap_retired_socket_generations()
            await asyncio.sleep(0.02)

            assert orphan.cancelled() or orphan.done(), "the orphan was not reaped"
            assert session.ws_connect_after_close == retries_before, (
                "the orphan kept retrying the closed session after the reap"
            )
        finally:
            if not orphan.done():
                orphan.cancel()
                await asyncio.sleep(0)
        warnings = [r for r in caplog.records if "Reaped" in r.getMessage()]
        assert len(warnings) == 1, f"expected one leak warning, got {len(warnings)}"

        # A second tick must not warn again (nor raise).
        with caplog.at_level("WARNING"):
            await adapter._reap_retired_socket_generations()
        warnings = [r for r in caplog.records if "Reaped" in r.getMessage()]
        assert len(warnings) == 1, "the leak was reported more than once"

    @pytest.mark.asyncio
    async def test_reap_never_touches_the_current_handler(self, adapter):
        """Only retired generations are reaped; live work keeps running."""
        live = asyncio.create_task(_spin())
        adapter._handler = _FakeHandler()
        adapter._socket_mode_task = live
        adapter._retired_socket_generations.append(
            _slack_mod._RetiredSocketGeneration(_FakeHandler(), None, None)
        )

        await adapter._reap_retired_socket_generations()

        assert not live.cancelled() and not live.done(), (
            "the reap cancelled the current handler's task"
        )
        live.cancel()

    @pytest.mark.asyncio
    async def test_retired_generations_are_pruned(self, adapter):
        """Repeated start/stop cycles leave a bounded, self-cleaning registry."""
        for _ in range(_slack_mod._MAX_RETIRED_SOCKET_GENERATIONS + 4):
            _attach(adapter, _FakeHandler())
            await asyncio.sleep(0.01)
            await adapter._stop_socket_mode_handler()

        assert (
            len(adapter._retired_socket_generations)
            <= _slack_mod._MAX_RETIRED_SOCKET_GENERATIONS
        )

        # Once their tasks are gone, generations are dropped, not accumulated.
        await adapter._reap_retired_socket_generations()
        assert adapter._retired_socket_generations == []
