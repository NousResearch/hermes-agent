"""
Socket Mode app-level liveness: a session can hold a healthy websocket (ping/pong
answered by Slack's edge) while Slack never routes an event to it. Seen after a
Slack real-time connection disruption: sockets that reconnected during the
incident answered pings for hours, delivered nothing, and got none of Slack's
periodic refreshes until the process restarted.

Slack sends ``hello`` once it registers a session for event routing, and every
event is a text frame. These tests pin how the watchdog uses those two signals:
an unregistered session is restarted, a silent one is renewed make-before-break,
and a client whose frames cannot be observed keeps the old behaviour. Time is a
fake clock and the watchdog is driven one tick at a time.
"""

import asyncio
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.waiting import async_wait_until


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

import plugins.platforms.slack.adapter as _slack_mod
import plugins.platforms.slack.adapter_socket as _socket_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig


def _hello(connections=1):
    return json.dumps({
        "type": "hello",
        "num_connections": connections,
        "debug_info": {"host": "applink-test", "approximate_connection_time": 18060},
        "connection_info": {"app_id": "A_TEST"},
    })


_EVENT = '{"envelope_id":"e1","type":"events_api","payload":{},"accepts_response_payload":false}'


# ---------------------------------------------------------------------------
# Fakes: the parts of slack_sdk's aiohttp SocketModeClient the watchdog reads
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for aiohttp's ClientWebSocketResponse."""

    def __init__(self) -> None:
        self.closed = False


class _FakeClient:
    def __init__(self) -> None:
        self.on_message_listeners: list = []
        self.current_session = None
        self.message_processor = None
        self.ping_interval = None  # ping/pong staleness has its own tests; keep it out of these
        self.last_ping_pong_time = None
        self.proxy = None

    async def is_connected(self) -> bool:
        return self.current_session is not None and not self.current_session.closed

    async def deliver(self, data) -> None:
        """Feed one text frame through the receive-loop hook, as receive_messages() does."""
        message = SimpleNamespace(type="TEXT", data=data)
        for listener in list(self.on_message_listeners):
            await listener(message)


class _Fleet:
    """AsyncSocketModeHandler factory. ``hello_plan[i]`` is the ``num_connections`` the i-th
    handler's session reports in its hello, or ``None`` when Slack never sends one."""

    def __init__(self, hello_plan) -> None:
        self.hello_plan = list(hello_plan)
        self.instances: list = []
        self.events: list = []

    def __call__(self, app, app_token, proxy=None):
        fleet = self
        index = len(self.instances)
        connections = self.hello_plan[index] if index < len(self.hello_plan) else 1

        class _Handler:
            def __init__(self) -> None:
                self.client = _FakeClient()
                self.closed = False
                self._stop = asyncio.Event()

            async def start_async(self) -> None:
                self.client.current_session = _FakeSession()
                fleet.events.append(("session", index))
                if connections is not None:
                    await self.client.deliver(_hello(connections))
                    fleet.events.append(("hello", index))
                await self._stop.wait()

            async def close_async(self) -> None:
                self.closed = True
                fleet.events.append(("closed", index))
                if self.client.current_session is not None:
                    self.client.current_session.closed = True
                self._stop.set()

        handler = _Handler()
        self.instances.append(handler)
        return handler


class _Clock:
    """Monotonic clock for the socket module only; asyncio keeps the real one."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock():
    fake = _Clock()
    with patch.object(_socket_mod, "time", SimpleNamespace(monotonic=fake.monotonic, time=time.time)):
        yield fake


@pytest.fixture()
def adapter(clock):
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
    a._app = MagicMock()
    a._app_token = "xapp-fake"
    a._proxy_url = None
    a._running = True
    yield a
    a._running = False


def _record_restarts(adapter) -> list:
    reasons: list = []

    async def _fake_restart(reason: str) -> None:
        reasons.append(reason)

    adapter._restart_socket_mode = _fake_restart
    return reasons


async def _start(adapter, fleet, index=0, hello=True):
    """Start the adapter's handler and wait until its session (and hello) has arrived."""
    adapter._start_socket_mode_handler()
    await async_wait_until(lambda: ("session", index) in fleet.events, what="session opened")
    if hello:
        await async_wait_until(lambda: ("hello", index) in fleet.events, what="hello delivered")


async def _shutdown(adapter):
    adapter._running = False
    await adapter._stop_socket_mode_handler()


# ---------------------------------------------------------------------------
# Registration (hello)
# ---------------------------------------------------------------------------


class TestSocketRegistration:
    @pytest.mark.asyncio
    async def test_registered_session_is_left_alone(self, adapter, clock):
        fleet = _Fleet([1])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            await _start(adapter, fleet)
            await adapter._socket_watchdog_tick()
            clock.advance(adapter._socket_hello_timeout_s + 1)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 1
            assert not fleet.instances[0].closed
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_session_without_hello_is_restarted(self, adapter, clock, caplog):
        """The deaf shape: the socket is up and answers pings, Slack never registers it."""
        fleet = _Fleet([None, 1])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet), \
                caplog.at_level(logging.WARNING):
            await _start(adapter, fleet, hello=False)
            await adapter._socket_watchdog_tick()  # first sight of the session
            clock.advance(adapter._socket_hello_timeout_s - 1)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 1, "restarted before the hello timeout"
            clock.advance(2)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 2
            assert fleet.instances[0].closed
            assert adapter._handler is fleet.instances[1]
            assert "session not registered" in caplog.text
            await async_wait_until(lambda: ("hello", 1) in fleet.events, what="replacement hello")
            await adapter._socket_watchdog_tick()
            clock.advance(adapter._socket_hello_timeout_s + 1)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 2, "a registered replacement must not be restarted"
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_sdk_reconnect_without_hello_is_restarted(self, adapter, clock):
        """The SDK swaps sessions on its own (Slack refresh, CLOSE frame); each one needs a hello."""
        fleet = _Fleet([1, 1])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            await _start(adapter, fleet)
            await adapter._socket_watchdog_tick()
            fleet.instances[0].client.current_session = _FakeSession()  # SDK reconnect, no hello
            await adapter._socket_watchdog_tick()
            clock.advance(adapter._socket_hello_timeout_s + 1)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 2
            assert fleet.instances[0].closed
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_zero_timeouts_disable_the_app_level_checks(self, adapter, clock):
        fleet = _Fleet([None])
        adapter._socket_hello_timeout_s = 0
        adapter._socket_silence_renew_s = 0
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            await _start(adapter, fleet, hello=False)
            await adapter._socket_watchdog_tick()
            clock.advance(86400)
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 1
            await _shutdown(adapter)


# ---------------------------------------------------------------------------
# Silence → make-before-break renewal
# ---------------------------------------------------------------------------


class TestSocketRenewal:
    @pytest.mark.asyncio
    async def test_silent_session_is_renewed_before_the_old_one_closes(self, adapter, clock, caplog):
        fleet = _Fleet([1, 2])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet), \
                caplog.at_level(logging.INFO):
            await _start(adapter, fleet)
            clock.advance(adapter._socket_silence_renew_s + 1)
            await adapter._socket_watchdog_tick()
            assert adapter._handler is fleet.instances[1]
            assert fleet.instances[0].closed and not fleet.instances[1].closed
            # No delivery gap: the replacement was registered before the old session closed.
            assert fleet.events.index(("hello", 1)) < fleet.events.index(("closed", 0))
            assert "Socket Mode renewed after 1801s without Slack frames (connections=2)" in caplog.text
            assert "not registered" not in caplog.text
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_frames_keep_a_busy_session(self, adapter, clock):
        fleet = _Fleet([1])
        adapter._socket_silence_renew_s = 100
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            await _start(adapter, fleet)
            for _ in range(5):
                clock.advance(60)
                await fleet.instances[0].client.deliver(_EVENT)
                await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 1
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_unregistered_candidate_keeps_current_and_backs_off(self, adapter, clock, caplog):
        fleet = _Fleet([1, None, 1])
        adapter._socket_silence_renew_s = 100
        adapter._socket_hello_timeout_s = 0.05  # the candidate wait is real time
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet), \
                caplog.at_level(logging.WARNING):
            await _start(adapter, fleet)
            clock.advance(101)
            await adapter._socket_watchdog_tick()
            assert adapter._handler is fleet.instances[0] and not fleet.instances[0].closed
            assert fleet.instances[1].closed
            assert "did not register a new session" in caplog.text
            await adapter._socket_watchdog_tick()
            assert len(fleet.instances) == 2, "renewal retried inside the backoff window"
            clock.advance(adapter._socket_renew_backoff_s + 1)
            await adapter._socket_watchdog_tick()
            assert adapter._handler is fleet.instances[2]
            assert fleet.instances[0].closed
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_cancelled_renewal_closes_its_candidate(self, adapter, clock):
        """connect()/disconnect() cancel the watchdog; a half-open candidate must not leak."""
        fleet = _Fleet([1, None])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            await _start(adapter, fleet)
            renewal = asyncio.create_task(adapter._renew_socket_mode())
            await async_wait_until(lambda: ("session", 1) in fleet.events, what="candidate opened")
            renewal.cancel()
            with pytest.raises(asyncio.CancelledError):
                await renewal
            assert fleet.instances[1].closed
            assert adapter._handler is fleet.instances[0] and not fleet.instances[0].closed
            await _shutdown(adapter)

    @pytest.mark.asyncio
    async def test_renewal_reports_a_session_slack_had_not_registered(self, adapter, clock, caplog):
        """Slack counting only the candidate means the live session was deaf."""
        fleet = _Fleet([1, 1])
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet), \
                caplog.at_level(logging.WARNING):
            await _start(adapter, fleet)
            await adapter._renew_socket_mode()
            assert adapter._handler is fleet.instances[1]
            assert "was not registered" in caplog.text
            await _shutdown(adapter)


# ---------------------------------------------------------------------------
# Other liveness signals
# ---------------------------------------------------------------------------


def _live_client_adapter(adapter, client):
    adapter._handler = SimpleNamespace(client=client)
    live_task = MagicMock()
    live_task.done.return_value = False
    adapter._socket_mode_task = live_task
    return _record_restarts(adapter)


class TestSocketSignals:
    @pytest.mark.asyncio
    async def test_stopped_message_processor_restarts(self, adapter):
        """Frames keep being read (pongs too) but nothing dispatches them."""
        client = _FakeClient()
        client.current_session = _FakeSession()
        dead = asyncio.get_running_loop().create_future()
        dead.set_result(None)
        client.message_processor = dead
        reasons = _live_client_adapter(adapter, client)
        await adapter._socket_watchdog_tick()
        assert reasons == ["message processor stopped"]

    @pytest.mark.asyncio
    async def test_unobservable_client_keeps_old_behaviour(self, adapter, clock):
        """No on_message_listeners list (older/newer SDK, MagicMock): checks stay off, no loops."""
        client = MagicMock()
        reasons = _live_client_adapter(adapter, client)
        adapter._socket_liveness = adapter._attach_socket_liveness(client)
        clock.advance(86400)
        await adapter._socket_watchdog_tick()
        assert not adapter._socket_liveness.observable
        assert reasons == []

    @pytest.mark.asyncio
    async def test_frame_hook_never_raises(self, adapter, clock):
        client = _FakeClient()
        client.current_session = _FakeSession()
        liveness = adapter._attach_socket_liveness(client)
        clock.advance(5)
        for data in ('x"hello"y', '["hello"]', b'{"type":"hello"}', None,
                     '{"type":"hello","text":"' + "a" * 10000 + '"}', '{"type":"disconnect"'):
            await client.deliver(data)
        assert liveness.last_frame == clock.now
        assert liveness.registered_session is None

    @pytest.mark.asyncio
    async def test_connection_count_above_one_is_reported(self, adapter, caplog):
        client = _FakeClient()
        client.current_session = _FakeSession()
        adapter._socket_liveness = adapter._attach_socket_liveness(client)
        with caplog.at_level(logging.INFO):
            await client.deliver(_hello(connections=2))
        assert adapter._socket_liveness.registered_session is client.current_session
        assert "Socket Mode session registered (connections=2" in caplog.text
        assert "another process using this SLACK_APP_TOKEN" in caplog.text

    @pytest.mark.asyncio
    async def test_disconnect_reasons_are_logged(self, adapter, caplog):
        client = _FakeClient()
        adapter._attach_socket_liveness(client)
        with caplog.at_level(logging.INFO):
            await client.deliver('{"type":"disconnect","reason":"refresh_requested"}')
            await client.deliver('{"type":"disconnect","reason":"link_disabled"}')
        assert "Slack requested a Socket Mode reconnect (reason=refresh_requested)" in caplog.text
        assert "Socket Mode is turned off" in caplog.text


# ---------------------------------------------------------------------------
# Restart hygiene
# ---------------------------------------------------------------------------


def _scripted_connectivity(client, answers):
    remaining = iter(answers)

    async def _is_connected():
        return next(remaining, True)

    client.is_connected = _is_connected


class TestSocketRestartHygiene:
    @pytest.mark.asyncio
    async def test_single_disconnected_sample_does_not_restart(self, adapter):
        """The SDK's own refresh briefly reports disconnected; a restart then only races it."""
        client = _FakeClient()
        client.current_session = _FakeSession()
        _scripted_connectivity(client, [False, True, False, True])
        reasons = _live_client_adapter(adapter, client)
        for _ in range(4):
            await adapter._socket_watchdog_tick()
        assert reasons == []

    @pytest.mark.asyncio
    async def test_two_disconnected_samples_restart(self, adapter):
        client = _FakeClient()
        client.current_session = _FakeSession()
        _scripted_connectivity(client, [False, False])
        reasons = _live_client_adapter(adapter, client)
        await adapter._socket_watchdog_tick()
        await adapter._socket_watchdog_tick()
        assert reasons == ["transport disconnected"]

    @pytest.mark.asyncio
    async def test_restart_skips_a_handler_replaced_while_waiting(self, adapter):
        """Watchdog and the task done-callback can both ask; the second must not tear down the fix."""
        fleet = _Fleet([1])
        adapter._handler = SimpleNamespace(client=_FakeClient())
        healed = SimpleNamespace(client=_FakeClient())
        with patch.object(_slack_mod, "AsyncSocketModeHandler", fleet):
            async with adapter._socket_reconnect_lock:
                pending = asyncio.create_task(adapter._restart_socket_mode("socket task exited"))
                await asyncio.sleep(0)  # let it reach the lock
                adapter._handler = healed
            await pending
        assert adapter._handler is healed
        assert fleet.instances == []

    @pytest.mark.asyncio
    async def test_hung_close_does_not_strand_the_watchdog(self, caplog):
        class _Hung:
            client = _FakeClient()
            release = asyncio.Event()  # never set: the close hangs until cancelled

            async def close_async(self):
                await self.release.wait()

        with patch.object(_socket_mod, "_SOCKET_CLOSE_TIMEOUT_S", 0.05), \
                caplog.at_level(logging.WARNING):
            await asyncio.wait_for(SlackAdapter._close_socket_mode_handler(_Hung(), None), 2.0)
        assert "did not close within" in caplog.text


class TestSocketKnobs:
    def test_env_overrides_and_disables(self):
        with patch.dict(os.environ, {"SLACK_SOCKET_HELLO_TIMEOUT_SECONDS": "0",
                                     "SLACK_SOCKET_SILENCE_RENEW_SECONDS": "600"}):
            a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
        assert a._socket_hello_timeout_s == 0
        assert a._socket_silence_renew_s == 600

    def test_invalid_env_falls_back_to_defaults(self, caplog):
        with patch.dict(os.environ, {"SLACK_SOCKET_HELLO_TIMEOUT_SECONDS": "soon",
                                     "SLACK_SOCKET_SILENCE_RENEW_SECONDS": "-5"}), \
                caplog.at_level(logging.WARNING):
            a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
        assert a._socket_hello_timeout_s == SlackAdapter._socket_hello_timeout_s
        assert a._socket_silence_renew_s == SlackAdapter._socket_silence_renew_s
        assert "Invalid SLACK_SOCKET_HELLO_TIMEOUT_SECONDS='soon'" in caplog.text
