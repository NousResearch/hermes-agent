"""Multiplex isolation for the lark_oapi WS client (issue #73779).

``lark_oapi.ws.client`` keeps the loop used by ``Client.start()`` in a
module-level global and Hermes monkey-patches ``websockets.connect`` on the
shared module. With N profile WS threads the globals were last-write-wins:
"Future attached to a different loop" crashes or a client bound to a
sibling's loop that never hears anything again.
"""

import asyncio
import logging
import socket
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import requests

from gateway.config import PlatformConfig
from plugins.platforms.feishu import adapter as feishu_adapter


def _inject_fake_lark_module(monkeypatch, connect=None):
    """Make ``import lark_oapi.ws.client`` resolve to a module with the SDK's
    global layout (``loop`` + ``websockets.connect``)."""
    if connect is None:
        connect = MagicMock(name="real-connect")
    lark = types.ModuleType("lark_oapi")
    lark_ws = types.ModuleType("lark_oapi.ws")
    client_mod = types.ModuleType("lark_oapi.ws.client")
    client_mod.loop = SimpleNamespace(name="sdk-default-loop")
    client_mod.websockets = SimpleNamespace(connect=connect)

    class Client:  # the SDK class whose receive loop the isolation shim wraps
        async def _receive_message_loop(self):
            await asyncio.sleep(3600)

    client_mod.Client = Client
    lark.ws = lark_ws
    lark_ws.client = client_mod
    monkeypatch.setitem(sys.modules, "lark_oapi", lark)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", lark_ws)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_mod)
    monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
    return client_mod


def _adapter_stub(**overrides):
    stub = SimpleNamespace(
        _loop=None,
        _ws_thread_loop=None,
        _ws_reconnect_nonce=None,
        _ws_reconnect_interval=None,
        _ws_ping_interval=None,
        _ws_ping_timeout=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def test_two_concurrent_clients_each_use_their_own_loop_and_overrides(monkeypatch):
    """Two profiles start() concurrently through the module global: each must
    run on its own loop, and websockets.connect must receive only the
    calling profile's ping overrides. On main both are last-write-wins."""
    real_connect = MagicMock(name="real-connect")
    client_mod = _inject_fake_lark_module(monkeypatch, connect=real_connect)

    results = {}
    barrier = threading.Barrier(2)

    class FakeClient:
        def __init__(self, name):
            self._name = name

        def start(self):
            barrier.wait(timeout=10)  # both threads past the global "assign"

            async def probe():
                await asyncio.sleep(0.02)
                return id(asyncio.get_running_loop())

            results[self._name] = client_mod.loop.run_until_complete(probe())
            client_mod.websockets.connect(f"wss://{self._name}")

    pings = {"p0": 10, "p1": 20}

    def run(name):
        feishu_adapter._run_official_feishu_ws_client(
            FakeClient(name), _adapter_stub(_ws_ping_interval=pings[name])
        )

    threads = [threading.Thread(target=run, args=(f"p{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()

    assert results["p0"] != results["p1"]
    calls = {c.args[0]: c.kwargs for c in real_connect.call_args_list}
    assert calls == {"wss://p0": {"ping_interval": 10}, "wss://p1": {"ping_interval": 20}}
    # Thread-local registrations are cleared for the pooled executor thread.
    assert getattr(feishu_adapter._ws_isolation_state, "loop", None) is None
    assert getattr(feishu_adapter._ws_isolation_state, "connect_kwargs", None) is None


def test_dead_receive_loop_unparks_start_and_exits_the_thread(monkeypatch, caplog):
    """#113662: with the SDK's reconnect ladder disabled, a receive-loop death
    used to strand ``start()`` in ``run_until_complete(_select())`` forever —
    the thread stayed alive on a deaf socket and the supervisor's executor
    future never completed. The isolation wrap must stop the worker loop so
    ``start()`` raises and the thread exits for the supervisor to rebuild."""
    client_mod = _inject_fake_lark_module(monkeypatch)

    class FakeSDKClient:
        def __init__(self):
            self._auto_reconnect = False

        async def _receive_message_loop(self):
            # Mirror lark_oapi 1.6.8: bare raise out of an unawaited create_task
            # (reconnect ladder disabled, or its ClientException re-raise).
            await asyncio.sleep(0.01)
            raise ConnectionError("simulated half-open peer")

        def start(self):
            loop = client_mod.loop

            async def _select():  # the SDK's forever-parked select loop
                while True:
                    await asyncio.sleep(3600)

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    client_mod.Client = FakeSDKClient
    stub = _adapter_stub()

    def run():
        feishu_adapter._run_official_feishu_ws_client(FakeSDKClient(), stub)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=10)
    # On main the thread parks inside start() forever and this assert fails.
    assert not thread.is_alive()
    # The executor future completes: _run_official_feishu_ws_client ran its
    # full teardown, which is what _supervise_websocket_thread awaits.
    assert stub._ws_thread_loop is None
    # The root cause is logged (with traceback) next to the supervisor's
    # rebuild line instead of dying unretrieved in the SDK's bare create_task.
    deaths = [r for r in caplog.records if "receive loop died" in r.getMessage()]
    assert deaths, "expected the receive-loop death to be logged"
    assert deaths[0].exc_info is not None
    assert "simulated half-open peer" in caplog.text


def test_receive_loop_death_during_disconnect_is_not_an_error(monkeypatch, caplog):
    """``disconnect()`` clears ``_running`` and sends the CLOSE frame itself, so the
    receive loop ending with ``ConnectionClosedOK`` is the expected shutdown path —
    it must not be reported as a died link (ERROR + traceback) on every graceful stop."""
    client_mod = _inject_fake_lark_module(monkeypatch)

    class FakeSDKClient:
        async def _receive_message_loop(self):
            await asyncio.sleep(0.01)
            raise ConnectionError("sent 1000 (OK); then received 1000 (OK)")

        def start(self):
            loop = client_mod.loop

            async def _select():
                while True:
                    await asyncio.sleep(3600)

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    client_mod.Client = FakeSDKClient
    stub = _adapter_stub(_running=False)  # disconnect() already flipped it

    thread = threading.Thread(
        target=lambda: feishu_adapter._run_official_feishu_ws_client(FakeSDKClient(), stub), daemon=True
    )
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and "receive loop" in r.getMessage()]
    assert errors == [], [r.getMessage() for r in errors]


def test_receive_loop_normal_return_keeps_start_parked(monkeypatch):
    """The exit-notify wrap must only fire on an exception. When the SDK's own
    reconnect ladder succeeds, the old receive loop *returns* (a fresh one was
    scheduled by ``_connect``) — stopping the loop there would tear down the
    healthy rebuilt link on every transient blip."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    returned = threading.Event()
    loop_holder = {}

    class FakeSDKClient:
        async def _receive_message_loop(self):
            await asyncio.sleep(0.01)
            returned.set()  # ladder reconnected: coroutine returns normally

        def start(self):
            loop = client_mod.loop
            # Resolve the real worker loop from inside the thread (the module
            # global is a thread-local proxy that falls back off-thread).
            loop_holder["loop"] = asyncio.get_event_loop()

            async def _select():
                while True:
                    await asyncio.sleep(3600)

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    client_mod.Client = FakeSDKClient
    stub = _adapter_stub()

    def run():
        feishu_adapter._run_official_feishu_ws_client(FakeSDKClient(), stub)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert returned.wait(timeout=10)
    thread.join(timeout=0.5)
    assert thread.is_alive()  # start() stays parked on the rebuilt link
    # Clean teardown for the test process: stop the worker loop from this
    # (foreign) thread, then let the thread's own finally block close it out.
    loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_sdk_reconnect_ladder_publishes_retrying(monkeypatch):
    """#113662, live path: with ``_auto_reconnect`` on, a receive-loop error runs
    the SDK's own ``_reconnect()`` ladder inside the WS thread — the thread does
    not die, so the supervisor never publishes ``retrying``. The SDK's
    ``on_reconnecting`` observer must hop to the adapter loop and publish it."""
    _inject_fake_lark_module(monkeypatch)
    adapter_loop = asyncio.new_event_loop()

    class FakeSDKClient:
        def __init__(self):
            self.on_reconnecting = lambda: None  # SDK default: no-op observer

        def start(self):
            self.on_reconnecting()  # what lark_oapi ``_reconnect()`` fires first

    client = FakeSDKClient()
    stub = _adapter_stub(_loop=adapter_loop, _running=True, _ws_client=client, status_writes=[])
    stub._write_runtime_status_safe = lambda context, **kw: stub.status_writes.append(kw["platform_state"])
    stub._ws_link_retrying = types.MethodType(feishu_adapter.FeishuAdapter._ws_link_retrying, stub)
    stub._arm_ws_stuck_timer = lambda: None

    thread = threading.Thread(target=feishu_adapter._run_official_feishu_ws_client, args=(client, stub), daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()

    async def drain():
        for _ in range(100):
            if stub.status_writes:
                break
            await asyncio.sleep(0.01)

    adapter_loop.run_until_complete(drain())
    adapter_loop.close()
    assert stub.status_writes == ["retrying"]


def test_stuck_endpoint_discovery_times_out_instead_of_blocking_forever(monkeypatch):
    """lark_oapi's endpoint POST sets no timeout: a TCP link that survived a network
    change or sleep blocked the WS thread in a socket read forever, so the profile
    stayed deaf with no log line and no retry. The call must raise, because the SDK's
    ``_try_connect`` turns any exception into a failed attempt its ladder retries."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    client_mod.requests = requests
    monkeypatch.setattr(feishu_adapter, "_WS_ENDPOINT_HTTP_TIMEOUT_SECONDS", (1, 0.2))
    monkeypatch.setenv("NO_PROXY", "*")
    feishu_adapter._install_lark_ws_isolation(client_mod)

    raised = []

    def discover_endpoint(url):
        try:
            client_mod.requests.post(url, headers={}, json={"AppID": "app-id"})  # the SDK's call shape
        except requests.exceptions.Timeout as exc:
            raised.append(exc)

    # The kernel completes the handshake from the backlog; nothing ever accepts or replies.
    with socket.create_server(("127.0.0.1", 0)) as blackhole:
        url = f"http://127.0.0.1:{blackhole.getsockname()[1]}/callback/ws/endpoint"
        thread = threading.Thread(target=discover_endpoint, args=(url,), daemon=True)
        thread.start()
        thread.join(timeout=5)
        stuck = thread.is_alive()

    assert not stuck, "the endpoint POST is still blocked on a peer that never answers"
    assert len(raised) == 1


def _supervisor_stub():
    stub = SimpleNamespace(
        _running=True,
        _ws_future=None,
        _ws_client=object(),
        _ws_restart_backoff=0.01,
        connect_calls=0,
        connect_should_fail=0,
        status_writes=[],
    )
    stub._write_runtime_status_safe = lambda context, **kw: stub.status_writes.append(kw["platform_state"])
    stub._arm_ws_stuck_timer = lambda: None

    async def _connect_websocket():
        stub.connect_calls += 1
        if stub.connect_should_fail > 0:
            stub.connect_should_fail -= 1
            raise RuntimeError("simulated restart failure")
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)  # new thread dies immediately too
        stub._ws_future = fut

    stub._connect_websocket = _connect_websocket
    return stub


def test_supervisor_restarts_a_dead_ws_thread_with_backoff():
    """A dead WS thread used to leave the profile silently deaf (the future
    was awaited only by disconnect()). The supervisor must rebuild the client
    and survive a failed restart without hot-looping."""

    async def scenario():
        stub = _supervisor_stub()
        stub.connect_should_fail = 1
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)  # the WS "thread" is already dead
        stub._ws_future = fut

        task = asyncio.ensure_future(
            feishu_adapter.FeishuAdapter._supervise_websocket_thread(stub)
        )
        for _ in range(300):
            await asyncio.sleep(0.01)
            if stub.connect_calls >= 2:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return stub

    stub = asyncio.run(scenario())
    assert stub.connect_calls == 2  # failed restart, then a successful one
    # The lost link is published as ``retrying`` (``connected`` is re-stamped by ``_ws_link_up``).
    assert stub.status_writes[0] == "retrying"


def test_supervisor_stops_when_disconnect_nils_the_client():
    async def scenario():
        stub = _supervisor_stub()
        fut = asyncio.get_running_loop().create_future()  # thread "alive"
        stub._ws_future = fut

        task = asyncio.ensure_future(
            feishu_adapter.FeishuAdapter._supervise_websocket_thread(stub)
        )
        await asyncio.sleep(0.01)
        stub._ws_client = None  # deliberate disconnect ...
        fut.set_result(None)  # ... then the thread exits
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        return stub, task

    stub, task = asyncio.run(scenario())
    assert task.done()
    assert stub.connect_calls == 0


def _real_adapter():
    """A real adapter with a live link registered. Tests fire the stuck timer themselves, not the clock."""
    adapter = feishu_adapter.FeishuAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "app_secret": "b"}))
    adapter._running = True
    adapter._ws_client = object()
    adapter.fatal_notifications = []

    async def _on_fatal(failed):
        adapter.fatal_notifications.append(failed.fatal_error_code)

    adapter.set_fatal_error_handler(_on_fatal)
    return adapter


def test_link_stuck_in_retrying_goes_to_the_gateway_reconnect_queue():
    """A wedged SDK call neither exits the WS thread nor reports the link up, so the
    profile stayed deaf behind ``retrying`` for hours with nothing escalating it."""
    adapter = _real_adapter()

    async def scenario():
        adapter._ws_link_retrying(adapter._ws_client)
        adapter._on_ws_stuck_timer()
        await adapter._ws_stuck_notify_task

    asyncio.run(scenario())
    assert adapter.fatal_error_code == feishu_adapter._WS_STUCK_ERROR_CODE
    assert adapter.fatal_error_retryable is True  # non-retryable would never be rebuilt
    assert adapter.fatal_notifications == [feishu_adapter._WS_STUCK_ERROR_CODE]


def test_retrying_ladder_rungs_do_not_restart_the_stuck_clock():
    """Only continuous failure escalates, so repeated ``retrying`` reports keep the first deadline."""
    adapter = _real_adapter()

    async def scenario():
        adapter._ws_link_retrying(adapter._ws_client)
        first = adapter._ws_stuck_timer
        adapter._ws_link_retrying(adapter._ws_client)
        assert adapter._ws_stuck_timer is first
        assert not first.cancelled()
        adapter._cancel_ws_stuck_timer()

    asyncio.run(scenario())


def test_link_recovering_in_time_is_not_escalated():
    adapter = _real_adapter()

    async def scenario():
        adapter._ws_link_retrying(adapter._ws_client)
        pending = adapter._ws_stuck_timer
        adapter._ws_link_up(adapter._ws_client)
        assert pending.cancelled()

    asyncio.run(scenario())
    assert adapter.fatal_error_code is None


def test_running_turn_defers_the_rebuild_until_it_ends():
    """The turn's reply still goes out through this adapter (REST works without the WS link)."""
    adapter = _real_adapter()

    async def scenario():
        adapter._active_sessions["session"] = asyncio.Event()
        adapter._ws_link_retrying(adapter._ws_client)
        adapter._on_ws_stuck_timer()
        assert adapter.fatal_error_code is None
        assert adapter._ws_stuck_timer is not None  # still watching
        adapter._active_sessions.clear()
        adapter._on_ws_stuck_timer()
        await adapter._ws_stuck_notify_task

    asyncio.run(scenario())
    assert adapter.fatal_notifications == [feishu_adapter._WS_STUCK_ERROR_CODE]


def test_wedged_session_cannot_defer_the_rebuild_forever():
    adapter = _real_adapter()

    async def scenario():
        adapter._active_sessions["session"] = asyncio.Event()
        adapter._ws_link_retrying(adapter._ws_client)
        adapter._ws_retrying_since -= feishu_adapter._WS_STUCK_BUSY_LIMIT_SECONDS
        adapter._on_ws_stuck_timer()
        await adapter._ws_stuck_notify_task

    asyncio.run(scenario())
    assert adapter.fatal_notifications == [feishu_adapter._WS_STUCK_ERROR_CODE]


def test_disconnect_stops_the_stuck_watch():
    """A deliberate shutdown must not report a stuck link later."""
    adapter = _real_adapter()

    async def scenario():
        adapter._ws_link_retrying(adapter._ws_client)
        pending = adapter._ws_stuck_timer
        await adapter.disconnect()
        assert pending.cancelled()

    asyncio.run(scenario())
    assert adapter.fatal_error_code is None


def test_stuck_link_teardown_does_not_wait_on_the_wedged_thread():
    """The runner gives disconnect() 5s, then detaches it. A detached disconnect that ends after the
    replacement connected would release the app lock the replacement holds, so it must finish in time."""
    adapter = _real_adapter()
    adapter._fatal_error_code = feishu_adapter._WS_STUCK_ERROR_CODE
    close_link = AsyncMock()
    wedged_thread_loop = asyncio.new_event_loop()  # never runs, like a loop whose thread is blocked

    async def scenario():
        adapter._ws_future = asyncio.get_running_loop().create_future()  # the thread never exits
        teardown = asyncio.ensure_future(
            adapter._teardown_ws_thread(SimpleNamespace(_disconnect=close_link), wedged_thread_loop)
        )
        await asyncio.sleep(0)
        assert teardown.done()  # both waits skipped: it never yielded to a timeout

    asyncio.run(scenario())
    wedged_thread_loop.close()
    close_link.assert_not_called()  # the SDK already dropped the socket; there is nothing to CLOSE
