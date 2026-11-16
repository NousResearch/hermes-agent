"""Receive/open timeouts for the lark_oapi WebSocket client (#156356).

Companion to ``test_feishu_ws_multiplex_isolation.py``: the isolation tests
cover loop/override ownership; these cover the timeout behavior bolted onto
the same seams — a stalled ``recv()`` must raise into the SDK's reconnect
ladder instead of parking the loop in ``SSL_read`` forever.
"""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.platforms.feishu import adapter as feishu_adapter
from plugins.platforms.feishu import adapter_ws


def _inject_fake_lark_module(monkeypatch, connect=None):
    """Same fake-module layout as the isolation tests, plus ``requests``."""
    if connect is None:
        connect = MagicMock(name="real-connect")
    lark = types.ModuleType("lark_oapi")
    lark_ws = types.ModuleType("lark_oapi.ws")
    client_mod = types.ModuleType("lark_oapi.ws.client")
    client_mod.loop = SimpleNamespace(name="sdk-default-loop")
    client_mod.websockets = SimpleNamespace(connect=connect)
    client_mod.requests = SimpleNamespace(post=MagicMock(name="sdk-post"))

    class Client:
        async def _receive_message_loop(self):
            await asyncio.Future()

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
        _ws_read_timeout=None,
        _ws_open_timeout=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


async def _parked_forever():
    """A coroutine that never completes and is not a wall-clock sleep (HX013)."""
    await asyncio.Future()


class _StalledConnection:
    """A connection whose recv() never returns — the half-dead TCP symptom."""

    def __init__(self):
        self.transport = SimpleNamespace(abort=MagicMock(name="abort"))
        self.aborted = threading.Event()
        transport_abort = self.transport.abort

        def _record_abort():
            self.aborted.set()
            transport_abort()

        self.transport.abort = _record_abort
        self.recv_entered = threading.Event()

    async def recv(self, *args, **kwargs):
        self.recv_entered.set()
        await asyncio.Future()

    async def send(self, data):
        await asyncio.Future()

    async def close(self):
        return None


def test_stalled_recv_times_out_and_unparks_the_worker(monkeypatch):
    """#156356 red test: on main, recv() has no budget, so the SDK's receive
    loop parks forever on a stalled connection and the worker thread never
    finishes. With the receive budget, the timeout aborts the transport, the
    SDK's exception handler runs, and (auto-reconnect off) our exit wrapper
    stops the worker loop so the supervisor can rebuild."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    stalled = _StalledConnection()

    class FakeSDKClient:
        """Mirrors lark_oapi's structure: connect stores ``_conn`` and schedules
        the receive loop; the receive loop's except runs _disconnect and, with
        ``_auto_reconnect`` disabled, re-raises (which is what our exit wrapper
        turns into a worker stop)."""

        def __init__(self):
            self._conn = None
            self._auto_reconnect = False
            self.disconnected = threading.Event()

        async def _receive_message_loop(self):
            try:
                while True:
                    if self._conn is None:
                        raise RuntimeError("connection is closed")
                    await self._conn.recv()
                    # a message arrived; ignore
            except Exception:
                await self._disconnect()
                if self._auto_reconnect:
                    await self._reconnect()
                else:
                    raise

        async def _disconnect(self):
            conn, self._conn = self._conn, None
            if conn is not None:
                await conn.close()
            self.disconnected.set()

        async def _reconnect(self):  # pragma: no cover - not used in this test
            raise AssertionError("auto-reconnect path must not run here")

        def start(self):
            loop = client_mod.loop
            self._conn = stalled

            async def _select():
                await _parked_forever()

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    stub = _adapter_stub(_ws_read_timeout=0.1)

    # The install wraps ``Client._receive_message_loop`` on the SDK module, so the
    # fake class must BE that Client for the wrapper (and the budget attach) to run.
    client_mod.Client = FakeSDKClient

    thread = threading.Thread(
        target=lambda: feishu_adapter._run_official_feishu_ws_client(
            FakeSDKClient(), stub
        ),
        daemon=True,
    )
    try:
        thread.start()
        assert stalled.recv_entered.wait(timeout=5), "receive loop never started"
        thread.join(timeout=5)
        # On main the worker parks inside start() forever: recv() never returns,
        # no exception reaches the exit wrapper, and this assert fails.
        assert not thread.is_alive(), (
            "worker wedged on a stalled recv (no receive budget)"
        )
        # The stalled transport was aborted before the SDK tried its graceful close.
        assert stalled.aborted.is_set(), "stalled transport was not aborted"
        assert stub._ws_thread_loop is None
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_stalled_recv_with_autoreconnect_recovers_via_sdk_ladder(monkeypatch):
    """The common production shape: auto-reconnect on. The timeout must hand
    control to the SDK's own ladder — disconnect, reconnect, fresh connection
    receives again — without stopping the worker loop, and the fresh link must
    itself be budget-wrapped (a second stall also recovers)."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    first_conn = _StalledConnection()
    second_stall = threading.Event()
    recovered = threading.Event()

    class RecoveringConnection:
        """First recv proves recovery; then the link stalls again."""

        def __init__(self):
            self.calls = 0

        async def recv(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                recovered.set()
            # Park after the first message: a quiet link that then stalls —
            # the SECOND recv must also be bounded by the re-attached proxy.
            second_stall.set()
            await asyncio.Future()

        async def send(self, data):
            return None

        async def close(self):
            return None

    class FakeSDKClient:
        def __init__(self):
            self._conn = None
            self._auto_reconnect = True
            self._reconnect_count = 3
            self._reconnect_nonce = 0
            self._reconnect_interval = 1
            self.reconnected = threading.Event()

        async def _receive_message_loop(self):
            try:
                while True:
                    if self._conn is None:
                        raise RuntimeError("connection is closed")
                    await self._conn.recv()
            except Exception:
                await self._disconnect()
                if self._auto_reconnect:
                    await self._reconnect()
                else:
                    raise

        async def _disconnect(self):
            conn, self._conn = self._conn, None
            if conn is not None:
                await conn.close()

        async def _reconnect(self):
            # Ladder's successful retry: fresh connection, fresh receive loop
            # (which the budget must re-wrap — proven by the second stall).
            self._conn = RecoveringConnection()
            client_mod.loop.create_task(self._receive_message_loop())
            self.reconnected.set()

        def start(self):
            loop = client_mod.loop
            self._conn = first_conn

            async def _select():
                await _parked_forever()

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    stub = _adapter_stub(_ws_read_timeout=0.1)
    client_mod.Client = FakeSDKClient

    thread = threading.Thread(
        target=lambda: feishu_adapter._run_official_feishu_ws_client(
            FakeSDKClient(), stub
        ),
        daemon=True,
    )
    try:
        thread.start()
        # The stalled first link times out, the ladder swaps in the fresh
        # connection, and its first recv runs — all without the worker dying.
        assert recovered.wait(timeout=5), (
            "SDK reconnect ladder never reached the fresh connection"
        )
        # The fresh link is budget-wrapped too: its second (stalled) recv must
        # ALSO time out and drive a second ladder round (reconnect fires again).
        assert second_stall.wait(timeout=5), (
            "fresh connection never received its first message"
        )
        assert client_mod.Client.__name__ == "FakeSDKClient"
        # Second stall on the fresh link resolves through the ladder again.
        assert thread.join(timeout=5) is None and thread.is_alive(), (
            "worker must survive recoverable stalls (SDK ladder owns them)"
        )
        assert first_conn.aborted.is_set()
        assert stub._ws_thread_loop is not None
    finally:
        # Clean shutdown of the parked worker for test hygiene.
        stub_loop = stub._ws_thread_loop
        if stub_loop is not None:
            stub_loop.call_soon_threadsafe(stub_loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_effective_read_timeout_resolution():
    # Explicit setting wins, including fractional test budgets.
    assert adapter_ws.effective_read_timeout(0.1, None) == 0.1
    assert adapter_ws.effective_read_timeout(12.0, 30) == 12.0
    # Explicit zero is the operator opt-out.
    assert adapter_ws.effective_read_timeout(0, 30) is None
    # Auto budget from the live ping interval, floored at 300s.
    assert adapter_ws.effective_read_timeout(None, 45) == 300.0
    assert adapter_ws.effective_read_timeout(None, 120) == 360.0
    # Nothing to derive from: no budget (pre-fix behavior stays reachable).
    assert adapter_ws.effective_read_timeout(None, None) is None


def test_auto_budget_follows_the_live_ping_interval(monkeypatch):
    """The auto budget must track the client's LIVE heartbeat, not a value
    captured at worker start: a long ping interval must not be cut down by a
    stale 300s floor budget."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    stalled = _StalledConnection()
    wrapped_budgets = []

    class FakeSDKClient:
        def __init__(self):
            self._conn = None
            self._auto_reconnect = False
            self._ping_interval = 600  # what _apply_runtime_ws_overrides installs

        async def _receive_message_loop(self):
            try:
                while True:
                    await self._conn.recv()
            except Exception:
                await self._disconnect()
                raise

        async def _disconnect(self):
            conn, self._conn = self._conn, None
            if conn is not None:
                await conn.close()

        def start(self):
            loop = client_mod.loop
            self._conn = stalled
            # Capture the budget the wrapper installed on this link.
            original_recv = type(stalled).recv

            async def _probe_recv(inner, *args, **kwargs):
                proxy = self._conn
                wrapped_budgets.append(getattr(proxy, "_read_timeout", "unwrapped"))
                return await original_recv(inner, *args, **kwargs)

            type(stalled).recv = _probe_recv
            try:
                loop.create_task(self._receive_message_loop())

                async def _select():
                    await _parked_forever()

                loop.run_until_complete(_select())
            finally:
                type(stalled).recv = original_recv

    client = FakeSDKClient()
    client_mod.Client = FakeSDKClient
    stub = _adapter_stub()  # no explicit read timeout: auto from live ping

    thread = threading.Thread(
        target=lambda: feishu_adapter._run_official_feishu_ws_client(client, stub),
        daemon=True,
    )
    try:
        thread.start()
        assert stalled.recv_entered.wait(timeout=5), "receive loop never started"
        # Live ping 600s -> budget 1800s, NOT the 300s floor of a stale default.
        assert wrapped_budgets == [1800.0], (
            f"auto budget did not follow live ping: {wrapped_budgets}"
        )
    finally:
        stub_loop = stub._ws_thread_loop
        if stub_loop is not None:
            stub_loop.call_soon_threadsafe(stub_loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_open_timeout_flows_into_connect_overrides(monkeypatch):
    """ws_open_timeout rides the same thread-local connect-override channel the
    ping settings use, and explicit caller kwargs keep precedence."""
    real_connect = MagicMock(name="real-connect")
    client_mod = _inject_fake_lark_module(monkeypatch, connect=real_connect)

    class FakeClient:
        def start(self):
            client_mod.websockets.connect("wss://unit")
            client_mod.websockets.connect("wss://explicit", open_timeout=99)

    stub = _adapter_stub(_ws_open_timeout=7.5)
    feishu_adapter._run_official_feishu_ws_client(FakeClient(), stub)
    calls = [c.kwargs for c in real_connect.call_args_list]
    assert calls[0].get("open_timeout") == 7.5
    assert calls[1].get("open_timeout") == 99  # explicit wins over setdefault


def test_bootstrap_post_gets_a_budget_on_registered_threads(monkeypatch):
    """The SDK module's requests.post gains a finite (connect, read) budget for
    registered worker threads; other callers keep the exact old behavior."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    real_post = (
        client_mod.requests.post
    )  # captured before the facade replaces the attribute

    class FakeClient:
        def start(self):
            client_mod.requests.post("https://endpoint.example", headers={}, json={})

    stub = _adapter_stub(_ws_open_timeout=5.0)
    feishu_adapter._run_official_feishu_ws_client(FakeClient(), stub)
    # Facade is installed over the SDK's module attribute.
    assert isinstance(client_mod.requests, adapter_ws.TimedRequestsFacade)
    (call,) = real_post.call_args_list
    # (connect, read): the open timeout bounds connecting, 30s bounds a stalled read.
    assert call.kwargs.get("timeout") == (5.0, 30.0)


@pytest.mark.parametrize(
    "value",
    [
        "garbage",
        0,
        -1,
        float("nan"),
        float("inf"),
        pytest.param(10**1000, id="conversion-overflow"),
    ],
)
def test_invalid_ws_open_timeout_fails_before_it_disables_the_guard(value):
    """An unusable explicit value must not coerce to None: that silently opts out
    of both the connect override and the bootstrap POST budget (#156394 review)."""
    with pytest.raises(ValueError, match=r"feishu\.ws_open_timeout"):
        feishu_adapter.FeishuAdapter._load_settings({"ws_open_timeout": value})


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({}, 10.0),
        ({"ws_open_timeout": None}, None),
        ({"ws_open_timeout": 7.5}, 7.5),
        ({"ws_open_timeout": "7.5"}, 7.5),
    ],
)
def test_ws_open_timeout_preserves_supported_settings(extra, expected):
    settings = feishu_adapter.FeishuAdapter._load_settings(extra)
    assert settings.ws_open_timeout == expected
