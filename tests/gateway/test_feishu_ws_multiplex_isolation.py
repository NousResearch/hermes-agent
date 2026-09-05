"""Multiplex isolation for the lark_oapi WS client (issue #73779).

``lark_oapi.ws.client`` keeps the loop used by ``Client.start()`` in a
module-level global and Hermes monkey-patches ``websockets.connect`` on the
shared module. With N profile WS threads the globals were last-write-wins:
"Future attached to a different loop" crashes or a client bound to a
sibling's loop that never hears anything again.
"""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    lark.ws = lark_ws
    lark_ws.client = client_mod
    monkeypatch.setitem(sys.modules, "lark_oapi", lark)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", lark_ws)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_mod)
    monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
    return client_mod


def _adapter_stub(**overrides):
    stub = SimpleNamespace(
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


def _supervisor_stub():
    stub = SimpleNamespace(
        _running=True,
        _ws_future=None,
        _ws_client=object(),
        _ws_restart_backoff=0.01,
        connect_calls=0,
        connect_should_fail=0,
    )

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
        return stub.connect_calls

    assert asyncio.run(scenario()) == 2  # failed restart, then a successful one


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
