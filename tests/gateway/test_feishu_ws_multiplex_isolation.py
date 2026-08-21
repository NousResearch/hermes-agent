"""Multiplex isolation for the lark_oapi WS client (issue #73779).

In multiplex mode every Feishu profile runs its own official lark_oapi WS
client on a dedicated thread. ``lark_oapi.ws.client`` keeps the loop used by
``Client.start()`` and all of its coroutines in a module-level global, and
Hermes additionally monkey-patches ``websockets.connect`` on the shared
``websockets`` module. Before this fix the N profile threads overwrote each
other's globals (last-write-wins), which either crashed clients with
"Future attached to a different loop" or bound a client to a sibling's loop
at construction time so the profile never heard anything again.

These tests cover:

  * the thread-local loop proxy (per-thread dispatch + fallback),
  * the per-thread ``websockets.connect`` override dispatcher,
  * the legacy fallback path when the shims cannot install,
  * the supervised restart of a dead WS client thread.
"""

import asyncio
import inspect
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from plugins.platforms.feishu import adapter as feishu_adapter


def _fake_ws_module(connect=None):
    """A stand-in for ``lark_oapi.ws.client`` with the same global layout."""
    if connect is None:
        async def connect(url, **kwargs):  # pragma: no cover - not awaited
            return MagicMock()

    websockets = SimpleNamespace(connect=connect)
    return SimpleNamespace(loop=SimpleNamespace(name="sdk-default-loop"), websockets=websockets)


def _inject_fake_lark_module(monkeypatch, module):
    """Make ``import lark_oapi.ws.client`` resolve to ``module``."""
    lark = types.ModuleType("lark_oapi")
    lark_ws = types.ModuleType("lark_oapi.ws")
    client_mod = types.ModuleType("lark_oapi.ws.client")
    client_mod.loop = module.loop
    client_mod.websockets = module.websockets
    lark.ws = lark_ws
    lark_ws.client = client_mod
    monkeypatch.setitem(sys.modules, "lark_oapi", lark)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", lark_ws)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_mod)
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


class TestThreadLocalLoopProxy:
    def test_dispatches_to_the_calling_threads_loop(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        mod = _fake_ws_module()
        feishu_adapter._install_lark_ws_isolation(mod)

        results = {}
        barrier = threading.Barrier(2)

        def worker(name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            feishu_adapter._ws_isolation_state.loop = loop
            try:
                # Both threads exercise the same module global concurrently;
                # each must see its own loop.
                barrier.wait(timeout=5)

                async def probe():
                    await asyncio.sleep(0.01)
                    return threading.get_ident()

                results[name] = mod.loop.run_until_complete(probe())
                results[f"{name}-loop-id"] = id(loop)
            finally:
                feishu_adapter._ws_isolation_state.loop = None
                loop.close()

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",)) for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive()

        assert results["t0"] != results["t1"]
        assert results["t0-loop-id"] != results["t1-loop-id"]

    def test_unregistered_thread_falls_back_to_sdk_loop(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        sdk_loop = MagicMock(name="sdk-default-loop")
        mod = _fake_ws_module()
        mod.loop = sdk_loop
        feishu_adapter._install_lark_ws_isolation(mod)

        # No TLS registration on this thread -> forwards to the SDK default.
        assert mod.loop.is_closed is sdk_loop.is_closed

    def test_install_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        mod = _fake_ws_module()
        feishu_adapter._install_lark_ws_isolation(mod)
        proxy = mod.loop
        dispatcher = mod.websockets.connect
        feishu_adapter._install_lark_ws_isolation(mod)
        assert mod.loop is proxy
        assert mod.websockets.connect is dispatcher


class TestConnectDispatcher:
    def test_applies_only_the_calling_threads_overrides(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        real_connect = MagicMock(name="real-connect")
        mod = _fake_ws_module(connect=real_connect)
        feishu_adapter._install_lark_ws_isolation(mod)

        feishu_adapter._ws_isolation_state.connect_kwargs = {"ping_interval": 15}
        mod.websockets.connect("wss://a", ping_timeout=5)
        real_connect.assert_called_once_with(
            "wss://a", ping_timeout=5, ping_interval=15
        )

        # A thread without registered overrides connects untouched.
        real_connect.reset_mock()
        seen = {}

        def other_thread():
            seen["called"] = False
            mod.websockets.connect("wss://b")
            seen["called"] = True

        t = threading.Thread(target=other_thread)
        t.start()
        t.join(timeout=5)
        assert seen["called"]
        real_connect.assert_called_once_with("wss://b")

        feishu_adapter._ws_isolation_state.connect_kwargs = None

    def test_explicit_kwargs_win_over_overrides(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        real_connect = MagicMock(name="real-connect")
        mod = _fake_ws_module(connect=real_connect)
        feishu_adapter._install_lark_ws_isolation(mod)

        feishu_adapter._ws_isolation_state.connect_kwargs = {"ping_interval": 15}
        mod.websockets.connect("wss://a", ping_interval=99)
        real_connect.assert_called_once_with("wss://a", ping_interval=99)
        feishu_adapter._ws_isolation_state.connect_kwargs = None

    def test_signature_probe_still_sees_real_connect(self, monkeypatch):
        """The SDK's ``_ws_connect_kwargs()`` inspects the connect signature
        to detect websockets-15 ``proxy`` support; the dispatcher must not
        hide it."""
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)

        async def real_connect(url, *, proxy=None, **kwargs):
            return None

        mod = _fake_ws_module(connect=real_connect)
        feishu_adapter._install_lark_ws_isolation(mod)
        params = inspect.signature(mod.websockets.connect).parameters
        assert "proxy" in params


class TestRunOfficialClient:
    def test_isolated_path_registers_tls_and_cleans_up(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        mod = _fake_ws_module()
        client_mod = _inject_fake_lark_module(monkeypatch, mod)

        seen = {}

        class FakeClient:
            def start(self):
                # SDK-style access through the module global: only works when
                # the proxy dispatches to this thread's registered loop.
                client_mod.loop.run_until_complete(asyncio.sleep(0))
                seen["ran_on_loop"] = asyncio.get_event_loop()

        adapter = _adapter_stub(_ws_ping_interval=30)
        feishu_adapter._run_official_feishu_ws_client(FakeClient(), adapter)

        assert "ran_on_loop" in seen
        assert seen["ran_on_loop"].is_closed()
        assert adapter._ws_thread_loop is None
        # TLS entries are cleaned up so a later thread on the same pooled
        # executor thread does not inherit a dead loop.
        assert getattr(feishu_adapter._ws_isolation_state, "loop", None) is None
        assert (
            getattr(feishu_adapter._ws_isolation_state, "connect_kwargs", None)
            is None
        )

    def test_two_concurrent_clients_each_use_their_own_loop(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)
        mod = _fake_ws_module()
        client_mod = _inject_fake_lark_module(monkeypatch, mod)

        results = {}
        barrier = threading.Barrier(2)

        class FakeClient:
            def __init__(self, name):
                self._name = name

            def start(self):
                barrier.wait(timeout=10)

                async def probe():
                    await asyncio.sleep(0.02)
                    return id(asyncio.get_running_loop())

                results[self._name] = client_mod.loop.run_until_complete(probe())

        def run(name):
            feishu_adapter._run_official_feishu_ws_client(
                FakeClient(name), _adapter_stub()
            )

        threads = [threading.Thread(target=run, args=(f"p{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive()

        assert results["p0"] != results["p1"]

    def test_legacy_fallback_when_install_fails(self, monkeypatch):
        monkeypatch.setattr(feishu_adapter, "_WS_ISOLATION_INSTALLED", False)

        def broken_install(module):
            raise RuntimeError("simulated SDK layout change")

        monkeypatch.setattr(
            feishu_adapter, "_install_lark_ws_isolation", broken_install
        )

        real_connect = MagicMock(name="real-connect")
        mod = _fake_ws_module(connect=real_connect)
        client_mod = _inject_fake_lark_module(monkeypatch, mod)

        seen = {}

        class FakeClient:
            def start(self):
                seen["loop_global"] = client_mod.loop
                seen["connect_during"] = client_mod.websockets.connect
                # Legacy path: the module global is the real loop directly.
                client_mod.loop.run_until_complete(asyncio.sleep(0))

        adapter = _adapter_stub(_ws_ping_interval=30)
        feishu_adapter._run_official_feishu_ws_client(FakeClient(), adapter)

        assert isinstance(seen["loop_global"], asyncio.AbstractEventLoop)
        # The legacy connect wrapper was active during start() ...
        assert seen["connect_during"] is not real_connect
        # ... and restored afterwards.
        assert client_mod.websockets.connect is real_connect
        assert adapter._ws_thread_loop is None


class TestSupervisedRestart:
    def _supervisor_stub(self):
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
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(None)  # new thread dies immediately too
            stub._ws_future = fut

        stub._connect_websocket = _connect_websocket
        return stub

    def test_restarts_a_dead_ws_thread(self):
        async def scenario():
            stub = self._supervisor_stub()
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(None)  # the WS "thread" is already dead
            stub._ws_future = fut

            task = asyncio.ensure_future(
                feishu_adapter.FeishuAdapter._supervise_websocket_thread(stub)
            )
            # Give the supervisor a few backoff cycles (0.01s each).
            for _ in range(200):
                await asyncio.sleep(0.01)
                if stub.connect_calls >= 2:
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return stub.connect_calls

        calls = asyncio.run(scenario())
        assert calls >= 2, f"supervisor should keep restarting, got {calls} restarts"

    def test_stops_when_disconnect_nils_the_client(self):
        async def scenario():
            stub = self._supervisor_stub()
            loop = asyncio.get_running_loop()
            fut = loop.create_future()  # not resolved yet: thread "alive"
            stub._ws_future = fut

            task = asyncio.ensure_future(
                feishu_adapter.FeishuAdapter._supervise_websocket_thread(stub)
            )
            await asyncio.sleep(0.01)  # supervisor blocks awaiting the future
            stub._ws_client = None  # deliberate disconnect ...
            fut.set_result(None)  # ... then the thread exits
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except asyncio.CancelledError:
                pass
            return stub, task

        stub, task = asyncio.run(scenario())
        assert task.done()
        assert stub.connect_calls == 0

    def test_restart_failure_backs_off_without_hot_loop(self):
        async def scenario():
            stub = self._supervisor_stub()
            stub.connect_should_fail = 1
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(None)
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

        calls = asyncio.run(scenario())
        # First restart fails, second succeeds: exactly the expected shape.
        assert calls == 2, f"expected failed-then-successful restarts, got {calls}"
