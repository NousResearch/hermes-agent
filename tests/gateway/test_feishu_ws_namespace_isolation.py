"""Real adapter imports must share one SDK installation across plugin namespaces."""
import asyncio
import importlib.util
import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from plugins.platforms.feishu import adapter as first


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
def test_second_namespace_preserves_shutdown_owner(monkeypatch, caplog, running, concurrent):
    name = "plugins.platforms.feishu._namespace_probe"
    spec = importlib.util.spec_from_file_location(name, first.__file__)
    second = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, second)
    spec.loader.exec_module(second)


    class Client:
        async def _receive_message_loop(self):
            raise ConnectionError("deliberate disconnect")

        def start(self):
            async def parked():
                await asyncio.Event().wait()
            sdk.loop.create_task(self._receive_message_loop())
            sdk.loop.run_until_complete(parked())

    sdk = SimpleNamespace(loop=SimpleNamespace(), websockets=SimpleNamespace(connect=lambda: None), Client=Client)
    if concurrent:
        from concurrent.futures import ThreadPoolExecutor
        barrier = threading.Barrier(2)

        def install(module):
            barrier.wait(timeout=10)
            module._install_lark_ws_isolation(sdk)
            return (sdk.loop, sdk.websockets.connect, sdk.Client._receive_message_loop)

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(install, first), pool.submit(install, second)
            assert a.result(timeout=15) == b.result(timeout=15)
    else:
        first._install_lark_ws_isolation(sdk)
    installed = (sdk.loop, sdk.websockets.connect, sdk.Client._receive_message_loop)
    second._install_lark_ws_isolation(sdk)
    assert installed == (sdk.loop, sdk.websockets.connect, sdk.Client._receive_message_loop)
    import types
    lark = types.ModuleType("lark_oapi")
    ws = types.ModuleType("lark_oapi.ws")
    lark.ws = ws
    ws.client = sdk
    monkeypatch.setitem(sys.modules, "lark_oapi", lark)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", ws)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", sdk)
    stub = SimpleNamespace(_loop=None, _ws_thread_loop=None, _ws_reconnect_nonce=None,
                           _ws_reconnect_interval=None, _ws_ping_interval=None,
                           _ws_ping_timeout=None, _running=running)
    caplog.set_level(logging.DEBUG)
    cleanup = []

    def run():
        second._run_official_feishu_ws_client(Client(), stub)
        cleanup.append(tuple(getattr(second._ws_isolation_state, key, None)
                             for key in ("loop", "adapter", "connect_kwargs", "on_link_up")))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and "receive loop" in r.getMessage()]
    assert len(errors) == int(running), "receive-loop severity must follow the actual owner"
    assert first._ws_isolation_state is second._ws_isolation_state
    assert cleanup == [(None, None, None, None)]
