import asyncio
import threading
from types import SimpleNamespace

from gateway.platforms.feishu import (
    _FEISHU_WS_THREAD_STATE,
    _FeishuWSLoopProxy,
    _feishu_ws_connect_proxy,
    _install_feishu_ws_runtime_proxies,
)


def test_feishu_ws_loop_proxy_uses_thread_local_loops():
    proxy = _FeishuWSLoopProxy()
    results = {}

    def worker(name: str) -> None:
        loop = asyncio.new_event_loop()
        _FEISHU_WS_THREAD_STATE.loop = loop
        try:
            async def current_loop_id() -> int:
                return id(asyncio.get_running_loop())

            results[name] = proxy.run_until_complete(current_loop_id())
        finally:
            delattr(_FEISHU_WS_THREAD_STATE, "loop")
            loop.close()

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results["a"] != results["b"]


def test_install_feishu_ws_runtime_proxies_patches_once():
    original_connect = object()
    original_loop = asyncio.new_event_loop()
    module = SimpleNamespace(
        websockets=SimpleNamespace(connect=original_connect),
        loop=original_loop,
    )

    try:
        _install_feishu_ws_runtime_proxies(module)
        first_loop = module.loop
        first_connect = module.websockets.connect

        _install_feishu_ws_runtime_proxies(module)

        assert module.loop is first_loop
        assert module.websockets.connect is first_connect
        assert module._hermes_ws_original_connect is original_connect
    finally:
        original_loop.close()


def test_feishu_ws_connect_proxy_applies_thread_local_ping_overrides():
    seen = {}

    async def fake_connect(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    loop = asyncio.new_event_loop()
    _FEISHU_WS_THREAD_STATE.loop = loop
    _FEISHU_WS_THREAD_STATE.original_connect = fake_connect
    _FEISHU_WS_THREAD_STATE.ping_interval = 7
    _FEISHU_WS_THREAD_STATE.ping_timeout = 3
    try:
        result = loop.run_until_complete(_feishu_ws_connect_proxy("wss://example.invalid/ws"))
    finally:
        for attr in ("loop", "original_connect", "ping_interval", "ping_timeout"):
            delattr(_FEISHU_WS_THREAD_STATE, attr)
        loop.close()

    assert result == "ok"
    assert seen["args"] == ("wss://example.invalid/ws",)
    assert seen["kwargs"]["ping_interval"] == 7
    assert seen["kwargs"]["ping_timeout"] == 3
