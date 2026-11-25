"""The gateway shares one npm install with setup surfaces and never runs it on the event loop."""

import asyncio
import threading

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_timed_out_install_worker_does_not_overlap_retry(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp import adapter as whatsapp
    from plugins.platforms.whatsapp.bridge_dependencies import stamp_bridge_dependencies

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "node_modules").mkdir()
    (bridge / "package.json").write_text('{"dependencies":{}}', encoding="utf-8")
    first = whatsapp.WhatsAppAdapter(PlatformConfig(extra={"session_path": str(tmp_path / "s1")}))
    second = whatsapp.WhatsAppAdapter(PlatformConfig(extra={"session_path": str(tmp_path / "s2")}))
    entered, release = threading.Event(), threading.Event()
    calls = []
    def install(_directory):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        stamp_bridge_dependencies(bridge)
        return True
    monkeypatch.setattr(first, "_install_bridge_deps", install)
    monkeypatch.setattr(second, "_install_bridge_deps", install)
    worker = asyncio.create_task(asyncio.to_thread(first._ensure_bridge_deps, bridge))
    assert await asyncio.to_thread(entered.wait, 5)
    worker.cancel()  # gateway connect timeout cancels the coroutine, not the npm thread
    with pytest.raises(asyncio.CancelledError):
        await worker
    retry_started = threading.Event()
    def retry_install():
        retry_started.set()
        return second._ensure_bridge_deps(bridge)
    retry = asyncio.create_task(asyncio.to_thread(retry_install))
    assert await asyncio.to_thread(retry_started.wait, 5)
    assert len(calls) == 1
    release.set()
    assert await retry is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_bridge_dependency_install_runs_off_gateway_loop(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp import adapter as whatsapp

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    instance = whatsapp.WhatsAppAdapter(PlatformConfig(extra={
        "bridge_script": str(bridge / "bridge.js"), "session_path": str(tmp_path / "session"),
    }))
    monkeypatch.setattr(whatsapp, "find_node_executable", lambda name: "node")
    monkeypatch.setattr(instance, "_preflight", lambda: True)
    monkeypatch.setattr(instance, "_acquire_platform_lock", lambda *args: True)
    monkeypatch.setattr(instance, "_release_platform_lock", lambda: None)
    on_loop = threading.get_ident()
    called_from = []

    def install(_directory):
        called_from.append(threading.get_ident())
        return False

    monkeypatch.setattr(instance, "_ensure_bridge_deps", install)
    assert await instance.connect() is False
    assert len(called_from) == 1
    assert called_from[0] != on_loop
