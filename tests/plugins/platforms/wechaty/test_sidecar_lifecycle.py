"""Sidecar lifecycle tests for WechatyAdapter."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wechaty import adapter as wechaty_adapter
from plugins.platforms.wechaty.adapter import WechatyAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> WechatyAdapter:
    monkeypatch.setenv("WECHATY_PUPPET", "wechaty-puppet-wechat4u")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return WechatyAdapter(cfg)


def _capture_kills(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[int, int]]:
    kills: List[Tuple[int, int]] = []

    def _fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(wechaty_adapter.os, "kill", _fake_kill)
    return kills


@pytest.mark.asyncio
async def test_reap_noop_when_port_free(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [])
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert kills == []


@pytest.mark.asyncio
async def test_reap_kills_verified_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [4242])
    monkeypatch.setattr(adapter, "_pid_is_sidecar", lambda pid: True)
    monkeypatch.setattr(adapter, "_pid_alive", lambda pid: False)
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert kills == [(4242, wechaty_adapter.signal.SIGTERM)]


@pytest.mark.asyncio
async def test_reap_skips_foreign_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [777])
    monkeypatch.setattr(adapter, "_pid_is_sidecar", lambda pid: False)
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert kills == []


@pytest.mark.asyncio
async def test_start_sidecar_spawns_with_stdin_pipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(wechaty_adapter, "_SIDECAR_DIR", tmp_path)

    spawned: Dict[str, Any] = {}

    class _FakeProc:
        pid = 999
        stdout = None
        stdin = None

        @staticmethod
        def poll() -> None:
            return None

    def _fake_popen(cmd: List[str], **kwargs: Any) -> _FakeProc:
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(wechaty_adapter.subprocess, "Popen", _fake_popen)

    class _HealthyClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> "_HealthyClient":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(self, *a: Any, **k: Any) -> Any:
            class _Resp:
                status_code = 200

            return _Resp()

    monkeypatch.setattr(wechaty_adapter.httpx, "AsyncClient", _HealthyClient)

    await adapter._start_sidecar()

    kwargs = spawned["kwargs"]
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["env"]["WECHATY_SIDECAR_WATCH_STDIN"] == "1"
    assert kwargs["env"]["WECHATY_SIDECAR_TOKEN"] == adapter._sidecar_token
