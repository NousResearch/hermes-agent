"""Sidecar lifecycle tests: orphan reaping and parent-death wiring.

A hard gateway exit used to leave the detached Node sidecar squatting the
loopback port with a token the next gateway run doesn't know — every
replacement spawn then died on EADDRINUSE. These tests cover the startup
reaper (`_reap_stale_sidecar`) and the stdin-pipe lifetime binding, without
spawning Node or binding ports.
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


class _ProbeClient:
    """Fake httpx.AsyncClient whose /healthz probe behavior is injectable."""

    connects = True

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self) -> "_ProbeClient":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False

    async def post(self, *a: Any, **k: Any) -> Any:
        if not self.connects:
            raise photon_adapter.httpx.ConnectError("connection refused")

        class _Resp:
            status_code = 401  # orphan with a different token

        return _Resp()


def _capture_kills(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[int, int]]:
    kills: List[Tuple[int, int]] = []

    def _fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(photon_adapter.os, "kill", _fake_kill)
    return kills


@pytest.mark.asyncio
async def test_reap_noop_when_port_free(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)

    class _Refused(_ProbeClient):
        connects = False

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _Refused)
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert kills == []


@pytest.mark.asyncio
async def test_reap_kills_verified_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [4242])
    monkeypatch.setattr(adapter, "_pid_is_sidecar", lambda pid: True)
    # Dies promptly on SIGTERM — no escalation expected.
    monkeypatch.setattr(adapter, "_pid_alive", lambda pid: False)
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert kills == [(4242, photon_adapter.signal.SIGTERM)]


@pytest.mark.asyncio
async def test_reap_escalates_to_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [4242])
    monkeypatch.setattr(adapter, "_pid_is_sidecar", lambda pid: True)
    monkeypatch.setattr(adapter, "_pid_alive", lambda pid: True)  # ignores TERM
    # No clock fakery (logging also calls time.time, which makes a fake clock
    # fragile) — this test rides out the real 3s SIGTERM grace window.
    kills = _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert (4242, photon_adapter.signal.SIGTERM) in kills
    assert (4242, photon_adapter.signal.SIGKILL) in kills


@pytest.mark.asyncio
async def test_reap_raises_for_foreign_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never signal a process whose command line isn't our sidecar."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)
    monkeypatch.setattr(adapter, "_find_listener_pids", lambda port: [777])
    monkeypatch.setattr(adapter, "_pid_is_sidecar", lambda pid: False)
    kills = _capture_kills(monkeypatch)

    with pytest.raises(RuntimeError, match="in use by another process"):
        await adapter._reap_stale_sidecar()

    assert kills == []


@pytest.mark.asyncio
async def test_start_sidecar_spawns_with_stdin_pipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The spawn must hold a stdin pipe and enable the sidecar's EOF watch."""
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    (tmp_path / "node_modules").mkdir()
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    spawned: Dict[str, Any] = {}
    hidden_flags = 0x08000000
    monkeypatch.setattr(
        "hermes_cli._subprocess_compat.windows_hide_flags",
        lambda: hidden_flags,
    )

    class _PatchResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd: List[str], **kwargs: Any) -> _PatchResult:
        spawned["patch_cmd"] = cmd
        spawned["patch_kwargs"] = kwargs
        return _PatchResult()

    monkeypatch.setattr(photon_adapter.subprocess, "run", _fake_run)

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

    monkeypatch.setattr(photon_adapter.subprocess, "Popen", _fake_popen)

    class _HealthyClient(_ProbeClient):
        async def post(self, *a: Any, **k: Any) -> Any:
            class _Resp:
                status_code = 200

            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _HealthyClient)

    await adapter._start_sidecar()

    kwargs = spawned["kwargs"]
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["env"]["PHOTON_SIDECAR_WATCH_STDIN"] == "1"
    assert spawned["patch_kwargs"]["creationflags"] == hidden_flags
    assert kwargs["creationflags"] == hidden_flags
