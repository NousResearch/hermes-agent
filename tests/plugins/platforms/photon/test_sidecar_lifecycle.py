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


def _make_local_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    # Local mode must not leak stale cloud credentials into the child.
    monkeypatch.setenv("PHOTON_PROJECT_ID", "stale-cloud-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "stale-cloud-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={"imessage_mode": "local"})
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
async def test_start_sidecar_spawns_with_stdin_pipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The spawn must hold a stdin pipe and enable the sidecar's EOF watch."""
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    (tmp_path / "node_modules" / "spectrum-ts").mkdir(parents=True)
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    spawned: Dict[str, Any] = {}
    hidden_flags = 0x08000000
    monkeypatch.setattr(
        "hermes_cli._subprocess_compat.windows_hide_flags",
        lambda: hidden_flags,
    )

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
    assert kwargs["creationflags"] == hidden_flags


@pytest.mark.asyncio
async def test_start_sidecar_local_mode_omits_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    adapter = _make_local_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    (tmp_path / "node_modules" / "spectrum-ts").mkdir(parents=True)
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    spawned: Dict[str, Any] = {}

    class _FakeProc:
        pid = 999
        stdout = None
        stdin = None

        @staticmethod
        def poll() -> None:
            return None

    def _fake_popen(cmd: List[str], **kwargs: Any) -> _FakeProc:
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

    env = spawned["kwargs"]["env"]
    assert env["PHOTON_IMESSAGE_MODE"] == "local"
    assert "PHOTON_PROJECT_ID" not in env
    assert "PHOTON_PROJECT_SECRET" not in env


@pytest.mark.asyncio
async def test_start_sidecar_cold_installs_missing_deps(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Missing dependencies are installed into the resolved writable sidecar."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    installs: List[str] = []

    def _fake_install() -> None:
        installs.append("ran")
        (tmp_path / "node_modules" / "spectrum-ts").mkdir(parents=True)

    monkeypatch.setattr(photon_adapter, "_reinstall_sidecar_deps", _fake_install)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)

    class _FakeProc:
        pid = 999
        stdout = None
        stdin = None

        @staticmethod
        def poll() -> None:
            return None
    monkeypatch.setattr(
        photon_adapter.subprocess, "Popen", lambda *a, **k: _FakeProc()
    )

    class _HealthyClient(_ProbeClient):
        async def post(self, *a: Any, **k: Any) -> Any:
            class _Resp:
                status_code = 200

            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _HealthyClient)

    await adapter._start_sidecar()

    assert installs == ["ran"]


@pytest.mark.asyncio
async def test_start_sidecar_reinstalls_empty_node_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A partial/aborted npm install leaves an empty node_modules/ behind.

    _start_sidecar() must treat it as not-installed the same way
    check_requirements() does (both go through sidecar_deps_installed())
    instead of spawning a sidecar that immediately crashes on a missing
    spectrum-ts module. With the NS-606 cold-install path, "treat as
    not-installed" means: attempt a reinstall, and raise the actionable
    error if that still doesn't produce spectrum-ts.
    """
    adapter = _make_adapter(monkeypatch)
    (tmp_path / "node_modules").mkdir()  # empty — spectrum-ts absent
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    installs: List[str] = []
    monkeypatch.setattr(
        photon_adapter, "_reinstall_sidecar_deps", lambda: installs.append("ran")
    )

    with pytest.raises(RuntimeError, match="could not be installed"):
        await adapter._start_sidecar()

    assert installs == ["ran"]


@pytest.mark.asyncio
async def test_start_sidecar_raises_when_cold_install_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """If the bootstrap install can't produce node_modules, fail with the
    actionable error (surfaced as SIDECAR_FAILED by connect())."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(photon_adapter, "_reinstall_sidecar_deps", lambda: None)

    with pytest.raises(RuntimeError, match="could not be installed"):
        await adapter._start_sidecar()


@pytest.mark.asyncio
async def test_reap_inspects_listeners_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listener inspection must not block the shared gateway event loop.

    ``_find_listener_pids`` shells out to ``lsof`` (timeout=5s) and
    ``_pid_is_sidecar`` runs a ``ps`` per candidate pid (timeout=5s each), so
    inline this holds the loop for 5 + 5·N seconds. ``_reap_stale_sidecar`` is
    awaited from ``_start_sidecar``, which runs on every reconnect — exactly
    when a crashed gateway left an orphan — so the stall lands on a live
    gateway still serving every other platform.
    """
    import threading

    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.sys, "platform", "linux")
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)

    main_thread = threading.current_thread()
    seen: Dict[str, Any] = {}

    def _fake_find(port: int) -> List[int]:
        seen["find"] = threading.current_thread()
        return [4242]

    def _fake_is_sidecar(pid: int) -> bool:
        seen["ps"] = threading.current_thread()
        return True

    monkeypatch.setattr(adapter, "_find_listener_pids", _fake_find)
    monkeypatch.setattr(adapter, "_pid_is_sidecar", _fake_is_sidecar)
    monkeypatch.setattr(adapter, "_pid_alive", lambda pid: False)
    _capture_kills(monkeypatch)

    await adapter._reap_stale_sidecar()

    assert seen.get("find") is not None, "lsof lookup never ran"
    assert seen.get("ps") is not None, "ps check never ran"
    for label in ("find", "ps"):
        assert seen[label] is not main_thread, (
            f"{label} ran on the event-loop thread; the listener inspection "
            "must be dispatched via asyncio.to_thread so a 5s lsof/ps spawn "
            "can't freeze every other platform on the gateway loop"
        )
