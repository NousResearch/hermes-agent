"""Sidecar lifecycle tests: orphan reaping and parent-death wiring.

A hard gateway exit used to leave the detached Node sidecar squatting the
loopback port with a token the next gateway run doesn't know — every
replacement spawn then died on EADDRINUSE. These tests cover the startup
reaper (`_reap_stale_sidecar`) and the stdin-pipe lifetime binding, without
spawning Node or binding ports.
"""
from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
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


@pytest.fixture()
def isolated_hermes_home(tmp_path: Path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


class TestSidecarTokenFile:
    def test_write_then_read_round_trip(self, isolated_hermes_home: Path) -> None:
        photon_adapter._write_sidecar_token_file(9001, "secret-token", 12345)

        path = isolated_hermes_home / "runtime" / "photon-sidecar-9001.json"
        assert path.exists()
        assert photon_adapter._read_sidecar_token_file(9001) == "secret-token"
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "port": 9001,
            "token": "secret-token",
            "pid": 12345,
        }
        if hasattr(stat, "S_IMODE"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_read_ignores_wrong_port(self, isolated_hermes_home: Path) -> None:
        path = isolated_hermes_home / "runtime" / "photon-sidecar-9002.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"port": 9003, "token": "wrong-port", "pid": 1}),
            encoding="utf-8",
        )

        assert photon_adapter._read_sidecar_token_file(9002) is None

    def test_read_missing_or_corrupt_file_returns_none(self, isolated_hermes_home: Path) -> None:
        assert photon_adapter._read_sidecar_token_file(9004) is None

        path = isolated_hermes_home / "runtime" / "photon-sidecar-9004.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf-8")

        assert photon_adapter._read_sidecar_token_file(9004) is None

    def test_remove_only_deletes_matching_token(self, isolated_hermes_home: Path) -> None:
        photon_adapter._write_sidecar_token_file(9005, "newer-token", 222)
        path = isolated_hermes_home / "runtime" / "photon-sidecar-9005.json"

        photon_adapter._remove_sidecar_token_file(9005, "older-token")
        assert path.exists()
        assert photon_adapter._read_sidecar_token_file(9005) == "newer-token"

        photon_adapter._remove_sidecar_token_file(9005, "newer-token")
        assert not path.exists()


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_hermes_home: Path
) -> None:
    """The spawn must hold a stdin pipe and enable the sidecar's EOF watch."""
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    (tmp_path / "node_modules").mkdir()
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
    assert photon_adapter._read_sidecar_token_file(adapter._sidecar_port) == adapter._sidecar_token
