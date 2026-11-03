"""Sidecar lifecycle tests: orphan reaping and parent-death wiring.

A hard gateway exit used to leave the detached Node sidecar squatting the
loopback port with a token the next gateway run doesn't know — every
replacement spawn then died on EADDRINUSE. These tests cover the startup
reaper (`_reap_stale_sidecar`) and the stdin-pipe lifetime binding, without
spawning Node or binding ports.
"""
from __future__ import annotations

import subprocess
import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import sidecar_paths
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    # Stand-in for PM's node; the user's PATH node is never picked up.
    monkeypatch.setattr(photon_adapter, "find_node_executable", lambda _name: "/pm/node")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


class _ProbeClient:
    """Fake httpx.AsyncClient whose /healthz probe behavior is injectable."""

    connects = True

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self) -> _ProbeClient:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    async def post(self, *a: Any, **k: Any) -> Any:
        if not self.connects:
            raise photon_adapter.httpx.ConnectError("connection refused")

        class _Resp:
            status_code = 401  # orphan with a different token

        return _Resp()


def _capture_kills(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    kills: list[tuple[int, int]] = []

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
    monkeypatch.setattr(sidecar_paths, "_SIDECAR_DIR", tmp_path)

    spawned: dict[str, Any] = {}
    hidden_flags = 0x08000000
    monkeypatch.setattr(
        "hermes_cli._subprocess_compat.windows_hide_flags",
        lambda: hidden_flags,
    )

    class _PatchResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd: list[str], **kwargs: Any) -> _PatchResult:
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

    def _fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
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


@pytest.mark.asyncio
async def test_spectrum_patch_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node patch run must not block the shared gateway event loop.

    ``_start_sidecar`` spawns the Spectrum patch script and *waits* for it
    (``timeout=10``). Run inline it holds the loop for that whole window, so
    every other platform's traffic stalls — and ``_start_sidecar`` runs on
    every reconnect (``connect(is_reconnect=True)``), not just startup, so the
    stall recurs on a live gateway. The dep reinstall a few lines above already
    hops to a thread for exactly this reason; the patch run must too.
    """
    import threading

    adapter = _make_adapter(monkeypatch)
    main_thread = threading.current_thread()
    seen: dict[str, Any] = {}

    # node_modules present + deps fresh, so we reach the patch run.
    monkeypatch.setattr(photon_adapter.Path, "exists", lambda self: True)
    monkeypatch.setattr(photon_adapter, "_sidecar_deps_stale", lambda: False)

    async def _no_reap() -> None:
        return None

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)

    def _fake_run(*a: Any, **k: Any) -> Any:
        seen["thread"] = threading.current_thread()

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Done()

    monkeypatch.setattr(photon_adapter.subprocess, "run", _fake_run)

    class _FakeProc:
        pid = 4242
        stdin = None
        stdout = None

        def poll(self) -> int:
            # Report "exited" so the readiness health-poll loop bails out
            # immediately instead of spinning for its full 15s deadline —
            # the assertion below only cares where the patch run executed.
            return 0

    monkeypatch.setattr(
        photon_adapter.subprocess, "Popen", lambda *a, **k: _FakeProc()
    )

    try:
        await adapter._start_sidecar()
    except Exception:
        # Readiness/handshake past the patch run may fail under the fakes —
        # irrelevant here; we only assert where the patch run executed.
        pass

    assert seen.get("thread") is not None, "patch run never executed"
    assert seen["thread"] is not main_thread, (
        "Spectrum patch subprocess ran on the event-loop thread; it must be "
        "dispatched via asyncio.to_thread so a 10s node spawn can't freeze "
        "every other platform on the gateway loop"
    )


# -- Windows reaping -------------------------------------------------------------
# Windows keeps its own path for every probe: no lsof, no ps, no SIGTERM/SIGKILL,
# and os.kill(pid, 0) signals a console group instead of probing. A fast gateway
# restart there still races the sidecar's port release, so the reaper must find
# and kill the stale sidecar rather than return early.


def _fake_psutil(monkeypatch: pytest.MonkeyPatch, **attrs: Any) -> None:
    """Install a stand-in psutil for the function-local ``import psutil``."""
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(**attrs))


def test_find_listener_pids_reads_the_windows_socket_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")

    def _conn(port: int, status: str, pid: int | None) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            status=status, pid=pid, laddr=types.SimpleNamespace(port=port)
        )

    _fake_psutil(
        monkeypatch,
        CONN_LISTEN="LISTEN",
        net_connections=lambda kind: [
            _conn(8789, "LISTEN", 4242),
            _conn(8789, "ESTABLISHED", 5151),  # a client, not the listener
            _conn(9999, "LISTEN", 8181),  # another port entirely
            _conn(8789, "LISTEN", None),  # unowned socket
        ],
    )

    assert PhotonAdapter._find_listener_pids(8789) == [4242]


def test_find_listener_pids_falls_back_to_netstat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A psutil that can't read the table (access denied) must not blind the reaper."""

    def _denied(kind: str) -> Any:
        raise PermissionError("access denied")

    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")
    _fake_psutil(monkeypatch, net_connections=_denied)

    netstat = "\r\n".join([
        "Active Connections",
        "  Proto  Local Address          Foreign Address        State           PID",
        "  TCP    127.0.0.1:8789         0.0.0.0:0              LISTENING       4242",
        "  TCP    127.0.0.1:8789         127.0.0.1:5555         ESTABLISHED     5151",
        "  TCP    0.0.0.0:8789           0.0.0.0:0              LISTENING       4243",
        "  TCP    127.0.0.1:9999         0.0.0.0:0              LISTENING       8181",
    ])
    monkeypatch.setattr(PhotonAdapter, "_quick_stdout", staticmethod(lambda cmd: netstat))

    assert PhotonAdapter._find_listener_pids(8789) == [4242, 4243]


def test_pid_is_sidecar_matches_a_windows_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")
    cmdline = [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\hermes\plugins\platforms\photon\sidecar\index.mjs",
    ]
    _fake_psutil(monkeypatch, Process=lambda pid: types.SimpleNamespace(cmdline=lambda: cmdline))
    assert PhotonAdapter._pid_is_sidecar(4242) is True

    other = [r"C:\Program Files\nodejs\node.exe", r"C:\apps\thing\server.mjs"]
    _fake_psutil(monkeypatch, Process=lambda pid: types.SimpleNamespace(cmdline=lambda: other))
    assert PhotonAdapter._pid_is_sidecar(4243) is False


def test_pid_alive_never_signals_the_pid_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.kill(pid, 0) raises CTRL_C_EVENT at the whole console group on Windows."""
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")
    monkeypatch.setattr(photon_adapter.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(photon_adapter, "_sidecar_pid_alive", lambda pid: True)

    assert PhotonAdapter._pid_alive(4242) is True
    assert signals == []


@pytest.mark.asyncio
async def test_reap_kills_stale_sidecar_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression: win32 used to return before reaping, so a restart race
    left the old sidecar on the port and the new one died on EADDRINUSE."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)  # port held, wrong token
    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")
    monkeypatch.setattr(PhotonAdapter, "_find_listener_pids", classmethod(lambda cls, port: [4242]))
    monkeypatch.setattr(PhotonAdapter, "_pid_is_sidecar", classmethod(lambda cls, pid: True))

    alive = {"value": True}
    monkeypatch.setattr(PhotonAdapter, "_pid_alive", staticmethod(lambda pid: alive["value"]))

    calls: list[tuple[str, int]] = []

    class _Proc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def terminate(self) -> None:
            calls.append(("terminate", self.pid))
            alive["value"] = False

        def kill(self) -> None:
            calls.append(("kill", self.pid))

    signals: list[tuple[int, int]] = []
    _fake_psutil(monkeypatch, Process=_Proc)
    monkeypatch.setattr(photon_adapter.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    await adapter._reap_stale_sidecar()

    assert calls == [("terminate", 4242)], "the Windows reaper must terminate the stale sidecar"
    assert signals == [], "SIGTERM/SIGKILL must never be sent on Windows"


@pytest.mark.asyncio
async def test_reap_escalates_to_a_hard_kill_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """signal.SIGKILL does not exist on Windows, so the escalation must not name it."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _ProbeClient)
    monkeypatch.setattr(photon_adapter.sys, "platform", "win32")
    monkeypatch.setattr(PhotonAdapter, "_find_listener_pids", classmethod(lambda cls, port: [4242]))
    monkeypatch.setattr(PhotonAdapter, "_pid_is_sidecar", classmethod(lambda cls, pid: True))
    monkeypatch.setattr(PhotonAdapter, "_pid_alive", staticmethod(lambda pid: True))  # ignores terminate()

    calls: list[tuple[str, int]] = []

    class _Proc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def terminate(self) -> None:
            calls.append(("terminate", self.pid))

        def kill(self) -> None:
            calls.append(("kill", self.pid))

    _fake_psutil(monkeypatch, Process=_Proc)
    # Skip the 3s liveness wait: the deadline is already past on the first check.
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(photon_adapter, "time", types.SimpleNamespace(time=lambda: next(clock, 10.0)))

    await adapter._reap_stale_sidecar()

    assert calls == [("terminate", 4242), ("kill", 4242)]

