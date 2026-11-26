"""Sidecar lifecycle tests: orphan reaping and parent-death wiring.

A hard gateway exit used to leave the detached Node sidecar squatting the
loopback port with a token the next gateway run doesn't know — every
replacement spawn then died on EADDRINUSE. These tests cover the startup
reaper (`_reap_stale_sidecar`) and the stdin-pipe lifetime binding, without
spawning Node or binding ports.
"""
from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import sidecar_paths
from plugins.platforms.photon.adapter import PhotonAdapter
from tests.waiting import async_wait_until


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
    monkeypatch: pytest.MonkeyPatch, tmp_path,
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
    monkeypatch.setattr(sidecar_paths, "_SIDECAR_DIR", tmp_path)  # the install lock lives in the sidecar dir
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


def _import_photon_for_another_profile(monkeypatch: pytest.MonkeyPatch, name: str):
    """Import the plugin package a second time under its own module name, as the plugin loader does for
    every served profile after the first (``hermes_plugins.<slug>__home_<digest>``)."""
    plugin_dir = Path(photon_adapter.__file__).parent
    spec = importlib.util.spec_from_file_location(name, plugin_dir / "__init__.py",
                                                  submodule_search_locations=[str(plugin_dir)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    loaded = {key: sys.modules.pop(key) for key in list(sys.modules) if key.startswith(f"{name}.")}
    for key, module in loaded.items():  # recorded as new keys, so teardown removes them again
        monkeypatch.setitem(sys.modules, key, module)
    return loaded[f"{name}.adapter"], loaded[f"{name}.sidecar_paths"]


@pytest.mark.asyncio
async def test_concurrent_profile_connects_install_shared_sidecar_deps_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """Multiplexed profiles connect concurrently, each through its own import of this plugin, and they all
    run the one sidecar dir while ``npm ci`` removes node_modules before installing: a connect that starts
    while another profile's install is running waits for it, even once that install has created
    spectrum-ts, instead of starting a second install or spawning against the half-written tree."""
    other_adapter, other_paths = _import_photon_for_another_profile(monkeypatch, "photon_second_profile_import")
    state = {"installed": False, "running": 0, "overlap": 0, "installs": 0}
    guard, second_reached, install_running_at_second_check = threading.Lock(), threading.Event(), []

    def reinstall() -> None:
        with guard:
            state["running"] += 1
            state["installs"] += 1
            state["overlap"] = max(state["overlap"], state["running"])
            state["installed"] = True  # npm creates node_modules/spectrum-ts before the install completes
        second_reached.wait(5)  # finish once the second connect has started looking at the sidecar dir
        with guard:
            state["running"] -= 1

    def installed_seen_by(module) -> bool:
        if module is other_adapter:
            with guard:  # recorded before the installer is released, so the result cannot race it
                install_running_at_second_check.append(state["running"] > 0)
            second_reached.set()
        return state["installed"]

    def second_sidecar_dir() -> Path:
        second_reached.set()
        return tmp_path

    for module, paths in ((photon_adapter, sidecar_paths), (other_adapter, other_paths)):
        monkeypatch.setattr(paths, "_SIDECAR_DIR", tmp_path)
        monkeypatch.setattr(module, "sidecar_deps_installed", lambda module=module: installed_seen_by(module))
        monkeypatch.setattr(module, "_sidecar_deps_stale", lambda: False)
        monkeypatch.setattr(module, "_reinstall_sidecar_deps", reinstall)
        monkeypatch.setattr(module, "find_node_executable", lambda _name: "/pm/node")
    monkeypatch.setattr(other_adapter, "_sidecar_dir", second_sidecar_dir)
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    first, second = (module.PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))
                     for module in (photon_adapter, other_adapter))

    first_connect = asyncio.ensure_future(first._ensure_sidecar_deps())
    await async_wait_until(lambda: state["running"] == 1, timeout=5, what="the first profile's install to start")
    await asyncio.gather(first_connect, second._ensure_sidecar_deps())

    assert (state["installs"], state["overlap"], install_running_at_second_check) == (1, 1, [False])


@pytest.mark.asyncio
@pytest.mark.parametrize(("writable", "installed", "outcome"), [
    (True, True, ("SIDECAR_LOCK_FAILED", True)),  # another profile may be installing: retry, never go on
    (False, True, None),  # a read-only tree with baked deps starts as it is
    (False, False, ("SIDECAR_DEPS_MISSING", False)),
])
async def test_unopenable_install_lock_never_installs_unlocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path, writable: bool, installed: bool, outcome,
) -> None:
    """Without the install lock a connect cannot know whether another profile is installing, so it never runs
    npm, and in a writable sidecar dir it does not start on deps another profile may be rewriting."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(sidecar_paths, "_SIDECAR_DIR", tmp_path)
    (tmp_path / ".photon-install.lock").mkdir()  # opening it as a file fails
    installs = []
    monkeypatch.setattr(photon_adapter, "_dir_writable", lambda _path: writable)
    monkeypatch.setattr(photon_adapter, "sidecar_deps_installed", lambda: installed)
    monkeypatch.setattr(photon_adapter, "_sidecar_deps_stale", lambda: True)
    monkeypatch.setattr(photon_adapter, "_reinstall_sidecar_deps", lambda: installs.append(1))

    if outcome is None:
        await adapter._ensure_sidecar_deps()
    else:
        with pytest.raises(photon_adapter.PhotonSidecarStartupError) as raised:
            await adapter._ensure_sidecar_deps()
        assert (raised.value.code, raised.value.retryable) == outcome
    assert installs == []
