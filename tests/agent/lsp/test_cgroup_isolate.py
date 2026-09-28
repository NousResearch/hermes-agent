"""cgroup v2 isolation for spawned language servers.

Coverage:

- ``attach()`` against a fake cgroup root (``HERMES_LSP_CGROUP_ROOT`` -> tmp
  dir): pid written to ``cgroup.procs``, ``memory.max`` set from the MB cap,
  cap of 0 skipped, failed cap write still counts as isolated;
- fallback: unavailable root returns False and warns exactly once;
- config defaults + service status surfacing + client spawn wiring;
- real-cgroup integration (spawned server lands in a different cgroup than
  the gateway; a memory spike inside the capped cgroup is OOM-killed while
  the caller survives) — auto-skipped where the cgroup mount is read-only
  (e.g. Docker's default), which is exactly the documented fallback case.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.lsp import cgroup
from agent.lsp.client import LSPClient
from agent.lsp.manager import LSPService
from hermes_cli.config_defaults import DEFAULT_CONFIG


@pytest.fixture(autouse=True)
def _reset_warning():
    cgroup.reset_warning_for_tests()
    yield
    cgroup.reset_warning_for_tests()


@pytest.fixture()
def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """Point the module at a temp 'cgroup root' so plain file writes stand in
    for the kernel interface (attach logic runs unmodified)."""
    monkeypatch.setenv(cgroup.CGROUP_ROOT_ENV, str(tmp_path))
    return tmp_path


# ---- attach() against the fake root ----


def test_attach_writes_pid_and_memory_cap(fake_root: Path) -> None:
    assert cgroup.attach(1234, 512) is True
    cell = fake_root / cgroup.DIR_NAME
    assert (cell / "cgroup.procs").read_text(encoding="utf-8") == "1234"
    assert (cell / "memory.max").read_text(encoding="utf-8") == str(512 * 1024 * 1024)


def test_attach_zero_cap_sets_no_memory_max(fake_root: Path) -> None:
    assert cgroup.attach(42, 0) is True
    cell = fake_root / cgroup.DIR_NAME
    assert (cell / "cgroup.procs").read_text(encoding="utf-8") == "42"
    assert not (cell / "memory.max").exists()


def test_cap_write_failure_keeps_isolation(fake_root: Path) -> None:
    """memory.max unavailable (no memory controller) -> separation still holds."""
    (fake_root / cgroup.DIR_NAME).mkdir()
    (fake_root / cgroup.DIR_NAME / "memory.max").mkdir()  # write -> EISDIR
    assert cgroup.attach(77, 256) is True
    assert (fake_root / cgroup.DIR_NAME / "cgroup.procs").read_text(encoding="utf-8") == "77"


def test_unavailable_root_falls_back_with_single_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Missing/unwritable cgroup root -> False (server shares gateway cgroup),
    one warning per process, repeats silent."""
    monkeypatch.setenv(cgroup.CGROUP_ROOT_ENV, str(tmp_path / "does-not-exist"))
    with caplog.at_level("WARNING", logger="agent.lsp.cgroup"):
        assert cgroup.attach(1, 64) is False
        assert cgroup.attach(2, 64) is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "cgroup isolation unavailable" in warnings[0].getMessage()


# ---- config + service wiring ----


def test_config_defaults() -> None:
    assert DEFAULT_CONFIG["lsp"]["cgroup_isolate"] is True
    assert DEFAULT_CONFIG["lsp"]["cgroup_memory_mb"] == 4096


def test_service_status_reports_cgroup_settings() -> None:
    svc = LSPService(
        enabled=False, wait_mode="document", wait_timeout=5.0, install_strategy="manual",
        cgroup_isolate=True, cgroup_memory_mb=128,
    )
    status = svc.get_status()
    assert status["cgroup_isolate"] is True
    assert status["cgroup_memory_mb"] == 128
    assert status["max_heap_mb"] == 2048  # heap cap independent of cgroup cap


# ---- client spawn path ----


@pytest.mark.asyncio
async def test_client_spawn_attaches_and_reports(fake_root: Path) -> None:
    client = LSPClient(
        server_id="fake-ls", workspace_root=str(fake_root), command=["sleep", "30"],
        isolate_cgroup=True, cgroup_memory_mb=64,
    )
    await client._spawn()
    try:
        assert client.cgroup_isolated is True
        procs = (fake_root / cgroup.DIR_NAME / "cgroup.procs").read_text(encoding="utf-8")
        assert procs == str(client._proc.pid)
        assert client._proc.returncode is None
    finally:
        client._proc.terminate()
        await asyncio.wait_for(client._proc.wait(), timeout=5)
        for task in (client._stderr_task, client._reader_task):
            if task is not None:
                task.cancel()


@pytest.mark.asyncio
async def test_client_spawn_without_isolation_still_runs(fake_root: Path) -> None:
    """isolate_cgroup=False: no attach attempt, process spawns normally."""
    client = LSPClient(
        server_id="fake-ls", workspace_root=str(fake_root), command=["sleep", "30"],
        isolate_cgroup=False,
    )
    await client._spawn()
    try:
        assert client.cgroup_isolated is False
        assert not (fake_root / cgroup.DIR_NAME).exists()
        assert client._proc.returncode is None
    finally:
        client._proc.terminate()
        await asyncio.wait_for(client._proc.wait(), timeout=5)
        for task in (client._stderr_task, client._reader_task):
            if task is not None:
                task.cancel()


# ---- real cgroup v2 (skipped on read-only/restricted mounts = fallback env) ----


def _real_root_writable() -> bool:
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").exists():
        return False
    probe = root / "hermes-lsp-probe"
    try:
        probe.mkdir(exist_ok=True)
        shutil.rmtree(probe)
        return True
    except OSError:
        return False


REAL_CGROUP = _real_root_writable()
_NEEDS_REAL = pytest.mark.skipif(
    not REAL_CGROUP, reason="cgroup v2 root not writable (restricted container -> fallback path)"
)


def _cgroup_of(pid: int) -> str:
    return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip()


@_NEEDS_REAL
def test_real_spawn_lands_in_different_cgroup_than_gateway() -> None:
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert cgroup.attach(proc.pid, 256) is True
        child = _cgroup_of(proc.pid)
        gateway = _cgroup_of(os.getpid())
        assert f"/{cgroup.DIR_NAME}" in child, f"child not isolated: {child}"
        assert child != gateway, f"child shares gateway cgroup: {child}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@_NEEDS_REAL
def test_real_memory_spike_killed_in_capped_cgroup_caller_survives() -> None:
    """Acceptance: an LSP-style memory spike inside the capped cgroup gets
    OOM-killed at memory.max; the caller (gateway stand-in) keeps running."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5); bytearray(512 * 1024 * 1024)"]
    )
    try:
        assert cgroup.attach(proc.pid, 64) is True
        cap = Path("/sys/fs/cgroup") / cgroup.DIR_NAME / "memory.max"
        if cap.read_text(encoding="utf-8").strip() == "max":
            pytest.skip("memory controller unavailable; separation-only mode")
        rc = proc.wait(timeout=60)
        assert rc == -9, f"spike process exited {rc}, expected SIGKILL at the cap"
        # Gateway stand-in: this process is untouched by the child's OOM.
        assert os.getpid() > 0 and _cgroup_of(os.getpid())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
