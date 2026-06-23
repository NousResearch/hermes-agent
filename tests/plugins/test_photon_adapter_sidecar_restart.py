from typing import Any, cast

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


@pytest.mark.asyncio
async def test_ensure_sidecar_running_restarts_exited_sidecar(monkeypatch):
    adapter = PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={"project_id": "pid", "project_secret": "secret"},
        )
    )
    adapter._sidecar_proc = cast(Any, _FakeProc(returncode=75))
    adapter._sidecar_supervisor_task = None
    calls = []

    async def fake_start_sidecar():
        calls.append("start")
        adapter._sidecar_proc = cast(Any, _FakeProc(returncode=None))

    monkeypatch.setattr(adapter, "_start_sidecar", fake_start_sidecar)

    await adapter._ensure_sidecar_running()

    assert calls == ["start"]
    assert adapter._sidecar_proc is not None
    assert adapter._sidecar_proc.poll() is None


@pytest.mark.asyncio
async def test_ensure_sidecar_running_keeps_live_sidecar(monkeypatch):
    adapter = PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={"project_id": "pid", "project_secret": "secret"},
        )
    )
    adapter._sidecar_proc = cast(Any, _FakeProc(returncode=None))

    async def fail_start_sidecar():  # pragma: no cover - should not be called
        raise AssertionError("live sidecar should not restart")

    monkeypatch.setattr(adapter, "_start_sidecar", fail_start_sidecar)

    await adapter._ensure_sidecar_running()

    assert adapter._sidecar_proc is not None
    assert adapter._sidecar_proc.poll() is None
