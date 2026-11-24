"""Sidecar runtime-record persistence tests (issue #69960).

The sidecar token is generated at spawn and used to exist only in the
gateway process memory + sidecar child env — so cron/`hermes send`
standalone sends structurally could not authenticate. The adapter now
persists ``<hermes-home>/photon-sidecar.json`` after the sidecar
passes its /healthz readiness check, deletes it on stop/failed-start, and
``_standalone_send`` falls back to it when PHOTON_SIDECAR_TOKEN is unset.
No Node, no ports, no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import sidecar_paths
from plugins.platforms.photon.adapter import PhotonAdapter


@pytest.fixture()
def record_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "photon-sidecar.json"
    monkeypatch.setattr(photon_adapter, "_runtime_record_path", lambda: path)
    return path


def _previous_record(record_path: Path, data: dict) -> Path:
    """A record where earlier releases kept it: ``<hermes-home>/runtime/``."""
    previous = record_path.parent / "runtime" / record_path.name
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text(json.dumps(data), encoding="utf-8")
    return previous


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    # Stand-in for PM's node; the user's PATH node is never picked up.
    monkeypatch.setattr(photon_adapter, "find_node_executable", lambda _name: "/pm/node")
    monkeypatch.delenv("PHOTON_SIDECAR_TOKEN", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- record helpers ----------------------------------------------------------


def test_write_read_delete_roundtrip(record_path: Path) -> None:
    photon_adapter._write_runtime_record(8789, "tok123", 4242)

    assert record_path.exists()
    data = json.loads(record_path.read_text(encoding="utf-8"))
    assert data == {"port": 8789, "token": "tok123", "pid": 4242}
    assert photon_adapter._read_runtime_record() == data

    photon_adapter._delete_runtime_record()
    assert not record_path.exists()
    # Idempotent: deleting a missing record must not raise.
    photon_adapter._delete_runtime_record()
    assert photon_adapter._read_runtime_record() is None


def test_record_lives_at_the_top_of_the_profile_home() -> None:
    from hermes_constants import get_hermes_home

    assert photon_adapter._runtime_record_path() == get_hermes_home() / "photon-sidecar.json"


def test_read_deletes_a_previous_record_unread(record_path: Path) -> None:
    previous = _previous_record(record_path, {"port": 8789, "token": "tok", "pid": 4242})

    assert photon_adapter._read_runtime_record() is None
    assert not previous.exists()
    assert not record_path.exists()


def test_a_previous_symlink_is_removed_and_its_target_kept(record_path: Path) -> None:
    target = record_path.parent / "workspace" / "leak.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    previous = record_path.parent / "runtime" / record_path.name
    previous.parent.mkdir()
    try:
        previous.symlink_to(target)
    except OSError:
        pytest.skip("creating a symlink needs a privilege this host lacks")

    photon_adapter._write_runtime_record(8789, "new", 4242)

    assert not os.path.lexists(previous)
    assert target.read_text(encoding="utf-8") == "{}"
    assert json.loads(record_path.read_text(encoding="utf-8"))["token"] == "new"


def test_write_and_delete_leave_no_previous_record(record_path: Path) -> None:
    previous = _previous_record(record_path, {"port": 1, "token": "old", "pid": 1})

    photon_adapter._write_runtime_record(8789, "new", 4242)
    assert not previous.exists()
    assert json.loads(record_path.read_text(encoding="utf-8"))["token"] == "new"

    _previous_record(record_path, {"port": 1, "token": "old", "pid": 1})
    photon_adapter._delete_runtime_record()
    assert not record_path.exists()
    assert not previous.exists()


@pytest.mark.platforms("posix")  # POSIX permission bits
def test_record_written_with_0600(record_path: Path) -> None:
    photon_adapter._write_runtime_record(8789, "secret", 1)
    mode = stat.S_IMODE(record_path.stat().st_mode)
    assert mode == 0o600


def test_read_tolerates_corrupt_record(record_path: Path) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{not json", encoding="utf-8")
    assert photon_adapter._read_runtime_record() is None


# -- lifecycle: written after healthz success, removed on stop/failure -------


class _HealthzClient:
    """Fake httpx.AsyncClient whose /healthz response is injectable."""

    status_code = 200

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self) -> _HealthzClient:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    async def post(self, *a: Any, **k: Any) -> Any:
        cls = type(self)

        class _Resp:
            status_code = cls.status_code

        return _Resp()


class _FakeProc:
    pid = 4242
    stdin = None
    returncode: int | None = None

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch, adapter: PhotonAdapter, tmp_path: Path
) -> None:
    """Stub everything _start_sidecar touches before the healthz loop."""
    sidecar_dir = tmp_path / "sidecar"
    # sidecar_deps_installed() checks the dependency's own directory, not just
    # node_modules/ (9cf2046081) — mirror a real completed install.
    (sidecar_dir / "node_modules" / "spectrum-ts").mkdir(parents=True)
    monkeypatch.setattr(sidecar_paths, "_SIDECAR_DIR", sidecar_dir)
    monkeypatch.setattr(photon_adapter, "_sidecar_deps_stale", lambda: False)

    async def _no_reap(self: PhotonAdapter) -> None:
        return None

    monkeypatch.setattr(PhotonAdapter, "_reap_stale_sidecar", _no_reap)
    monkeypatch.setattr(
        photon_adapter.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(
        photon_adapter.subprocess, "Popen", lambda *a, **k: _FakeProc()
    )

    async def _no_supervise(self: PhotonAdapter, proc: Any) -> None:
        return None

    monkeypatch.setattr(PhotonAdapter, "_supervise_sidecar", _no_supervise)


@pytest.mark.asyncio
async def test_record_written_after_healthz_success(
    monkeypatch: pytest.MonkeyPatch, record_path: Path, tmp_path: Path
) -> None:
    adapter = _make_adapter(monkeypatch)
    _patch_spawn(monkeypatch, adapter, tmp_path)
    _HealthzClient.status_code = 200
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _HealthzClient)

    await adapter._start_sidecar()

    data = json.loads(record_path.read_text(encoding="utf-8"))
    assert data["port"] == adapter._sidecar_port
    assert data["token"] == adapter._sidecar_token
    assert data["pid"] == 4242

    # Cleanup so the fake supervisor task doesn't leak between tests.
    if adapter._sidecar_supervisor_task is not None:
        adapter._sidecar_supervisor_task.cancel()


@pytest.mark.asyncio
async def test_stop_without_proc_still_clears_record(
    monkeypatch: pytest.MonkeyPatch, record_path: Path
) -> None:
    adapter = _make_adapter(monkeypatch)
    photon_adapter._write_runtime_record(8789, "tok", 4242)
    adapter._sidecar_proc = None

    await adapter._stop_sidecar()

    assert not record_path.exists()


# -- _standalone_send fallback ------------------------------------------------


class _SendClient:
    """Fake httpx.AsyncClient capturing /send calls."""

    calls: list = []

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    async def __aenter__(self) -> _SendClient:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    async def post(self, url: str, json: Any = None, headers: Any = None) -> Any:
        type(self).calls.append((url, json, headers))

        class _Resp:
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict:
                return {"ok": True, "messageId": "m1"}

        return _Resp()


@pytest.mark.asyncio
async def test_standalone_send_consumes_record_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, record_path: Path
) -> None:
    monkeypatch.delenv("PHOTON_SIDECAR_TOKEN", raising=False)
    monkeypatch.delenv("PHOTON_SIDECAR_PORT", raising=False)
    photon_adapter._write_runtime_record(9111, "record-token", os.getpid())
    _SendClient.calls = []
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _SendClient)

    result = await photon_adapter._standalone_send(
        PlatformConfig(enabled=True, token="", extra={}), "+15551234567", "hi"
    )

    assert result == {"success": True, "message_id": "m1"}
    url, _body, headers = _SendClient.calls[0]
    assert ":9111/" in url
    assert headers["X-Hermes-Sidecar-Token"] == "record-token"


