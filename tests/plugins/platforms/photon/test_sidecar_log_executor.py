"""Regression for #127018 (photon sibling): a quiet sidecar must not hold a default-executor worker.

``_supervise_sidecar`` reads the sidecar's stdout for the sidecar's whole life, and
readline() blocks for as long as the sidecar has nothing to say. On the shared default
executor that parks one worker per served profile, the same starvation as google_chat.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter


class _QuietStdout:
    """A live sidecar's stdout: hands out its lines, then readline() blocks until EOF."""

    def __init__(self, *lines: bytes) -> None:
        self._lines = list(lines)
        self._eof = threading.Event()

    def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        self._eof.wait(30)  # bounded, so a broken run cannot hang the suite
        return b""

    def close(self) -> None:
        self._eof.set()


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.setattr(photon_adapter, "find_node_executable", lambda _name: "/pm/node")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))


async def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_quiet_sidecar_leaves_the_default_executor_free(monkeypatch, caplog):
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    caplog.set_level(logging.INFO, logger=photon_adapter.logger.name)
    adapter = _make_adapter(monkeypatch)
    stdout = _QuietStdout(b"sidecar ready\n")
    pump = asyncio.create_task(adapter._supervise_sidecar(SimpleNamespace(stdout=stdout, poll=lambda: 0)))
    probe = None
    try:
        # Once the first line is logged, the pump is blocked reading the next one.
        await _wait_until(lambda: any("sidecar ready" in r.getMessage() for r in caplog.records))
        probe = asyncio.ensure_future(asyncio.to_thread(threading.get_ident))
        done, _ = await asyncio.wait({probe}, timeout=5)
        assert probe in done, "a quiet sidecar's log pump is holding the only default-executor worker"
    finally:
        stdout.close()
        await asyncio.wait_for(pump, 5)
        if probe is not None:
            await asyncio.wait_for(probe, 5)
