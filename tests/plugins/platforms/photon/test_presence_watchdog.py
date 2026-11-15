"""Presence-watchdog tests.

spectrum-ts only reconnects when its inbound iterator throws or ends; a
half-open ("zombie") gRPC socket makes the iterator hang forever (no error, no
end), so inbound silently dies until the sidecar is restarted. The adapter's
presence watchdog probes the upstream channel via the sidecar's ``/probe``
endpoint and respawns the sidecar after repeated probe failures.

These tests exercise the watchdog's decision logic (probe -> count failures ->
respawn; success resets; recent inbound traffic skips the probe) without
spawning Node, binding ports, or hitting the network.
"""
from __future__ import annotations

import time
from typing import Any
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, **extra: Any) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra=dict(extra))
    return PhotonAdapter(cfg)




def test_probe_config_explicit_zero_disables_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit 0 in extra must NOT fall through to the default (``or`` bug):
    # a non-positive interval is the documented escape hatch that disables the watchdog.
    a = _make_adapter(monkeypatch, probe_interval_seconds=0)
    assert a._probe_interval == 0
    assert a._probe_enabled is False


def test_probe_config_invalid_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unparseable values degrade to defaults instead of aborting construction.
    a = _make_adapter(monkeypatch, probe_interval_seconds="soon", probe_timeout_seconds=None,
                      probe_max_failures="lots")
    assert a._probe_interval == 600.0
    assert a._probe_timeout == 10.0
    assert a._probe_max_failures == 3
    assert a._probe_enabled is True


def test_note_activity_resets_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    a._probe_failures = 2
    before = a._last_upstream_activity
    time.sleep(0.001)
    a._note_upstream_activity()
    assert a._probe_failures == 0
    assert a._last_upstream_activity > before


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence,respawns,failures", [
    (["hung", "hung", "hung"], 1, 3),
    (["hung", "hung", "alive", "hung", "hung"], 0, 2),
    (["hung", "inconclusive", "hung"], 0, 2),
    (["recent", "hung"], 0, 1),
])
async def test_watchdog_decisions(monkeypatch, sequence, respawns, failures):
    import plugins.platforms.photon.adapter as module

    adapter = _make_adapter(monkeypatch, probe_max_failures=3)
    adapter._watchdog_running = True
    steps = iter(sequence)
    verdicts = iter([step for step in sequence if step != "recent"])
    probe = AsyncMock(side_effect=lambda: next(verdicts))
    respawn = AsyncMock()

    async def tick(_delay):
        step = next(steps, None)
        if step is None:
            raise asyncio.CancelledError
        adapter._last_upstream_activity = time.monotonic() - (0 if step == "recent" else 999)

    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=tick, CancelledError=asyncio.CancelledError))
    monkeypatch.setattr(adapter, "_probe_once", probe)
    monkeypatch.setattr(adapter, "_respawn_sidecar", respawn)
    task = asyncio.create_task(adapter._presence_watchdog())
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert respawn.await_count == respawns
        assert probe.await_count == len(sequence) - sequence.count("recent")
        assert adapter._probe_failures == failures
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
