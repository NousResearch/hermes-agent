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
from typing import Any, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, **extra: Any) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra=dict(extra))
    return PhotonAdapter(cfg)


def test_probe_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    assert a._probe_interval == 60.0
    assert a._probe_timeout == 10.0
    assert a._probe_max_failures == 3
    assert a._probe_enabled is True


def test_probe_config_from_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(
        monkeypatch,
        probe_interval_seconds=30,
        probe_timeout_seconds=5,
        probe_max_failures=2,
    )
    assert a._probe_interval == 30.0
    assert a._probe_timeout == 5.0
    assert a._probe_max_failures == 2
    assert a._probe_enabled is True


def test_probe_disabled_when_interval_nonpositive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _make_adapter(monkeypatch, probe_interval_seconds=0)
    assert a._probe_enabled is False


def test_note_activity_resets_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    a._probe_failures = 2
    before = a._last_upstream_activity
    time.sleep(0.001)
    a._note_upstream_activity()
    assert a._probe_failures == 0
    assert a._last_upstream_activity > before


@pytest.mark.asyncio
async def test_probe_once_alive_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)

    class _Resp:
        status_code = 200

    class _Client:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return _Resp()

    a._http_client = _Client()  # type: ignore[assignment]
    assert await a._probe_once() is True


@pytest.mark.asyncio
async def test_probe_once_dead_on_500(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)

    class _Resp:
        status_code = 500

    class _Client:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return _Resp()

    a._http_client = _Client()  # type: ignore[assignment]
    assert await a._probe_once() is False


@pytest.mark.asyncio
async def test_probe_once_dead_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)

    class _Client:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("zombie stream — probe hung")

    a._http_client = _Client()  # type: ignore[assignment]
    # A hung/half-open probe must read as NOT alive (this is the zombie case).
    assert await a._probe_once() is False


@pytest.mark.asyncio
async def test_probe_once_false_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _make_adapter(monkeypatch)
    a._http_client = None
    assert await a._probe_once() is False


@pytest.mark.asyncio
async def test_respawn_after_max_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core fix: N consecutive dead probes -> exactly one respawn."""
    a = _make_adapter(monkeypatch, probe_max_failures=3)

    respawns: List[str] = []

    async def _fake_respawn(reason: str) -> None:
        respawns.append(reason)
        a._note_upstream_activity()  # mirror real respawn (clears failures)

    async def _dead_probe() -> bool:
        return False

    monkeypatch.setattr(a, "_respawn_sidecar", _fake_respawn)
    monkeypatch.setattr(a, "_probe_once", _dead_probe)

    # Simulate the watchdog's per-iteration decision logic directly (no sleeps).
    a._last_upstream_activity = time.monotonic() - 999  # force a probe each time
    for _ in range(3):
        alive = await a._probe_once()
        assert alive is False
        a._probe_failures += 1
        if a._probe_failures >= a._probe_max_failures:
            await a._respawn_sidecar("test")

    assert respawns == ["test"]
    assert a._probe_failures == 0  # reset by the (faked) respawn


@pytest.mark.asyncio
async def test_success_resets_failure_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live probe between dead ones prevents a respawn (failures reset)."""
    a = _make_adapter(monkeypatch, probe_max_failures=3)

    respawns: List[str] = []

    async def _fake_respawn(reason: str) -> None:
        respawns.append(reason)

    monkeypatch.setattr(a, "_respawn_sidecar", _fake_respawn)

    # Two failures, then a success, then two more failures: never hits 3 in a row.
    sequence = [False, False, True, False, False]
    for alive in sequence:
        if alive:
            a._note_upstream_activity()
        else:
            a._probe_failures += 1
            if a._probe_failures >= a._probe_max_failures:
                await a._respawn_sidecar("should-not-fire")

    assert respawns == []
    assert a._probe_failures == 2


@pytest.mark.asyncio
async def test_respawn_calls_stop_then_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respawn tears the sidecar down and brings a fresh one up, in order."""
    a = _make_adapter(monkeypatch)
    calls: List[str] = []

    async def _fake_stop() -> None:
        calls.append("stop")

    async def _fake_start() -> None:
        calls.append("start")

    monkeypatch.setattr(a, "_stop_sidecar", _fake_stop)
    monkeypatch.setattr(a, "_start_sidecar", _fake_start)

    await a._respawn_sidecar("test reason")
    assert calls == ["stop", "start"]
    # A fresh stream means failures are cleared.
    assert a._probe_failures == 0


@pytest.mark.asyncio
async def test_respawn_is_lock_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second respawn while one is in flight is skipped, not double-spawned."""
    import asyncio

    a = _make_adapter(monkeypatch)
    starts: List[str] = []

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_stop() -> None:
        started.set()
        await release.wait()

    async def _fake_start() -> None:
        starts.append("start")

    monkeypatch.setattr(a, "_stop_sidecar", _slow_stop)
    monkeypatch.setattr(a, "_start_sidecar", _fake_start)
    a._respawn_lock = asyncio.Lock()

    first = asyncio.create_task(a._respawn_sidecar("first"))
    await started.wait()  # first respawn is now holding the lock inside _stop
    # Second call should see the lock held and return immediately.
    await a._respawn_sidecar("second")
    assert starts == []  # second did not proceed to start
    release.set()
    await first
    assert starts == ["start"]  # only the first respawn started a new sidecar
