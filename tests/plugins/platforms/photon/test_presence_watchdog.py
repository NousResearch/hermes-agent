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
    # Conservative by default: probe only after 10+ minutes of stream silence
    # so quiet shared lines never trigger restart storms.
    assert a._probe_interval == 600.0
    assert a._probe_timeout == 10.0
    assert a._probe_max_failures == 3
    assert a._probe_enabled is True


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), -5.0])
def test_probe_interval_rejects_non_finite_and_negative(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    """float() parses nan/inf without raising (YAML 1.1 even has bare .nan/
    .inf literals). Unguarded, nan/negative would silently defeat the
    ``_probe_interval > 0`` enabled check -- disabling the zombie-stream
    watchdog with no error and no log -- instead of falling back to the
    documented default like any other malformed value."""
    a = _make_adapter(monkeypatch, probe_interval_seconds=bad_value)

    assert a._probe_interval == 600.0
    assert a._probe_enabled is True


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), -5.0])
def test_probe_timeout_rejects_non_finite_and_negative(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    """A bad probe_timeout isn't gated by _probe_enabled: it reaches the
    sidecar HTTP call's ``timeout=`` kwarg directly. Falling back to the
    default here avoids a permanently "inconclusive" probe (or a raised
    timeout construction error) masking a genuinely hung sidecar."""
    a = _make_adapter(monkeypatch, probe_timeout_seconds=bad_value)

    assert a._probe_timeout == 10.0


def test_note_activity_resets_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    a._probe_failures = 2
    before = a._last_upstream_activity
    time.sleep(0.001)
    a._note_upstream_activity()
    assert a._probe_failures == 0
    assert a._last_upstream_activity > before


@pytest.mark.asyncio
async def test_respawn_after_max_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core fix: N consecutive dead probes -> exactly one respawn."""
    a = _make_adapter(monkeypatch, probe_max_failures=3)

    respawns: List[str] = []

    async def _fake_respawn(reason: str) -> None:
        respawns.append(reason)
        a._note_upstream_activity()  # mirror real respawn (clears failures)

    async def _hung_probe() -> str:
        return "hung"

    monkeypatch.setattr(a, "_respawn_sidecar", _fake_respawn)
    monkeypatch.setattr(a, "_probe_once", _hung_probe)

    # Simulate the watchdog's per-iteration decision logic directly (no sleeps).
    a._last_upstream_activity = time.monotonic() - 999  # force a probe each time
    for _ in range(3):
        verdict = await a._probe_once()
        assert verdict == "hung"
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


