"""Photon can tolerate brief upstream stream degradation before going fatal.

Default behaviour remains immediate escalation (degradedForMs grace = 0), so
existing fatal-path tests keep their contracts. Operators who hit restart
storms on transient network blips can set PHOTON_DEGRADED_FATAL_MS.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))


@pytest.mark.asyncio
async def test_default_still_fatal_on_brief_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0
    monkeypatch.delenv("PHOTON_DEGRADED_FATAL_MS", raising=False)

    async def degraded(_path: str, _payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 4000,
                "lastIssue": "brief blip",
            }
        }

    monkeypatch.setattr(adapter, "_sidecar_call", degraded)
    monkeypatch.setattr(adapter, "_dispatch_fatal_notification", lambda: None)

    await asyncio.wait_for(adapter._monitor_sidecar_health(), timeout=2.0)

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"


@pytest.mark.asyncio
async def test_grace_window_tolerates_brief_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0
    monkeypatch.setenv("PHOTON_DEGRADED_FATAL_MS", "30000")

    polls = {"n": 0}

    async def flaky(_path: str, _payload: Dict[str, Any]) -> Dict[str, Any]:
        polls["n"] += 1
        if polls["n"] == 1:
            return {
                "stream": {
                    "ok": False,
                    "state": "degraded",
                    "degradedForMs": 4000,
                    "lastIssue": "brief blip",
                }
            }
        # Recover on the next poll, then stop the monitor.
        adapter._inbound_running = False
        return {"stream": {"ok": True, "state": "healthy", "degradedForMs": 0}}

    monkeypatch.setattr(adapter, "_sidecar_call", flaky)
    monkeypatch.setattr(adapter, "_dispatch_fatal_notification", lambda: None)

    await asyncio.wait_for(adapter._monitor_sidecar_health(), timeout=2.0)

    assert not adapter.has_fatal_error
    assert polls["n"] >= 2


@pytest.mark.asyncio
async def test_grace_window_still_fatals_when_degradation_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0
    monkeypatch.setenv("PHOTON_DEGRADED_FATAL_MS", "5000")

    async def degraded(_path: str, _payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 12000,
                "lastIssue": "still down",
            }
        }

    monkeypatch.setattr(adapter, "_sidecar_call", degraded)
    monkeypatch.setattr(adapter, "_dispatch_fatal_notification", lambda: None)

    await asyncio.wait_for(adapter._monitor_sidecar_health(), timeout=2.0)

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
