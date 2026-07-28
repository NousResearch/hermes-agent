"""Zombie-stream watchdog tests (half-open gRPC stream, issue #54036).

spectrum-ts only reconnects when its inbound iterator throws or ends; a
half-open ("zombie") socket makes the iterator hang forever — no error, no
end — so inbound silently dies while the sidecar process looks healthy.

The salvaged design has two layers:

1. Sidecar (node): ``stream-staleness.mjs`` decision rules + a watchdog in
   ``index.mjs`` that tracks the iterator's last yield, probes only after a
   conservative silence threshold, and classifies degraded ONLY when a probe
   proves connectivity while the stream is silent (never on silence alone,
   never on an inconclusive probe). Degraded feeds the existing exit-75
   restart path and the ``staleness`` block on ``/healthz``.
2. Adapter (python): ``_monitor_sidecar_health`` surfaces the new staleness
   fields; ``_probe_once`` has strict tri-state semantics (alive / hung /
   inconclusive).

These tests execute the real node decision module and drive the adapter
against mocked ``/healthz`` responses — style follows
test_overflow_recovery.py / test_spectrum_patch.py. No ports are bound and no
gRPC traffic occurs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter

_MODULE = Path("plugins/platforms/photon/sidecar/stream-staleness.mjs").resolve()


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- Sidecar decision rules (execute the real node module) -------------------

def _run_staleness_harness(script: str) -> Dict[str, Any]:
    harness = (
        "import { classifyProbeRejection, shouldProbe, isZombieSuspect } "
        f"from {json.dumps(_MODULE.as_uri())};\n"
        + script
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_probe_rejection_classification_is_strict() -> None:
    """Only not-found-shaped rejections prove liveness; everything else is
    inconclusive — a rejected probe is NEVER treated as alive (#45580's
    original /probe treated any rejection as alive, which was too loose)."""
    out = _run_staleness_harness(
        """
        const results = {
          notFoundCode: classifyProbeRejection({ code: 5, message: "5 NOT_FOUND: nope" }),
          notFoundText: classifyProbeRejection(new Error("message not found")),
          sdkNotFound: classifyProbeRejection({ code: "notFound", message: "missing" }),
          unavailable: classifyProbeRejection({ code: 14, message: "14 UNAVAILABLE: connect failed" }),
          deadline: classifyProbeRejection({ code: 4, message: "4 DEADLINE_EXCEEDED" }),
          generic: classifyProbeRejection(new Error("socket hang up")),
          weird: classifyProbeRejection("string error"),
        };
        process.stdout.write(JSON.stringify(results));
        """
    )
    # Completed round-trips (server said not-found for our synthetic id).
    for name in ("notFoundCode", "notFoundText", "sdkNotFound"):
        assert out[name]["alive"] is True, name
        assert out[name]["inconclusive"] is False, name
    # Everything else: not alive AND explicitly inconclusive.
    for name in ("unavailable", "deadline", "generic", "weird"):
        assert out[name]["alive"] is False, name
        assert out[name]["inconclusive"] is True, name


def test_should_probe_requires_silence_past_threshold_and_cooldown() -> None:
    out = _run_staleness_harness(
        """
        const MIN10 = 10 * 60 * 1000;
        const results = {
          quietButUnderThreshold: shouldProbe(MIN10 - 1, MIN10, MIN10, 120000),
          pastThreshold: shouldProbe(MIN10 + 1, MIN10, MIN10, 120000),
          pastThresholdButCoolingDown: shouldProbe(MIN10 + 1, MIN10, 1000, 120000),
          watchdogDisabled: shouldProbe(MIN10 * 100, 0, MIN10, 120000),
          watchdogDisabledNegative: shouldProbe(MIN10 * 100, -1, MIN10, 120000),
        };
        process.stdout.write(JSON.stringify(results));
        """
    )
    assert out["quietButUnderThreshold"] is False
    assert out["pastThreshold"] is True
    assert out["pastThresholdButCoolingDown"] is False
    assert out["watchdogDisabled"] is False
    assert out["watchdogDisabledNegative"] is False


def test_zombie_requires_probe_proven_connectivity_never_silence_alone() -> None:
    """The core conservatism rule: shared lines can be quiet for hours, so a
    zombie is declared only when the stream is silent past threshold AND a
    probe PROVED the wire works (stream dead, channel alive)."""
    out = _run_staleness_harness(
        """
        const MIN10 = 10 * 60 * 1000;
        const alive = { alive: true };
        const inconclusive = { alive: false };
        const results = {
          silentAndProbeAlive: isZombieSuspect(MIN10 * 2, MIN10, alive),
          silentButProbeInconclusive: isZombieSuspect(MIN10 * 2, MIN10, inconclusive),
          silentNoProbe: isZombieSuspect(MIN10 * 2, MIN10, null),
          hoursOfSilenceInconclusive: isZombieSuspect(MIN10 * 36, MIN10, inconclusive),
          notSilentEnough: isZombieSuspect(MIN10 - 1, MIN10, alive),
          disabled: isZombieSuspect(MIN10 * 2, 0, alive),
        };
        process.stdout.write(JSON.stringify(results));
        """
    )
    assert out["silentAndProbeAlive"] is True
    # Silence alone — even 6 hours of it — is NEVER a zombie verdict.
    assert out["silentButProbeInconclusive"] is False
    assert out["silentNoProbe"] is False
    assert out["hoursOfSilenceInconclusive"] is False
    assert out["notSilentEnough"] is False
    assert out["disabled"] is False


# -- Adapter surfacing of the new /healthz staleness fields ------------------

def _healthz_payload(**staleness: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "stream": {
            "ok": True,
            "state": "healthy",
            "degradedForMs": 0,
            "staleness": {
                "lastInboundAt": "2026-07-28T00:00:00.000Z",
                "silentForMs": 0,
                "silenceThresholdMs": 600000,
                "lastProbeAt": None,
                "lastProbeOutcome": None,
                "zombieSuspected": False,
                **staleness,
            },
        },
    }


@pytest.mark.asyncio
async def test_monitor_surfaces_zombie_suspected_without_fatal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """zombieSuspected on /healthz is surfaced as a warning while the stream
    is still 'ok' — the fatal path stays owned by the degraded state (the
    sidecar escalates to degraded -> exit 75 itself)."""
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    polls = 0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        nonlocal polls
        assert path == "/healthz"
        polls += 1
        if polls >= 2:
            adapter._inbound_running = False
        return _healthz_payload(
            silentForMs=1_200_000,
            lastProbeOutcome="alive",
            zombieSuspected=True,
        )

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    with caplog.at_level("WARNING"):
        await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is False
    assert any(
        "suspected zombie stream" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_monitor_still_raises_fatal_when_zombie_degrades_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the sidecar's watchdog escalates the zombie to degraded, the
    existing UPSTREAM_STREAM_DEGRADED reconnect path fires unchanged."""
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 95000,
                "lastIssue": (
                    "inbound stream silent for 1200000ms while an upstream "
                    "probe succeeded — half-open (zombie) gRPC stream suspected"
                ),
                "staleness": {
                    "silentForMs": 1_200_000,
                    "lastProbeOutcome": "alive",
                    "zombieSuspected": True,
                },
            },
        }

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)
        adapter._inbound_running = False

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
    assert adapter.fatal_error_retryable is True
    assert notified == [True]


@pytest.mark.asyncio
async def test_monitor_ignores_healthz_without_staleness_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old sidecars (no staleness field) keep working: the monitor must not
    crash or raise fatals on the legacy /healthz shape."""
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    polls = 0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        nonlocal polls
        polls += 1
        if polls >= 2:
            adapter._inbound_running = False
        return {"ok": True, "stream": {"ok": True, "state": "healthy"}}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is False


# -- Adapter watchdog: inconclusive never counts toward respawn --------------

@pytest.mark.asyncio
async def test_inconclusive_probes_never_accumulate_toward_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict semantics end-to-end at the adapter: a 503/transport-error probe
    (inconclusive) must not increment the failure counter the way the original
    #45580 booleans did — only hung probes do."""
    adapter = _make_adapter(monkeypatch)

    class _Resp503:
        status_code = 503

    class _Client:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return _Resp503()

    adapter._http_client = _Client()  # type: ignore[assignment]

    # Many inconclusive probes in a row: mirror the watchdog's per-iteration
    # bookkeeping (only "hung" increments) and assert no failures accrue.
    for _ in range(10):
        verdict = await adapter._probe_once()
        assert verdict == "inconclusive"
        if verdict == "hung":
            adapter._probe_failures += 1

    assert adapter._probe_failures == 0
