"""Pluggable believed-state feeds for AMDP.

AMDP needs a view of the system's current state to satisfy the paper's step-1
precondition (do not plan blind). The *source* of that state is deliberately
pluggable so AMDP is a standalone orchestration-layer feature, not one coupled
to any particular monitoring plugin:

* ``gateway``       — universal. Reads the gateway runtime status file that
                      every Hermes install writes. No external dependencies.
                      This is the default and the upstream-safe baseline.
* ``telemetry``     — fast native host telemetry (nvidia-smi / df / tailscale /
                      gateway file, in-process, no HTTP dashboard) via the
                      proprioception telemetry sensor. Avoids a seconds-slow
                      ops-dashboard read; falls back to ``gateway`` if absent.
* ``proprioception``— optional enrichment. If the proprioception plugin is
                      installed, use its richer snapshot (external system
                      dashboard + attention states + staleness). Falls back to
                      the gateway feed if the plugin is absent.
* ``auto``          — proprioception if importable, else gateway (the default).

Every feed returns the same believed-state contract consumed by the planner::

    {brief, sensors_down, staleness_s, verdict, gateway_state, system_count,
     dashboard_up}

Only ``"gateway-status"`` appearing in ``sensors_down`` (or staleness beyond the
configured max) blinds the planner into refusing; a richer feed's enrichment
being unavailable degrades gracefully rather than refusing.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _blind() -> dict[str, Any]:
    """No usable state at all — the planner will refuse (truly blind)."""
    return {"brief": "", "sensors_down": ["gateway-status"], "staleness_s": 0.0,
            "verdict": "unknown", "gateway_state": "unknown", "system_count": 0,
            "dashboard_up": False}


def gateway_feed(config: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    """Universal feed: the gateway runtime status file. No plugins required."""
    try:
        from gateway.status import read_runtime_status

        status = read_runtime_status()
        if not status:
            return _blind()
        gw = str(status.get("gateway_state") or status.get("state") or "running")
        agents = status.get("active_agents")
        brief = f"gateway: {gw}"
        if agents is not None:
            brief += f" ({agents} active agent(s))"
        brief += "\nno external system dashboard configured — planning on gateway status"
        return {
            "brief": brief,
            "sensors_down": [],          # gateway status IS present → not blind
            "staleness_s": 0.0,          # runtime file is live
            "verdict": "ok",
            "gateway_state": gw,
            "system_count": 0,
            "dashboard_up": False,
        }
    except Exception as exc:
        logger.warning("AMDP gateway state feed failed: %s", exc)
        return _blind()


def proprioception_feed(config: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    """Optional enrichment feed via the proprioception plugin. Falls back to the
    gateway feed if the plugin is not installed."""
    try:
        from plugins.proprioception.collector import ATTENTION_STATES, get_snapshot
        from plugins.proprioception.settings import DEFAULTS
    except Exception:
        logger.debug("AMDP: proprioception plugin unavailable; using gateway feed")
        return gateway_feed(config, timeout_s)
    try:
        settings = dict(DEFAULTS)
        block = (config or {}).get("proprioception")
        if isinstance(block, dict):
            settings.update(block)
        if timeout_s:
            settings = {**settings, "timeout_seconds": float(timeout_s)}
        snap = get_snapshot(settings, force=False)
        systems = (snap.dashboard or {}).get("systems") or []
        attention = [s for s in systems if isinstance(s, dict) and str(s.get("state")) in ATTENTION_STATES]
        dashboard_up = snap.dashboard is not None
        verdict = "attention" if attention else ("ok" if dashboard_up else "unknown")
        blinding = ["gateway-status"] if "gateway-status" in list(snap.sensors_down) else []
        lines = [f"overall verdict: {verdict}", f"gateway: {snap.gateway_state}"]
        if not dashboard_up:
            lines.append("system dashboard unavailable — planning on gateway status alone")
        if snap.dashboard_stale_for:
            lines.append(f"state staleness: {snap.dashboard_stale_for:.0f}s")
        if attention:
            lines.append("systems needing attention:")
            for s in attention[:12]:
                lines.append(f"  - {s.get('label', s.get('id', '?'))}: {s.get('state')} ({s.get('detail', '')})")
        elif dashboard_up:
            lines.append(f"all {len(systems)} monitored systems calm")
        return {
            "brief": "\n".join(lines),
            "sensors_down": blinding,
            "staleness_s": float(snap.dashboard_stale_for or 0.0),
            "verdict": verdict,
            "gateway_state": snap.gateway_state,
            "system_count": len(systems),
            "dashboard_up": dashboard_up,
        }
    except Exception as exc:
        logger.warning("AMDP proprioception feed failed, falling back to gateway: %s", exc)
        return gateway_feed(config, timeout_s)


def telemetry_feed(config: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    """Fast, in-process system state via the native telemetry sensor (direct CLI
    probes: nvidia-smi / df / tailscale / gateway file — no HTTP dashboard, so it
    avoids the seconds-slow ops-dashboard read). Uses a freshly persisted
    snapshot if one is available, else collects one live. Falls back to the
    gateway feed if the sensor module is unavailable."""
    try:
        from plugins.proprioception import telemetry
    except Exception:
        logger.debug("AMDP: telemetry sensor unavailable; using gateway feed")
        return gateway_feed(config, timeout_s)
    try:
        import time

        snap = telemetry.load()
        age: float = 0.0
        collected = snap.get("collected_at") if isinstance(snap, dict) else None
        if snap and isinstance(collected, (int, float)):
            age = max(0.0, time.time() - collected)
        # Persisted snapshot only if genuinely fresh; otherwise collect live.
        if not snap or age > 60.0:
            snap = telemetry.snapshot()
            age = 0.0
        gw = snap.get("gateway") or {}
        gateway_state = str(gw.get("gateway_state") or gw.get("state") or ("running" if gw else "unknown"))
        sensors_down = [] if gw else ["gateway-status"]
        gpu, disk, net = snap.get("gpu"), snap.get("disk"), snap.get("network")

        lines = [f"gateway: {gateway_state}"]
        signals = 0
        if isinstance(gpu, list) and gpu and isinstance(gpu[0], dict):
            g0 = gpu[0]
            lines.append(f"GPU {g0.get('name', '?')}: {g0.get('temp_c', '?')}C, "
                         f"VRAM {g0.get('vram_used_mb', 0)}/{g0.get('vram_total_mb', 0)} MB")
            signals += 1
        if isinstance(disk, dict) and not disk.get("error"):
            lines.append(f"disk C: {disk.get('free_gb', '?')} free of {disk.get('total_gb', '?')}")
            signals += 1
        if isinstance(net, dict) and not net.get("error"):
            lines.append(f"tailscale: {net.get('tailscale_peers_online', '?')}/{net.get('total_peers', '?')} peers online")
            signals += 1
        if age:
            lines.append(f"telemetry age: {age:.0f}s")
        return {
            "brief": "\n".join(lines),
            "sensors_down": sensors_down,
            "staleness_s": float(age),
            "verdict": "ok" if gw else "unknown",
            "gateway_state": gateway_state,
            "system_count": signals,
            "dashboard_up": False,
        }
    except Exception as exc:
        logger.warning("AMDP telemetry feed failed, falling back to gateway: %s", exc)
        return gateway_feed(config, timeout_s)


def _proprioception_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("plugins.proprioception.collector") is not None
    except Exception:
        return False


def get_believed_state(
    config: dict[str, Any], timeout_s: float | None = None, mode: str = "auto"
) -> dict[str, Any]:
    """Resolve the configured state feed and return the believed-state dict.
    Never raises — a total failure returns a blind state (planner refuses)."""
    try:
        mode = (mode or "auto").strip().lower()
        if mode == "gateway":
            return gateway_feed(config, timeout_s)
        if mode == "telemetry":
            return telemetry_feed(config, timeout_s)
        if mode == "proprioception":
            return proprioception_feed(config, timeout_s)
        # auto: enrichment if present, else the universal gateway feed
        if _proprioception_available():
            return proprioception_feed(config, timeout_s)
        return gateway_feed(config, timeout_s)
    except Exception as exc:
        logger.warning("AMDP state feed resolution failed: %s", exc)
        return _blind()
