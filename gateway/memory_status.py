"""Memory status rollup for ``/api/status`` (NS-656).

The gateway already *produces* every memory-pressure signal a user would
want to know about, but all of it dies in log files:

* :func:`gateway.shutdown_watchdog.write_loop_heartbeat` embeds a
  :func:`gateway.lifecycle_ledger.sample_memory` snapshot (gateway RSS +
  system MemAvailable/MemTotal + swap) in ``state/gateway.heartbeat``
  every 30 seconds.
* :func:`gateway.lifecycle_ledger.record_startup` detects an unclean
  previous death and flags ``suspected_oom`` — but only into
  ``gateway-exit-diag.log`` and a WARNING line.
* ``gateway/agent_cache_pressure.py`` evicts transcripts under pressure,
  again log-only.

So a hosted agent can be OOM-killed hourly (the BlueAtlas incident,
NS-608) while its dashboard and the NAS agent card both look perfectly
healthy.  This module is the read side that closes the gap: it distills
the *already-persisted* heartbeat + lifecycle sentinel into a compact,
public-safe block that ``/api/status`` can serve to the dashboard SPA
and the NAS availability sweep — no new sampling, no IPC with the
gateway process, just two small file reads.

Public-safety note: ``/api/status`` is an unauthenticated liveness probe
(``PUBLIC_API_PATHS``), which is exactly why NAS can consume it.  This
block therefore carries only coarse numbers (MB granularity), enums, and
booleans — the same disclosure class as the existing ``active_agents``
count and ``nous_session_valid`` field (which was added for the same
NAS-sweep audience).

Everything here is best-effort and read-only: a missing/corrupt file
degrades to ``pressure="unknown"`` rather than raising into the status
endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Pressure thresholds on system MemAvailable.  ``critical`` deliberately
# mirrors the lifecycle ledger's OOM-suspicion heuristics
# (:data:`gateway.lifecycle_ledger._LOW_MEM_AVAILABLE_KIB` /
# ``_LOW_MEM_AVAILABLE_FRACTION``): if a memory level would make a
# subsequent unclean death "suspected OOM", the user should already have
# been warned at that level while the process was still alive.
_CRITICAL_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_CRITICAL_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal
_ELEVATED_AVAILABLE_KIB = 128 * 1024  # < 128 MiB available
_ELEVATED_AVAILABLE_FRACTION = 0.15  # < 15% of MemTotal

# Configurable overrides (issue #90713). On ZFS-based hosts (TrueNAS etc.)
# MemAvailable chronically under-credits the reclaimable ARC, so the fixed
# 15% elevated band fires 24/7 on a host that is nowhere near OOM. Users
# tune these via config.yaml under ``dashboard.memory_pressure``:
#
#   dashboard:
#     memory_pressure:
#       elevated_fraction: 0.08    # default 0.15
#       critical_fraction: 0.05    # default 0.05 -- stays conservative
#       elevated_kib: 131072       # optional absolute floor
#       critical_kib: 65536        # optional absolute floor
#       show_elevated_banner: true # false silences ONLY the elevated class
#
# NO new env vars: .env is secrets-only (repo policy); all behavioural
# settings live in config.yaml. The critical class is deliberately NOT
# silenceable -- it feeds the lifecycle ledger's OOM-suspicion heuristics.


def _memory_pressure_config() -> Dict[str, Any]:
    """Best-effort read of ``dashboard.memory_pressure`` from config.yaml.

    Cached on config mtime by ``load_config_readonly``; any failure (missing
    file, parse error, import trouble) degrades to an empty dict so the
    hardcoded defaults below keep the classifier alive.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    dash = cfg.get("dashboard")
    if not isinstance(dash, dict):
        return {}
    mp = dash.get("memory_pressure")
    return mp if isinstance(mp, dict) else {}


def _resolve_thresholds() -> Dict[str, Any]:
    """Merge config overrides over the hardcoded defaults, validated.

    Invalid values (wrong type, out of range) are dropped rather than
    clamped silently -- a typo must not turn the classifier into nonsense.
    """
    overrides = _memory_pressure_config()
    resolved: Dict[str, Any] = {
        "critical_kib": _CRITICAL_AVAILABLE_KIB,
        "critical_fraction": _CRITICAL_AVAILABLE_FRACTION,
        "elevated_kib": _ELEVATED_AVAILABLE_KIB,
        "elevated_fraction": _ELEVATED_AVAILABLE_FRACTION,
    }
    for key in resolved:
        value = overrides.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value <= 0:
            continue
        if key.endswith("_kib"):
            if not isinstance(value, int) or value < 1024:
                continue
        else:  # fraction
            if value > 1.0:
                continue
        resolved[key] = value

    # Cross-validation: if the elevated band was overridden to land inside
    # or below the critical band, drop the elevated overrides.  Without this
    # a typo like ``elevated_fraction: 0.02, critical_fraction: 0.05`` would
    # make everything at/below 5% report critical — false OOM evidence for
    # the lifecycle ledger.
    if resolved["elevated_fraction"] <= resolved["critical_fraction"]:
        resolved["elevated_fraction"] = _ELEVATED_AVAILABLE_FRACTION
    if resolved["elevated_kib"] <= resolved["critical_kib"]:
        resolved["elevated_kib"] = _ELEVATED_AVAILABLE_KIB

    return resolved

# A heartbeat older than this no longer describes the present.  The writer
# cadence is 30s (DEFAULT_HEARTBEAT_INTERVAL_S); 150s of slack tolerates a
# briefly stalled loop without letting a long-dead gateway's last sample
# masquerade as current pressure.
_HEARTBEAT_FRESH_TTL_S = 150.0

_KIB_PER_MB = 1024


def _mb(kib: Any) -> Optional[int]:
    if isinstance(kib, bool) or not isinstance(kib, int) or kib < 0:
        return None
    return kib // _KIB_PER_MB


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def classify_pressure(
    available_kib: Any, total_kib: Any, thresholds: Optional[Dict[str, Any]] = None
) -> str:
    """Map a MemAvailable/MemTotal pair to ``ok``/``elevated``/``critical``.

    ``unknown`` when the sample is missing or malformed — the caller must
    not treat "we could not read it" as "memory is fine".

    ``thresholds`` overrides the four band boundaries (``critical_kib``,
    ``critical_fraction``, ``elevated_kib``, ``elevated_fraction``); when
    omitted they come from ``dashboard.memory_pressure`` config merged over
    the hardcoded defaults (issue #90713). Both consumers — the dashboard
    banner and the kanban dispatch guard — call this same function, so a
    tune applies everywhere at once.
    """
    if (
        isinstance(available_kib, bool)
        or not isinstance(available_kib, int)
        or available_kib < 0
    ):
        return "unknown"
    fraction: Optional[float] = None
    if (
        not isinstance(total_kib, bool)
        and isinstance(total_kib, int)
        and total_kib > 0
    ):
        fraction = available_kib / total_kib
    th = thresholds if thresholds is not None else _resolve_thresholds()
    if available_kib < th["critical_kib"] or (
        fraction is not None and fraction < th["critical_fraction"]
    ):
        return "critical"
    if available_kib < th["elevated_kib"] or (
        fraction is not None and fraction < th["elevated_fraction"]
    ):
        return "elevated"
    return "ok"


def _read_heartbeat(home: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        from gateway.lifecycle_ledger import _read_json
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        return _read_json(get_loop_heartbeat_path(home))
    except Exception:
        return None


def _read_sentinel(home: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        from gateway.lifecycle_ledger import (
            _read_json,
            get_lifecycle_sentinel_path,
        )

        return _read_json(get_lifecycle_sentinel_path(home))
    except Exception:
        return None


def collect_memory_status(
    home: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the ``memory`` block for ``/api/status``.

    ``home`` scopes the read to a profile's HERMES_HOME (the status
    endpoint's ``?profile=`` handling passes it through); ``None`` means
    the active profile.  ``now`` is injectable for tests.

    Always returns a dict — a gateway that is down, has never written a
    heartbeat, or whose files are corrupt yields
    ``{"pressure": "unknown", ...}`` with whatever fields could still be
    recovered.  Never raises.
    """
    moment = now or datetime.now(timezone.utc)
    status: Dict[str, Any] = {
        "pressure": "unknown",
        "gateway_rss_mb": None,
        "system_total_mb": None,
        "system_available_mb": None,
        "swap_used_mb": None,
        "sampled_at": None,
        "last_boot_unclean": False,
        "last_boot_suspected_oom": False,
        # Identity of the CURRENT gateway life (the sentinel's started_at).
        # A suspected-OOM restart writes a fresh sentinel, so this changes on
        # every boot — the dashboard keys banner dismissal on it so that
        # acknowledging one OOM restart does not mute reports of the NEXT
        # one (the hourly-restart-loop case is exactly the one that matters).
        "boot_id": None,
    }

    heartbeat = _read_heartbeat(home)
    if heartbeat:
        sampled_at = _parse_iso(heartbeat.get("updated_at"))
        mem = heartbeat.get("mem")
        if isinstance(mem, dict):
            status["gateway_rss_mb"] = _mb(mem.get("rss_kib"))
            status["system_total_mb"] = _mb(mem.get("mem_total_kib"))
            status["system_available_mb"] = _mb(mem.get("mem_available_kib"))
            status["swap_used_mb"] = _mb(mem.get("swap_used_kib"))
            if sampled_at is not None:
                status["sampled_at"] = sampled_at.isoformat()
                age_s = (moment - sampled_at).total_seconds()
                if 0 <= age_s <= _HEARTBEAT_FRESH_TTL_S:
                    pressure = classify_pressure(
                        mem.get("mem_available_kib"),
                        mem.get("mem_total_kib"),
                    )
                    # Config escape hatch for chronic-elevated hosts
                    # (ZFS ARC, #90713): ``show_elevated_banner: false``
                    # downgrades an *elevated* verdict to ``ok`` in this
                    # public rollup.  Raw numbers stay visible; the
                    # critical class is never silenced.  Note: the kanban
                    # dispatch guard calls ``classify_pressure()`` directly
                    # and still sees the raw elevated — this flag controls
                    # the /api/status rollup only (display concern, not
                    # dispatch logic).
                    if (
                        pressure == "elevated"
                        and _memory_pressure_config().get(
                            "show_elevated_banner", True
                        )
                        is False
                    ):
                        pressure = "ok"
                    status["pressure"] = pressure
                # else: stale sample — numbers are still reported (they are
                # honest about *when* via sampled_at) but pressure stays
                # "unknown" so a dead gateway's final gasp cannot render a
                # live "critical" banner forever.

    sentinel = _read_sentinel(home)
    if sentinel:
        status["last_boot_unclean"] = bool(sentinel.get("prior_unclean_exit"))
        status["last_boot_suspected_oom"] = bool(
            sentinel.get("prior_suspected_oom")
        )
        started_at = sentinel.get("started_at")
        if isinstance(started_at, str) and started_at:
            status["boot_id"] = started_at

    return status
