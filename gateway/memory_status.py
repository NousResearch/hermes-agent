"""Memory status rollup for ``/api/status``.

Read side for signals the gateway already persists: the 30s ``state/gateway.heartbeat``
(RSS + MemAvailable/MemTotal + swap) and the lifecycle sentinel's ``suspected_oom``
flag — two small file reads, no IPC.  ``/api/status`` is unauthenticated, so only
coarse numbers (MB), enums and booleans.  A missing/corrupt file degrades to
``pressure="unknown"`` rather than raising into the status endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Thresholds on system MemAvailable.  ``critical`` doubles as the lifecycle
# ledger's OOM-suspicion heuristic: a level that makes a later unclean death
# "suspected OOM" already warns while the process is alive.
_CRITICAL_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_CRITICAL_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal
_ELEVATED_AVAILABLE_KIB = 128 * 1024  # < 128 MiB available
_ELEVATED_AVAILABLE_FRACTION = 0.15  # < 15% of MemTotal
_PRESSURE_TIERS = (  # order-sensitive: worst first
    ("critical", _CRITICAL_AVAILABLE_KIB, _CRITICAL_AVAILABLE_FRACTION),
    ("elevated", _ELEVATED_AVAILABLE_KIB, _ELEVATED_AVAILABLE_FRACTION),
)

# Swap occupancy that escalates the MemAvailable verdict (never downgrades
# it): a host deep into swap OOM-kills while MemAvailable still reads fine.
_ELEVATED_SWAP_FRACTION = 0.5  # >= 50% of swap in use
_CRITICAL_SWAP_FRACTION = 0.8  # >= 80% of swap in use
_SEVERITY = {"unknown": 0, "ok": 1, "elevated": 2, "critical": 3}

# Writer cadence is 30s; 150s tolerates a briefly stalled loop without letting
# a long-dead gateway's last sample pose as current.
_HEARTBEAT_FRESH_TTL_S = 150.0


def _nonneg_int(value: Any) -> Optional[int]:
    """Return *value* if it is a non-negative int (bools rejected), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _mb(kib: Any) -> Optional[int]:
    return None if _nonneg_int(kib) is None else kib // 1024


def _parse_iso(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) and value else None
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed is not None and parsed.tzinfo is None else parsed


def _swap_tier(swap_used_kib: Any, swap_total_kib: Any) -> Optional[str]:
    """Elevated/critical from swap occupancy, else None (skip the tier)."""
    used, total = _nonneg_int(swap_used_kib), _nonneg_int(swap_total_kib)
    if used is None or total is None or total == 0 or used > total:
        return None
    occupancy = used / total
    if occupancy >= _CRITICAL_SWAP_FRACTION:
        return "critical"
    if occupancy >= _ELEVATED_SWAP_FRACTION:
        return "elevated"
    return None


def classify_pressure(
    available_kib: Any,
    total_kib: Any,
    swap_used_kib: Any = None,
    swap_total_kib: Any = None,
) -> str:
    """Ok/elevated/critical from MemAvailable/MemTotal, escalated only by swap."""
    available, total = _nonneg_int(available_kib), _nonneg_int(total_kib)
    if available is None:
        level = "unknown"
    else:
        level = "ok"
        fraction = available / total if total else None
        for tier, kib_floor, frac_floor in _PRESSURE_TIERS:
            if available < kib_floor or (fraction is not None and fraction < frac_floor):
                level = tier
                break
    swap_tier = _swap_tier(swap_used_kib, swap_total_kib)
    if swap_tier is not None and _SEVERITY[swap_tier] > _SEVERITY[level]:
        level = swap_tier
    return level


def _read_state_files(home: Optional[Path]) -> tuple:
    """``(heartbeat, sentinel)`` dicts, each ``None`` when unreadable."""
    try:
        from gateway.lifecycle_ledger import _read_json, get_lifecycle_sentinel_path
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        return _read_json(get_loop_heartbeat_path(home)), _read_json(get_lifecycle_sentinel_path(home))
    except Exception:
        return None, None


def collect_memory_status(
    home: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """``memory`` block for ``/api/status``; ``home`` scopes to a profile (``None`` =
    active), ``now`` is injectable.  Never raises — a down gateway or corrupt files
    yield ``pressure="unknown"`` plus whatever fields could be recovered."""
    moment = now or datetime.now(timezone.utc)
    status: Dict[str, Any] = {
        "pressure": "unknown", "gateway_rss_mb": None, "system_total_mb": None, "system_available_mb": None,
        "swap_used_mb": None, "sampled_at": None, "last_boot_unclean": False, "last_boot_suspected_oom": False,
        # Identity of the CURRENT life (sentinel started_at): the dashboard keys
        # banner dismissal on it so acknowledging one OOM restart does not mute the NEXT.
        "boot_id": None,
    }

    heartbeat, sentinel = _read_state_files(home)
    if heartbeat:
        sampled_at, mem = _parse_iso(heartbeat.get("updated_at")), heartbeat.get("mem")
        if isinstance(mem, dict):
            for dst, src in (("gateway_rss_mb", "rss_kib"), ("system_total_mb", "mem_total_kib"),
                             ("system_available_mb", "mem_available_kib"), ("swap_used_mb", "swap_used_kib")):
                status[dst] = _mb(mem.get(src))
            if sampled_at is not None:
                status["sampled_at"] = sampled_at.isoformat()
                # Stale sample: numbers still reported (sampled_at says when) but
                # pressure stays "unknown" so a dead gateway's final gasp cannot
                # render a live "critical" banner forever.
                if 0 <= (moment - sampled_at).total_seconds() <= _HEARTBEAT_FRESH_TTL_S:
                    status["pressure"] = classify_pressure(
                        mem.get("mem_available_kib"), mem.get("mem_total_kib"),
                        mem.get("swap_used_kib"), mem.get("swap_total_kib"),
                    )

    if sentinel:
        status["last_boot_unclean"] = bool(sentinel.get("prior_unclean_exit"))
        status["last_boot_suspected_oom"] = bool(sentinel.get("prior_suspected_oom"))
        started_at = sentinel.get("started_at")
        status["boot_id"] = started_at if isinstance(started_at, str) and started_at else None

    return status


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'logger': ('gateway.run', 'logger'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
