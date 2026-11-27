"""Memory status rollup for ``/api/status``.

Read side for signals the gateway already persists: the 30s ``state/gateway.heartbeat``
(RSS + MemAvailable/MemTotal + swap) and the lifecycle sentinel's prior-exit flags
(``suspected_oom``, kernel OOM-kill activity, watchdog exit, memory starvation) — two small file reads, no IPC.
``/api/status`` is unauthenticated, so only coarse numbers (MB), enums and booleans.  A
missing/corrupt file degrades to ``pressure="unknown"`` rather than raising into the status
endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Dict, Optional

# Thresholds on system MemAvailable.  ``critical`` doubles as the lifecycle
# ledger's OOM-suspicion heuristic: a level that makes a later unclean death
# "suspected OOM" already warns while the process is alive.
_CRITICAL_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_CRITICAL_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal
_ELEVATED_AVAILABLE_KIB = 128 * 1024  # < 128 MiB available
_ELEVATED_AVAILABLE_FRACTION = 0.15  # < 15% of MemTotal
PRESSURE_CRITICAL = "critical"
_PRESSURE_TIERS = (  # order-sensitive: worst first
    (PRESSURE_CRITICAL, _CRITICAL_AVAILABLE_KIB, _CRITICAL_AVAILABLE_FRACTION),
    ("elevated", _ELEVATED_AVAILABLE_KIB, _ELEVATED_AVAILABLE_FRACTION),
)
# Every level ``classify_pressure`` (and so ``collect_memory_status``'s ``pressure``) can return.
PRESSURE_LEVELS = ("ok", *(tier[0] for tier in _PRESSURE_TIERS), "unknown")

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
    return parsed.replace(tzinfo=UTC) if parsed is not None and parsed.tzinfo is None else parsed


def classify_pressure(available_kib: Any, total_kib: Any) -> str:
    """``ok``/``elevated``/``critical`` from MemAvailable/MemTotal; ``unknown`` when the
    sample is missing/malformed — "could not read it" must never read as "fine"."""
    available, total = _nonneg_int(available_kib), _nonneg_int(total_kib)
    if available is None:
        return "unknown"
    fraction = available / total if total else None
    for level, kib_floor, frac_floor in _PRESSURE_TIERS:
        if available < kib_floor or (fraction is not None and fraction < frac_floor):
            return level
    return "ok"


def _read_state_files(home: Optional[Path]) -> tuple:
    """``(heartbeat, sentinel)`` dicts, each ``None`` when unreadable."""
    try:
        from gateway.lifecycle_ledger import _read_json, get_lifecycle_sentinel_path
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        return _read_json(get_loop_heartbeat_path(home)), _read_json(get_lifecycle_sentinel_path(home))
    except Exception:
        return None, None


def _prior_watchdog_flags(sentinel: dict[str, Any]) -> tuple[bool, bool]:
    """``(watchdog_exit, memory_starved)`` for the previous life, from the ``prior_exit_reason`` /
    ``prior_host_memory_pressure`` keys the lifecycle ledger stamps at boot (#148527)."""
    from gateway.status import is_watchdog_exit_reason

    watchdog = is_watchdog_exit_reason(sentinel.get("prior_exit_reason"))
    return watchdog, watchdog and sentinel.get("prior_host_memory_pressure") == PRESSURE_CRITICAL


def collect_memory_status(
    home: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """``memory`` block for ``/api/status``; ``home`` scopes to a profile (``None`` =
    active), ``now`` is injectable.  Never raises — a down gateway or corrupt files
    yield ``pressure="unknown"`` plus whatever fields could be recovered."""
    moment = now or datetime.now(UTC)
    status: dict[str, Any] = {
        "pressure": "unknown", "gateway_rss_mb": None, "system_total_mb": None, "system_available_mb": None,
        "swap_used_mb": None, "sampled_at": None, "last_boot_unclean": False, "last_boot_suspected_oom": False,
        # The previous life was killed by a watchdog / and the host was critically short of memory then.
        "last_boot_watchdog_exit": False, "last_boot_memory_starved": False,
        # The previous life died uncleanly and, between its last heartbeat and this boot, the kernel
        # OOM-killed a process in the gateway's cgroup ("cgroup") or on the host ("host"); else None.
        "last_boot_oom_kill": None,
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
                    status["pressure"] = classify_pressure(mem.get("mem_available_kib"), mem.get("mem_total_kib"))

    if sentinel:
        status["last_boot_unclean"] = bool(sentinel.get("prior_unclean_exit"))
        status["last_boot_suspected_oom"] = bool(sentinel.get("prior_suspected_oom"))
        scope = sentinel.get("prior_oom_kill_scope")
        status["last_boot_oom_kill"] = scope if sentinel.get("prior_oom_kill_seen") is True and scope in (
            "cgroup", "host") else None
        status["last_boot_watchdog_exit"], status["last_boot_memory_starved"] = _prior_watchdog_flags(sentinel)
        started_at = sentinel.get("started_at")
        status["boot_id"] = started_at if isinstance(started_at, str) and started_at else None

    return status
