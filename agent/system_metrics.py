"""Live host telemetry for dashboards: CPU/memory/disk/network via ``psutil``, SoC sensors via ``hermes_platform``.

Rates (bytes/s, CPU %) are deltas against the previous read, so one process-wide sampler owns the
counters. Pollers share it through a short TTL cache: two dashboards polling at 1 Hz see the same
window instead of halving each other's. Every section degrades to empty/``None`` independently;
:func:`read_system_metrics` always answers.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from hermes_platform.host import facts
from hermes_platform.sensors import apple_silicon

_CACHE_TTL_SECONDS = 1.0
_MAX_VOLUMES = 8
_MIN_VOLUME_BYTES = 1 << 30  # skips macOS xART/iSCPreboot/Hardware helper containers
_lock = threading.Lock()
_sampler: "_Sampler | None" = None
_cache: tuple[float, dict[str, Any]] | None = None


def _rate(now: float, prev: float | None, dt: float) -> float | None:
    if prev is None or dt <= 0 or now < prev:  # counter reset (interface bounce, wrap)
        return None
    return (now - prev) / dt


def _domain(d: apple_silicon.Domain) -> dict[str, Any]:
    return {"name": d.name, "kind": d.kind, "active": round(d.active, 4),
            "freq_mhz": None if d.freq_mhz is None else round(d.freq_mhz, 1)}


def _volumes(psutil) -> list[dict[str, Any]]:
    """One row per filesystem. APFS volumes in a container share free space, so ``(total, free)``
    collapses the container's mounts to its first (shortest) mount and container-level usage."""
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for part in sorted(psutil.disk_partitions(all=False), key=lambda p: len(p.mountpoint)):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        if usage.total < _MIN_VOLUME_BYTES or (usage.total, usage.free) in rows:
            continue
        used = usage.total - usage.free
        rows[(usage.total, usage.free)] = {"mount": part.mountpoint, "fstype": part.fstype, "total": usage.total,
                                           "used": used, "percent": round(100 * used / usage.total, 1)}
    return sorted(rows.values(), key=lambda r: r["total"], reverse=True)[:_MAX_VOLUMES]


def _linux_temps(psutil) -> list[dict[str, Any]]:
    read = getattr(psutil, "sensors_temperatures", None)
    if read is None:
        return []
    try:
        chips = read() or {}
    except (OSError, RuntimeError):
        return []
    return [{"name": f"{chip}/{t.label or i}", "celsius": round(t.current, 1)}
            for chip, temps in sorted(chips.items()) for i, t in enumerate(temps) if t.current]


class _Sampler:
    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())
        self._soc = apple_silicon.open_sampler()
        self._prev_t: float | None = None
        self._prev_disk: tuple[int, int] | None = None
        self._prev_net: tuple[int, int] | None = None
        psutil.cpu_percent(percpu=True)  # prime: the first interval=None read is meaningless
        self._proc.cpu_percent()

    def read(self) -> dict[str, Any]:
        ps = self._psutil
        now = time.monotonic()
        dt = 0.0 if self._prev_t is None else now - self._prev_t
        self._prev_t = now

        per_core = ps.cpu_percent(percpu=True)
        vm, swap = ps.virtual_memory(), ps.swap_memory()
        load = os.getloadavg() if hasattr(os, "getloadavg") else None

        disk = ps.disk_io_counters()
        disk_now = (disk.read_bytes, disk.write_bytes) if disk else None
        net = ps.net_io_counters()
        net_now = (net.bytes_recv, net.bytes_sent) if net else None
        prev_disk, prev_net = self._prev_disk, self._prev_net
        self._prev_disk, self._prev_net = disk_now, net_now

        # The sampler primed at construction; a first-frame delta spans only milliseconds.
        sampler = self._soc
        soc = sampler.sample() if sampler and dt else None
        with self._proc.oneshot():
            process = {"pid": self._proc.pid, "rss": self._proc.memory_info().rss,
                       "cpu_percent": self._proc.cpu_percent(), "threads": self._proc.num_threads()}

        gpus: list[dict[str, Any]] = []
        if sampler and soc and soc.gpu:
            gpus.append({**_domain(soc.gpu), "name": "Apple GPU", "cores": sampler.gpu_cores,
                         "power_w": soc.power_w.get("gpu")})
        temps = ([{"name": k, "celsius": round(v, 1)} for k, v in soc.temps_c.items()]
                 if soc else _linux_temps(ps))
        return {
            "ts": time.time(),
            "interval_s": round(dt, 3) if dt else None,
            "host": {"os": facts.os_family(), "arch": facts.native_arch(), "cpu_model": facts.cpu_model(),
                     "boot_time": ps.boot_time(), "uptime_s": max(0.0, time.time() - ps.boot_time())},
            "cpu": {
                "percent": round(sum(per_core) / len(per_core), 1) if per_core else 0.0,
                "per_core": per_core,
                "count_logical": ps.cpu_count(logical=True),
                "count_physical": ps.cpu_count(logical=False),
                "load_avg": list(load) if load else None,
                "clusters": [_domain(d) for d in soc.clusters] if soc else [],
                "cores": [_domain(d) for d in soc.cores] if soc else [],
            },
            "gpus": gpus,
            "memory": {"total": vm.total, "used": vm.used, "available": vm.available, "percent": vm.percent,
                       "swap_total": swap.total, "swap_used": swap.used},
            "power_w": {k: round(v, 3) for k, v in soc.power_w.items()} if soc else {},
            "temps": temps,
            "disk": {
                "read_bps": _rate(disk_now[0], prev_disk and prev_disk[0], dt) if disk_now else None,
                "write_bps": _rate(disk_now[1], prev_disk and prev_disk[1], dt) if disk_now else None,
                "volumes": _volumes(ps),
            },
            "net": {
                "rx_bps": _rate(net_now[0], prev_net and prev_net[0], dt) if net_now else None,
                "tx_bps": _rate(net_now[1], prev_net and prev_net[1], dt) if net_now else None,
                "rx_total": net_now[0] if net_now else None,
                "tx_total": net_now[1] if net_now else None,
            },
            "process": process,
        }


def read_system_metrics(use_cache: bool = True) -> dict[str, Any]:
    """One telemetry frame; readings younger than the TTL are shared between callers."""
    global _sampler, _cache
    with _lock:
        if use_cache and _cache is not None and time.monotonic() - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]
        if _sampler is None:
            _sampler = _Sampler()
        frame = _sampler.read()
        _cache = (time.monotonic(), frame)
        return frame


def reset() -> None:
    """Drop the sampler and cache (tests)."""
    global _sampler, _cache
    with _lock:
        _sampler, _cache = None, None
