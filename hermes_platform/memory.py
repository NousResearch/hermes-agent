"""How much memory is really left for a process on this host (Linux; None elsewhere).

Headroom is the tighter of cgroup v2 headroom (``memory.max``/``memory.high`` minus
``memory.current``, with reclaimable ``inactive_file`` page cache credited back) along the
process's cgroup chain, and ``/proc/meminfo`` ``MemAvailable``. Read live on every call: memory
is a moving fact, so nothing here is cached (callers that poll throttle themselves).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class MemoryReading:
    available_bytes: int
    ceiling_bytes: Optional[int]
    source: str  # "cgroup" when a cgroup limit is the tighter bound, else "meminfo"


def _read_int(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _inactive_file(directory: Path) -> int:
    try:
        lines = (directory / "memory.stat").read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return 0
    for line in lines:
        key, _, value = line.partition(" ")
        if key == "inactive_file" and value.strip().isdigit():
            return int(value)
    return 0


def _cgroup_of(pid: Optional[int]) -> Optional[str]:
    try:
        text = Path(f"/proc/{pid or 'self'}/cgroup").read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or "/"
    return None


def cgroup_headroom(rel_path: str, root: Path = _CGROUP_ROOT) -> Optional[tuple[int, int]]:
    """``(headroom, limit)`` for the tightest finite memory limit from ``rel_path`` up to the
    root, or None when no level is limited. A limit on an ancestor (a systemd slice, the
    container's namespace root) caps this process just as hard as one on its own cgroup."""
    parts = [p for p in rel_path.strip("/").split("/") if p]
    best: Optional[tuple[int, int]] = None
    for depth in range(len(parts), -1, -1):
        directory = root.joinpath(*parts[:depth])
        limits = [v for v in (_read_int(directory / "memory.max"), _read_int(directory / "memory.high"))
                  if v is not None and 0 < v < (1 << 62)]
        current = _read_int(directory / "memory.current")
        if not limits or current is None:
            continue
        limit = min(limits)
        headroom = limit - max(0, current - _inactive_file(directory))
        if best is None or headroom < best[0]:
            best = (headroom, limit)
    return best


def _meminfo() -> tuple[Optional[int], Optional[int]]:
    """``(MemAvailable, MemTotal)`` in bytes."""
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None, None
    for line in lines:
        key, _, rest = line.partition(":")
        fields = rest.split()
        if key in ("MemAvailable", "MemTotal") and fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024
    return values.get("MemAvailable"), values.get("MemTotal")


def read_memory(pid: Optional[int] = None) -> Optional[MemoryReading]:
    """Memory left for ``pid`` (default: this process); None when unknowable (non-Linux)."""
    if sys.platform != "linux":
        return None
    available, total = _meminfo()
    rel = _cgroup_of(pid)
    capped = cgroup_headroom(rel) if rel is not None else None
    if capped is not None and (available is None or capped[0] <= available):
        return MemoryReading(max(0, capped[0]), capped[1], "cgroup")
    if available is None:
        return None
    return MemoryReading(available, capped[1] if capped else total, "meminfo")
