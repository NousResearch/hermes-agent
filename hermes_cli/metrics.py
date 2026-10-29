"""`/metrics` command — VPS resource snapshot.

Reads CPU/RAM/disk + Docker container count + uptime. Returns a one-shot
plain-text block formatted for Telegram (no Markdown tables, monospace-safe).
Stdlib + psutil only. Safe to call from any host; degrades gracefully if
Docker isn't installed (returns '?' for container count).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # ponytail: psutil is the dep that makes this trivial; surface clearly if missing
    psutil = None  # type: ignore[assignment]


def _read_uptime_seconds() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _container_count() -> int | None:
    """Count Docker containers in 'running' state. None if Docker unreachable."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-q"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return len([line for line in out.stdout.splitlines() if line.strip()])


def _fmt_uptime(seconds: float) -> str:
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _bar(pct: float) -> str:
    """Compact ASCII bar: ████░░░░ 50% (no emoji to stay safe in monospace)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(pct // 10)
    return "█" * filled + "░" * (10 - filled)


def format_metrics() -> str:
    """Build the /metrics reply string. Sync, safe to run via asyncio.to_thread."""
    if psutil is None:
        return "metrics: psutil no instalado en el host (`pip install psutil`)."

    cpu_pct = psutil.cpu_percent(interval=0.4)
    vm = psutil.virtual_memory()
    du = shutil.disk_usage("/")

    ram_pct = vm.percent
    disk_pct = (du.used / du.total) * 100.0 if du.total else 0.0  # ponytail: shutil.usage lacks .percent on py<3.12

    containers = _container_count()
    container_str = str(containers) if containers is not None else "?"

    uptime_s = _read_uptime_seconds()
    uptime_str = _fmt_uptime(uptime_s) if uptime_s is not None else "?"

    lines = [
        "🖥  VPS metrics",
        f"CPU    {_bar(cpu_pct)} {cpu_pct:5.1f}%",
        f"RAM    {_bar(ram_pct)} {ram_pct:5.1f}%   {vm.used // (1024**3):>3}/{vm.total // (1024**3)} GiB",
        f"DISK   {_bar(disk_pct)} {disk_pct:5.1f}%   {du.used // (1024**3):>3}/{du.total // (1024**3)} GiB",
        f"DOCKER {container_str} contenedores corriendo",
        f"UP     {uptime_str}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # ponytail: self-check, prints to stdout so `python -m hermes_cli.metrics` verifies the read path
    print(format_metrics())
