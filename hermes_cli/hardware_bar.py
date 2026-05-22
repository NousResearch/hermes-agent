"""Live hardware stats bar for Hermes CLI.

Polls CPU, RAM, GPU, battery via psutil and renders a compact
one-line status bar below the main status bar. Designed to be
skin-aware — uses the active skin's color tokens.
"""

import logging
import os
import platform
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Hardware Polling ─────────────────────────────────────────────────

class HardwareMonitor:
    """Background thread that polls system stats at a fixed interval."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stats: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_available = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        # Detect GPU once
        self._detect_gpu()

    def stop(self):
        self._running = False

    def get_stats(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._stats)

    def _detect_gpu(self):
        """Check if nvidia-smi is available for GPU monitoring."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0 and r.stdout.strip():
                self._gpu_available = True
        except Exception:
            self._gpu_available = False

    def _poll_loop(self):
        import psutil
        # Prime the CPU percent calculator
        psutil.cpu_percent(interval=None)
        time.sleep(0.5)

        while self._running:
            try:
                stats = self._poll_once(psutil)
                with self._lock:
                    self._stats = stats
            except Exception as e:
                logger.debug("Hardware poll error: %s", e)
            time.sleep(self.interval)

    def _poll_once(self, psutil) -> Dict[str, str]:
        stats = {}

        # CPU
        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        freq_str = f"{cpu_freq.current / 1000:.1f}GHz" if cpu_freq else ""
        stats["cpu"] = f"{cpu_pct:.0f}%"
        if freq_str:
            stats["cpu_freq"] = freq_str

        # RAM
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        stats["ram"] = f"{used_gb:.1f}/{total_gb:.0f}G"
        stats["ram_pct"] = f"{mem.percent:.0f}%"

        # Battery
        try:
            bat = psutil.sensors_battery()
            if bat:
                plugged = bat.power_plugged
                stats["bat"] = f"{bat.percent}%"
                stats["bat_status"] = "charging" if plugged else "discharging"
                if plugged and bat.percent >= 100:
                    stats["bat_status"] = "full"
        except Exception:
            pass

        # GPU (nvidia-smi)
        if self._gpu_available:
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                if r.returncode == 0:
                    parts = [p.strip() for p in r.stdout.strip().split(",")]
                    if len(parts) >= 4:
                        stats["gpu"] = f"{parts[0]}%"
                        stats["gpu_mem"] = f"{parts[1]}/{parts[2]}M"
                        stats["gpu_temp"] = f"{parts[3]}°C"
            except Exception:
                pass

        # Disk I/O (optional, lightweight)
        try:
            disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
            stats["disk"] = f"{disk.percent:.0f}%"
        except Exception:
            pass

        return stats


# ─── Singleton ────────────────────────────────────────────────────────

_monitor: Optional[HardwareMonitor] = None


def get_monitor() -> HardwareMonitor:
    global _monitor
    if _monitor is None:
        _monitor = HardwareMonitor(interval=2.0)
    return _monitor


# ─── Fragment Builder (for prompt_toolkit) ────────────────────────────

def build_hardware_bar_fragments(skin_colors: Dict[str, str] = None) -> List[Tuple[str, str]]:
    """Build prompt_toolkit fragments for the hardware stats bar.

    Uses skin colors if provided, otherwise falls back to class names.
    """
    monitor = get_monitor()
    stats = monitor.get_stats()

    if not stats:
        return []

    # Color helpers
    def c_accent(s):
        if skin_colors and skin_colors.get("ui_accent"):
            return f"[{skin_colors['ui_accent']}]{s}[/]"
        return s

    def c_text(s):
        if skin_colors and skin_colors.get("banner_text"):
            return f"[{skin_colors['banner_text']}]{s}[/]"
        return s

    def c_dim(s):
        if skin_colors and skin_colors.get("banner_dim"):
            return f"[{skin_colors['banner_dim']}]{s}[/]"
        return s

    def c_good(s):
        if skin_colors and skin_colors.get("status_bar_good"):
            return f"[{skin_colors['status_bar_good']}]{s}[/]"
        return s

    def c_warn(s):
        if skin_colors and skin_colors.get("status_bar_warn"):
            return f"[{skin_colors['status_bar_warn']}]{s}[/]"
        return s

    def c_bad(s):
        if skin_colors and skin_colors.get("status_bar_bad"):
            return f"[{skin_colors['status_bar_bad']}]{s}[/]"
        return s

    # Build parts using class-based styling (works with any skin)
    frags = [("class:status-bar", " ")]

    # CPU
    cpu_pct = float(stats.get("cpu", "0").rstrip("%"))
    cpu_style = "class:status-bar-good" if cpu_pct < 60 else (
        "class:status-bar-warn" if cpu_pct < 85 else "class:status-bar-bad"
    )
    frags.append(("class:status-bar-strong", "CPU"))
    frags.append((cpu_style, f" {stats['cpu']} "))

    # RAM
    ram_pct = float(stats.get("ram_pct", "0").rstrip("%"))
    ram_style = "class:status-bar-good" if ram_pct < 70 else (
        "class:status-bar-warn" if ram_pct < 90 else "class:status-bar-bad"
    )
    frags.append(("class:status-bar-dim", "│"))
    frags.append(("class:status-bar-strong", " RAM"))
    frags.append((ram_style, f" {stats['ram']} "))

    # GPU (if available)
    if "gpu" in stats:
        gpu_pct = float(stats["gpu"].rstrip("%"))
        gpu_style = "class:status-bar-good" if gpu_pct < 60 else (
            "class:status-bar-warn" if gpu_pct < 85 else "class:status-bar-bad"
        )
        frags.append(("class:status-bar-dim", "│"))
        frags.append(("class:status-bar-strong", " GPU"))
        frags.append((gpu_style, f" {stats['gpu']}"))
        if "gpu_temp" in stats:
            temp = float(stats["gpu_temp"].rstrip("°C"))
            temp_style = "class:status-bar-good" if temp < 70 else (
                "class:status-bar-warn" if temp < 85 else "class:status-bar-bad"
            )
            frags.append((temp_style, f" {stats['gpu_temp']}"))

    # Battery
    if "bat" in stats:
        bat_pct = float(stats["bat"].rstrip("%"))
        bat_status = stats.get("bat_status", "")
        if bat_status == "full":
            bat_style = "class:status-bar-good"
            bat_icon = "⚡"
        elif bat_status == "charging":
            bat_style = "class:status-bar-good"
            bat_icon = "⚡"
        elif bat_pct < 20:
            bat_style = "class:status-bar-bad"
            bat_icon = "🪫"
        elif bat_pct < 50:
            bat_style = "class:status-bar-warn"
            bat_icon = "🔋"
        else:
            bat_style = "class:status-bar-good"
            bat_icon = "🔋"
        frags.append(("class:status-bar-dim", "│"))
        frags.append((bat_style, f" {bat_icon} {stats['bat']}"))

    # Disk
    if "disk" in stats:
        disk_pct = float(stats["disk"].rstrip("%"))
        disk_style = "class:status-bar-good" if disk_pct < 80 else (
            "class:status-bar-warn" if disk_pct < 95 else "class:status-bar-bad"
        )
        frags.append(("class:status-bar-dim", "│"))
        frags.append(("class:status-bar-strong", " DISK"))
        frags.append((disk_style, f" {stats['disk']}"))

    frags.append(("class:status-bar", " "))
    return frags
