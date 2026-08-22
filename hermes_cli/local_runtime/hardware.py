"""Live hardware budget probe.

Budget-source rule: discrete cards may trust the device query (measured
honest within rounding); unified-memory devices must budget from OS free
physical memory minus headroom — their device queries have been observed
off by 3x in both directions. The probe classifies the device and
constructs the right HardwareBudget for the estimator.
"""

from __future__ import annotations

import logging
import subprocess

from hermes_cli.local_runtime.estimator import HardwareBudget

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# Fit margin per design: max(512 MiB, ~7% of device) — the upstream flat
# 1 GiB is regressive on 8 GB cards and generous on 24 GB ones.
_MARGIN_FLOOR = 512 << 20
_MARGIN_FRACTION = 0.07
# UMA headroom: on unified-memory machines (Apple Silicon) the model shares
# physical memory with the OS and every app, so budget from RAM minus this
# fraction.
_UMA_HEADROOM_FRACTION = 0.20


def _ram_bytes() -> tuple[int, int]:
    """(total, available) physical memory, cross-platform stdlib."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys, stat.ullAvailPhys
    except (AttributeError, OSError):
        pass
    # POSIX
    try:
        page = int(subprocess.run(["getconf", "PAGE_SIZE"], capture_output=True,
                                  text=True, timeout=5).stdout or 4096)
        total = int(subprocess.run(["getconf", "_PHYS_PAGES"], capture_output=True,
                                   text=True, timeout=5).stdout or 0) * page
        avail = total // 2  # conservative when _AVPHYS is unavailable
        try:
            avail = int(subprocess.run(["getconf", "_AVPHYS_PAGES"],
                                       capture_output=True, text=True,
                                       timeout=5).stdout or 0) * page or avail
        except (OSError, ValueError):
            pass
        return total, avail
    except (OSError, ValueError):
        return 0, 0


def _nvidia_vram() -> tuple[int, int] | None:
    """(total, free) MiB->bytes from nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        total_mib, free_mib = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return total_mib << 20, free_mib << 20
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def probe_budget(*, planning: bool = False) -> HardwareBudget:
    """Construct the budget per the source rules above.

    ``planning=False`` (default): LIVE budget — free VRAM right now. The
    right input for launch-time fit decisions and growth re-grants.

    ``planning=True``: CAPACITY budget — what this machine can run once
    the runtime manages placement (total device memory minus the margin).
    The right input for catalog pricing and quant selection: pricing
    against live-free while a model is already loaded made every row read
    'larger than your GPU memory' and degraded quant picks to Q2 on a
    32 GiB card. The managed server
    unloads/relaunches models itself, so at load time the capacity is
    genuinely available.
    """
    ram_total, ram_avail = _ram_bytes()
    vram = _nvidia_vram()

    if vram is None:
        # No NVIDIA GPU visible: Metal/Vulkan/CPU paths budget from RAM as
        # UMA (Apple Silicon) — conservative for discrete AMD until a
        # vendor probe lands (E3 hardware).
        base = ram_total if planning else ram_avail
        usable = max(0, int(base * (1 - _UMA_HEADROOM_FRACTION)))
        return HardwareBudget(usable_vram_bytes=usable,
                              total_device_bytes=ram_total,
                              ram_available_bytes=0, uma=True)

    total, free = vram
    margin = max(_MARGIN_FLOOR, int(total * _MARGIN_FRACTION))
    base = total if planning else free
    return HardwareBudget(usable_vram_bytes=max(0, base - margin),
                          total_device_bytes=total,
                          ram_available_bytes=ram_avail if not planning else ram_total,
                          uma=False)
