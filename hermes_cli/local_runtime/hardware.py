"""Live hardware budget probe.

Budget-source rule: discrete cards may trust the device query (measured
honest within rounding); unified-memory devices must budget from OS free
physical memory minus headroom — their device queries have been observed
off by 3x in both directions. The probe classifies the device and
constructs the right HardwareBudget for the estimator.

Vendor probe quirk (WDDM carve-out): on unified-memory NVIDIA devices
under Windows, nvidia-smi answers from the legacy dedicated-VRAM
carve-out — a fraction of the pool the CUDA allocator actually
addresses uniformly at full bandwidth. The CUDA driver API is
the tiebreaker: cuDeviceGetAttribute(INTEGRATED) is the vendor's own
declaration and always wins — 1 budgets unified, 0 stays discrete no
matter what any other number says. Only when the driver API is
unreachable does the engine's --list-devices view apply, and then only
behind two independent conditions no discrete card can meet.

Every probe here must work under a stripped PATH — gateway and service
sessions don't inherit the interactive environment. nvcuda/libcuda load
through the system loader (PATH plays no part), so classification never
depends on PATH; nvidia-smi resolves through an explicit candidate
ladder (PATH first, then the driver's known install locations) and its
absence only softens the live number, never the verdict.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hermes_cli.local_runtime.estimator import HardwareBudget

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# Reserve carved off the card before any grant: the desktop's own
# co-residents (compositor, browser, Electron) measure ~2-2.5 GiB on a
# working machine, and a window granted into that space demotes silently
# under WDDM. 7% covers big cards; the 2 GiB floor is what the margin's
# old 512 MiB floor failed to cover in practice (a 221K grant measured
# 31.9/32.6 GiB with the desktop running — 'fits' by the math, demoted
# in reality). Small cards give up window to this; spill mode is their
# path to big models regardless.
_MARGIN_FLOOR = 2 << 30
_MARGIN_FRACTION = 0.09
# UMA headroom: on unified-memory machines (Apple Silicon, unified-memory
# NVIDIA) the model shares physical memory with the OS and every app, so
# budget from RAM minus this fraction.
_UMA_HEADROOM_FRACTION = 0.20

# Engine-fallback gates for the unified-pool quirk — BOTH must hold, and
# no discrete card can meet either: (1) the allocator's pool exceeds the
# smi report by well past rounding/ECC slack (discrete cards agree within
# ~2%; carve-out disagreement runs to whole multiples), and (2) the pool is
# system-RAM-sized — a workstation card in a RAM-matched box fails (1)
# because its smi and allocator AGREE, and a big discrete card in a
# bigger box fails (2). The driver's INTEGRATED attribute, when
# readable, bypasses both gates in whichever direction it points.
_POOL_DISAGREEMENT_FACTOR = 1.5
_POOL_RAM_FRACTION = 0.75

# cuDeviceGetAttribute enum: device is integrated with host memory.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18

# One probe per process once a device answers (silicon doesn't change);
# a miss retries after this long so a runtime installed mid-session gets
# picked up by the engine fallback.
_POOL_NEGATIVE_TTL_S = 60.0
_pool_probe_cache: tuple[float, "tuple[int, bool | None] | None"] | None = None

# '  CUDA0: NVIDIA Example Device (1234-core Example GPU) (46464 MiB, 46284 MiB free)'
# — greedy .* pins the LAST parenthesized group, so device names carrying
# their own parentheses parse correctly.
_DEVICE_LINE_RE = re.compile(r"CUDA\d+:.*\((\d+)\s*MiB,\s*\d+\s*MiB free\)\s*$")


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
    if sys.platform == "darwin":
        # macOS getconf has no _PHYS_PAGES/_AVPHYS_PAGES (exit 64, "no such
        # configuration parameter") — the POSIX branch below returns (0, 0)
        # and every model reads unavailable. sysctl is the platform truth.
        try:
            total = int(subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5).stdout.strip() or 0)
            if total <= 0:
                return 0, 0
            avail = total // 2  # conservative fallback
            try:
                out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True,
                                     text=True, timeout=5).stdout
                page_m = re.search(r"page size of (\d+)", out)
                page = int(page_m.group(1)) if page_m else 16384
                pages = 0
                # free + inactive + purgeable ≈ reclaimable-on-demand; the
                # speculative pool is dropped by the OS under pressure too.
                for key in ("Pages free", "Pages inactive", "Pages purgeable",
                            "Pages speculative"):
                    m = re.search(rf"{key}:\s+(\d+)\.", out)
                    if m:
                        pages += int(m.group(1))
                if pages > 0:
                    avail = pages * page
            except (OSError, ValueError):
                pass
            return total, avail
        except (OSError, ValueError):
            return 0, 0
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


# nvidia-smi lives at a fixed path under the driver install; PATH presence
# varies by session type (services and gateways often run with a minimal
# environment) and by driver generation (legacy NVSMI dir was never on
# PATH). Resolution result is cached: the driver doesn't move mid-process.
_smi_path_cache: "tuple[str | None] | None" = None


def _nvidia_smi_path() -> str | None:
    """Absolute path to nvidia-smi, or None. PATH first (respects user
    overrides), then the driver's known install locations on Windows;
    on Linux/WSL the PATH lookup is the whole ladder."""
    global _smi_path_cache
    if _smi_path_cache is not None:
        return _smi_path_cache[0]
    found = shutil.which("nvidia-smi")
    if found is None and os.name == "nt":
        windir = os.environ.get("SystemRoot", r"C:\Windows")
        for candidate in (
            # DCH drivers (every modern install) place it in System32.
            Path(windir) / "System32" / "nvidia-smi.exe",
            # Legacy standalone drivers used NVSMI, never on PATH.
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ):
            if candidate.exists():
                found = str(candidate)
                break
    _smi_path_cache = (found,)
    return found


def _nvidia_vram() -> tuple[int, int] | None:
    """(total, free) MiB->bytes from nvidia-smi, or None."""
    exe = _nvidia_smi_path()
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        total_mib, free_mib = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return total_mib << 20, free_mib << 20
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _cuda_driver_pool() -> "tuple[int, bool | None] | None":
    """(allocator_total_bytes, integrated_or_None) from the CUDA driver
    API, or None when unreachable. ctypes against the driver's own DLL/SO
    — no toolkit, no subprocess, ~ms. INTEGRATED is the vendor's own
    unified-memory declaration; total is the pool the allocator will
    actually hand out (on carve-out devices, several times what
    nvidia-smi reports)."""
    import ctypes

    for name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return None
    try:
        if cuda.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None
        total = ctypes.c_size_t()
        getter = getattr(cuda, "cuDeviceTotalMem_v2", None) or cuda.cuDeviceTotalMem
        if getter(ctypes.byref(total), dev) != 0 or total.value <= 0:
            return None
        integrated: bool | None = None
        attr = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(
                ctypes.byref(attr), _CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) == 0:
            integrated = bool(attr.value)
        return total.value, integrated
    except (OSError, AttributeError):
        return None


def _engine_device_pool() -> "tuple[int, bool | None] | None":
    """(engine_total_bytes, None) from the installed runtime's own
    --list-devices, or None. The fallback truth source when the driver
    API is unreachable: asks the exact binary that will do the
    allocating. Carries no integrated verdict — callers must gate it."""
    try:
        from hermes_cli.local_runtime.binaries import (
            installed_tags,
            runtimes_root,
            server_binary,
        )

        tags = installed_tags()
        if not tags:
            return None
        tag_dir = runtimes_root() / tags[0]
        backend_dirs = [d for d in tag_dir.iterdir() if d.is_dir()]
        if not backend_dirs:
            return None
        exe = server_binary(backend_dirs[0])
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                             text=True, timeout=30, cwd=str(exe.parent))
        if out.returncode != 0:
            return None
        for line in (out.stdout + out.stderr).splitlines():
            m = _DEVICE_LINE_RE.search(line)
            if m:
                return int(m.group(1)) << 20, None
        return None
    except Exception:  # noqa: BLE001 — a probe miss must never block budgeting
        return None


def _device_pool_view() -> "tuple[int, bool | None] | None":
    """Best available allocator-side view, cached: a hit is permanent for
    the process, a miss retries after a short TTL (the engine binary can
    appear mid-session via a pane install)."""
    global _pool_probe_cache
    now = time.monotonic()
    if _pool_probe_cache is not None:
        stamp, view = _pool_probe_cache
        if view is not None or now - stamp < _POOL_NEGATIVE_TTL_S:
            return view
    view = _cuda_driver_pool() or _engine_device_pool()
    _pool_probe_cache = (now, view)
    return view


def _unified_pool_bytes(smi_total: int, ram_total: int) -> int | None:
    """The real pool size when this NVIDIA device is unified memory behind
    a WDDM carve-out, else None (trust nvidia-smi as ever).

    The driver's INTEGRATED attribute decides when readable — in BOTH
    directions (0 pins discrete even if the numbers look weird; a driver
    that declares integrated is believed even at modest pool sizes). Only
    an attribute-less view (engine fallback) needs the two numeric gates;
    both must hold and no discrete card meets either.
    """
    view = _device_pool_view()
    if view is None:
        return None
    pool, integrated = view
    if integrated is False:
        return None
    if integrated is True:
        return pool
    if (smi_total > 0 and pool >= int(smi_total * _POOL_DISAGREEMENT_FACTOR)
            and ram_total > 0 and pool >= int(ram_total * _POOL_RAM_FRACTION)):
        return pool
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

    # Unified-memory NVIDIA: the CUDA allocator pool is the real
    # capacity. Classification comes from the driver API/engine — it
    # must not require nvidia-smi (stripped-PATH sessions lose smi but
    # nvcuda loads via the system loader regardless). Crossing the
    # carve-out costs nothing (effective bandwidth is flat through the
    # boundary; smi's used/total merely saturate at it) — the carve-out
    # is an OS accounting knob, not a GPU limit. Deliberately NOT
    # clamped to OS RAM: carved-out memory is invisible to
    # GlobalMemoryStatusEx (the OS reports correspondingly less total
    # RAM), so a RAM clamp would throw away exactly the carved capacity.
    unified = _unified_pool_bytes(vram[0] if vram else 0, ram_total)
    if unified is not None:
        logger.info(
            "unified-memory NVIDIA device: allocator pool %.1f GiB "
            "(nvidia-smi carve-out: %s); budgeting from the pool",
            unified / _GIB,
            f"{vram[0] / _GIB:.1f} GiB" if vram else "unavailable")
        if planning:
            base = unified
        else:
            # Live: dedicated-free plus what the OS can still give. smi's
            # free saturates at the carve-out so this under-counts a bit —
            # the safe direction (the pool edge is a measured soft cliff:
            # decode collapses ~3.5x when concurrent demand hits it).
            # Without smi, OS-available alone is the honest floor.
            live = (vram[1] + ram_avail) if vram else ram_avail
            base = min(unified, live)
        usable = max(0, int(base * (1 - _UMA_HEADROOM_FRACTION)))
        return HardwareBudget(usable_vram_bytes=usable,
                              total_device_bytes=unified,
                              ram_available_bytes=0, uma=True)

    if vram is None:
        # No NVIDIA device visible: Metal/Vulkan/CPU paths budget from RAM
        # as UMA (Apple Silicon) — conservative for discrete AMD until a
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
