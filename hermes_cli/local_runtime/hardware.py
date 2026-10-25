"""Live hardware budget probe.

Budget-source rule: discrete cards may trust the device query (measured honest within rounding);
unified-memory devices must budget from OS free physical memory minus headroom — their device
queries have been observed off by 3x in both directions. Every probe here must work under a
stripped PATH — gateway and service sessions don't inherit the interactive environment.
"""

from __future__ import annotations

from contextlib import suppress
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from hermes_cli.local_runtime.devices import GGML_DEVICE_GPU
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_platform.host.facts import gpu_class

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# Reserve carved off the card before any grant: the desktop's own co-residents (compositor,
# browser, Electron) measure ~2-2.5 GiB, and a window granted into that space demotes silently
# under WDDM. 7% covers big cards; the 2 GiB floor is what a 512 MiB floor failed to cover (a
# 221K grant measured 31.9/32.6 GiB with the desktop running — 'fits' by the math, demoted in
# reality). Small cards give up window to this; spill mode is their path to big models.
_MARGIN_FLOOR = 2 << 30
_MARGIN_FRACTION = 0.09
# Room left on top of what other programs hold when a model launches: a browser tab or a chat app
# can take another GiB between launch and the next request.
_LAUNCH_HEADROOM = 1 << 30
# UMA headroom: on unified-memory machines the model shares physical memory with the OS and every
# app, so budget from RAM minus this fraction.
_UMA_HEADROOM_FRACTION = 0.20
# llama-server's prompt cache: its own default size, and the smallest worth keeping. The cache holds
# conversations the server switched away from (a background job, an earlier chat); without it each
# switch re-reads the whole conversation.
_PROMPT_CACHE_MAX = 8 << 30
_PROMPT_CACHE_MIN = 512 << 20

# Engine-fallback gates for the unified-pool quirk — BOTH must hold, and no discrete card can
# meet either: (1) the allocator's pool exceeds the smi report by well past rounding/ECC slack
# (discrete cards agree within ~2%; carve-out disagreement runs to whole multiples), and (2) the
# pool is system-RAM-sized. The driver's INTEGRATED attribute, when readable, bypasses both gates
# in whichever direction it points.
_POOL_DISAGREEMENT_FACTOR = 1.5
_POOL_RAM_FRACTION = 0.75

# cuDeviceGetAttribute enum: device is integrated with host memory.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
_NVIDIA_VENDOR_ID = 0x10DE
# Engines whose devices the ggml probe reports; the CPU, CUDA and Metal builds answer elsewhere.
_GPU_ENGINE_BACKENDS = ("vulkan", "hip")

# One probe per process once a device answers (silicon doesn't change); a miss retries after this
# long so a runtime installed mid-session gets picked up by the engine fallback.
_POOL_NEGATIVE_TTL_S = 60.0
_pool_probe_cache: tuple[float, tuple[int, bool | None] | None] | None = None
_pool_probe_key: tuple | None = None

# '  CUDA0: NVIDIA Example Device (1234-core Example GPU) (46464 MiB, 46284 MiB free)'
# — greedy .* pins the LAST parenthesized group, so device names with parentheses parse.
_DEVICE_LINE_RE = re.compile(r"CUDA\d+:.*\((\d+)\s*MiB,\s*\d+\s*MiB free\)\s*$")


def _stdout(*argv: str) -> str:
    return subprocess.run(
        list(argv), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        check=False,
    ).stdout


_LINUX_RAM_KEYS = frozenset({"MemTotal", "MemAvailable", "MemFree"})


def _linux_meminfo_text() -> str | None:
    """Raw /proc/meminfo, or None when procfs cannot be read."""
    try:
        return Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None


def _linux_ram_from_meminfo(text: str) -> tuple[int, int] | None:
    """(total, available) from meminfo text, or None if it cannot be trusted.

    MemAvailable includes reclaimable page cache, which ``getconf _AVPHYS_PAGES``
    counts as used. Kernels that omit MemAvailable fall back to MemFree. Zero is
    a real available value — a missing-field check must not treat it as absent.
    """
    fields: dict[str, int] = {}
    try:
        for line in text.splitlines():
            name, separator, value = line.partition(":")
            if not separator or name not in _LINUX_RAM_KEYS:
                continue
            parts = value.split()
            if len(parts) != 2 or parts[1] != "kB":
                continue
            fields[name] = int(parts[0]) * 1024
        total = fields.get("MemTotal")
        # Key presence, not truthiness: MemAvailable 0 must not fall through to MemFree.
        if "MemAvailable" in fields:
            available = fields["MemAvailable"]
        else:
            available = fields.get("MemFree")
        if (
            total is None
            or total <= 0
            or available is None
            or not 0 <= available <= total
        ):
            return None
        return total, available
    except (AttributeError, TypeError, ValueError):
        return None


def _windows_memory_status():
    """GlobalMemoryStatusEx, or None off Windows."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
                    + [(name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")])

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return stat
    except (AttributeError, OSError):
        return None


def _ram_bytes() -> tuple[int, int]:
    """(total, available) physical memory, cross-platform stdlib."""
    stat = _windows_memory_status()
    if stat is not None:
        return stat.ullTotalPhys, stat.ullAvailPhys
    try:
        if sys.platform == "darwin":
            # macOS getconf has no _PHYS_PAGES/_AVPHYS_PAGES (exit 64) — the POSIX branch would
            # return (0, 0) and every model would read unavailable. sysctl is the platform truth.
            total = int(_stdout("/usr/sbin/sysctl", "-n", "hw.memsize").strip() or 0)
            if total <= 0:
                return 0, 0
            avail = total // 2  # conservative fallback
            with suppress(OSError, ValueError):
                out = _stdout("/usr/bin/vm_stat")
                page_m = re.search(r"page size of (\d+)", out)
                page = int(page_m.group(1)) if page_m else 16384
                # free + inactive + purgeable ≈ reclaimable-on-demand; the speculative pool is
                # dropped by the OS under pressure too.
                pages = sum(int(m.group(1)) for key in (
                    "Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
                    if (m := re.search(rf"{key}:\s+(\d+)\.", out)))
                if pages > 0:
                    avail = pages * page
            return total, avail
        if sys.platform.startswith("linux"):
            meminfo = _linux_meminfo_text()
            if meminfo is not None:
                linux_ram = _linux_ram_from_meminfo(meminfo)
                if linux_ram is not None:
                    return linux_ram
        # POSIX
        page = int(_stdout("getconf", "PAGE_SIZE") or 4096)
        total = int(_stdout("getconf", "_PHYS_PAGES") or 0) * page
        avail = total // 2  # conservative when _AVPHYS is unavailable
        with suppress(OSError, ValueError):
            avail = int(_stdout("getconf", "_AVPHYS_PAGES") or 0) * page or avail
        return total, avail
    except (OSError, ValueError):
        return 0, 0


# nvidia-smi lives at a fixed path under the driver install; PATH presence varies by session type
# (services and gateways often run minimal environments) and by driver generation (the legacy
# NVSMI dir was never on PATH). Cached: the driver doesn't move mid-process.
_smi_path_cache: tuple[str | None] | None = None


def _nvidia_smi_path() -> str | None:
    """Absolute path to nvidia-smi, or None. PATH first (respects user overrides), then the
    driver's known install locations on Windows and WSL."""
    global _smi_path_cache
    if _smi_path_cache is not None:
        return _smi_path_cache[0]
    found = shutil.which("nvidia-smi")
    if found is None and os.name == "nt":
        windir = os.environ.get("SystemRoot", r"C:\Windows")
        candidates = (
            # DCH drivers (every modern install) place it in System32.
            Path(windir) / "System32" / "nvidia-smi.exe",
            # Legacy standalone drivers used NVSMI, never on PATH.
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        )
        found = next((str(c) for c in candidates if c.exists()), None)
    elif found is None and sys.platform.startswith("linux"):
        # The Windows display driver exposes nvidia-smi here inside WSL, but does not guarantee
        # that the directory is present on PATH.
        candidate = Path("/usr/lib/wsl/lib/nvidia-smi")
        if candidate.exists():
            found = str(candidate)
    _smi_path_cache = (found,)
    return found


def _nvidia_vram() -> tuple[int, int, str, int | None] | None:
    """(total bytes, free bytes, name, optional packed PCI ID) of the SELECTED card, or None.

    Selection-aware (#134227): the card priced here is the card the spawn pins, via
    ``selected_nvidia_card()``. Single-card machines get their only card — the historical answer.
    """
    engine = _configured_engine()
    if engine is not None and engine.backend != "cuda":
        return None
    card = selected_nvidia_card()
    if card is None:
        return None
    return card["total_bytes"], card["free_bytes"], card["gpu_name"], card.get("gpu_pci_id")


# ── GPU selection (#134227): one resolver, two consumers ──────
#
# A machine can hold a display card plus a dedicated compute card. The budget probe and the
# managed server's spawn must agree on WHICH card that is, or the catalog prices a card the
# engine never touches. Everything below reads the same shared query and the same config key.


def nvidia_cards() -> list[dict]:
    """Every NVIDIA card with its explicit smi index and physical UUID; [] on probe miss."""
    query = _cached_nvidia_gpu_query()
    if query is None:
        return []
    return list(query.get("cards") or [query])


def _configured_gpu_selection() -> str:
    """Read this profile's requested choice without treating YAML integer 0 as missing."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.local_runtime.gpu_selection import normalize

    return normalize((load_config_readonly().get("local_runtime") or {}).get("gpu", "auto"))


def _effective_gpu_selection() -> str:
    """Budget the shared server's frozen choice, not a different profile's preference."""
    from hermes_cli.local_runtime.bootstrap import serving_gpu_selection

    serving = serving_gpu_selection()
    return serving if serving is not None else _configured_gpu_selection()


_cuda_visibility_cache: dict[tuple, tuple[float, list[str] | None]] = {}


def _cuda_visible_uuids() -> list[str] | None:
    from hermes_cli.local_runtime.devices import cuda_visible_uuids

    key = (os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER"))
    now = time.monotonic()
    cached = _cuda_visibility_cache.get(key)
    if cached and now - cached[0] < _GPU_QUERY_TTL_S:
        return cached[1]
    result = cuda_visible_uuids()
    _cuda_visibility_cache[key] = (now, result)
    return result


def selected_nvidia_card(selection: str | None = None) -> dict | None:
    from hermes_cli.local_runtime.gpu_selection import GpuSelectionError, choose

    choice = _effective_gpu_selection() if selection is None else selection
    env = os.environ
    mask = env.get("CUDA_VISIBLE_DEVICES", "")
    if choice == "auto" and any(re.fullmatch(r"[0-9]+", t.strip()) for t in mask.split(",")):
        visible = _cuda_visible_uuids()
        if visible is None:
            raise GpuSelectionError("CUDA visibility could not be resolved; use a GPU UUID or select a detected GPU")
        env = dict(env, CUDA_VISIBLE_DEVICES=",".join(visible))
    return choose(nvidia_cards(), choice, env=env)


def gpu_pin_env(backend: str, selection: str | None = None) -> dict[str, str]:
    """Pin CUDA by UUID, independently of CUDA/smi/PCI enumeration order.

    Automatic also pins the card it prices. An unavailable explicit choice raises:
    running unpinned would violate a reserved display/other-workload GPU.
    Vulkan/HIP use the engine's own device names (``gpu_device_args``), never smi ordinals.
    """
    if backend != "cuda":
        return {}
    card = selected_nvidia_card(selection)
    if card is None:
        # Some Windows drivers interpret an empty mask as unset. Explicitly disable
        # CUDA when an inherited visibility mask selected no known device.
        return {"CUDA_VISIBLE_DEVICES": "-1"} if nvidia_cards() else {}
    from hermes_cli.local_runtime.gpu_selection import GpuSelectionError

    uuid = card.get("uuid")
    if not uuid:
        # Driver probes predate UUID discovery only on legacy fixtures/probe misses.
        # A single auto card requires no pin. Multi-device launch must not guess an ordinal.
        if len(nvidia_cards()) == 1 and (selection or _effective_gpu_selection()) == "auto":
            return {}
        raise GpuSelectionError("GPU UUID is unavailable; refresh GPU detection before launching")
    return {"CUDA_VISIBLE_DEVICES": uuid}


_engine_cards_cache: dict[str, tuple[float, list[dict]]] = {}


def accelerator_cards(*, fresh: bool = False) -> list[dict]:
    """Vulkan/HIP inventory in the engine's device namespace, including mixed vendors."""
    from hermes_cli.local_runtime.devices import probe_devices

    engine = _configured_engine()
    if engine is None or engine.backend not in _GPU_ENGINE_BACKENDS:
        return []
    key, now = str(engine.binary), time.monotonic()
    cached = _engine_cards_cache.get(key)
    if not fresh and cached and now - cached[0] < _GPU_QUERY_TTL_S:
        return cached[1]
    devices = probe_devices(engine.binary.parent, engine.backend)
    cards = [{"index": i, "id": d["name"], "name": d["name"], "backend": engine.backend,
              "gpu_name": d["description"], "total_bytes": d["total"],
              "free_bytes": d["free"], "type": d["type"]}
             for i, d in enumerate(devices) if d.get("name")]
    _engine_cards_cache[key] = (now, cards)
    return cards


def gpu_cards() -> list[dict]:
    """Selectable GPUs of the configured/serving backend, not another vendor's ordinal space."""
    engine = _configured_engine()
    if engine is not None:
        if engine.backend in _GPU_ENGINE_BACKENDS:
            return accelerator_cards()
        if engine.backend != "cuda":
            return []
    return nvidia_cards()


def selected_gpu_card(selection: str | None = None) -> dict | None:
    from hermes_cli.local_runtime.gpu_selection import choose

    engine = _configured_engine()
    if engine is not None and engine.backend in _GPU_ENGINE_BACKENDS:
        return choose(accelerator_cards(), _effective_gpu_selection() if selection is None else selection)
    return selected_nvidia_card(selection)


def gpu_device_args(backend: str, selection: str) -> list[str]:
    if backend not in _GPU_ENGINE_BACKENDS:
        return []
    from hermes_cli.local_runtime.gpu_selection import choose

    card = choose(accelerator_cards(fresh=True), selection)
    return ["--device", card["name"], "--split-mode", "none"] if card else []


_gpu_query_cache: tuple[float, dict | None] | None = None
# The statusbar polls /api/local-models/hardware every 5s and the endpoint needs
# name/util/vram; the budget probe needs total/free. One shared query (with a TTL
# shorter than the poll) serves both, so one poll = one nvidia-smi spawn (#120262)
# instead of two, and the spawn carries CREATE_NO_WINDOW so a console-less backend
# never flashes a window (#101895).
_GPU_QUERY_TTL_S = 4.0


def _cached_nvidia_gpu_query(ttl_s: float = _GPU_QUERY_TTL_S) -> dict | None:
    """One nvidia-smi read shared by the budget probe and the hardware endpoint.

    Returns ``dict(gpu_name=, total_bytes=, free_bytes=, used_bytes=, gpu_util_percent=,
    gpu_pci_id=)`` or None when nvidia-smi is absent, fails, or is not an NVIDIA card.
    Cached for ``ttl_s`` (failures too — a missing smi must not spawn per poll).
    """
    global _gpu_query_cache
    now = time.monotonic()
    if _gpu_query_cache is not None and now - _gpu_query_cache[0] < ttl_s:
        return _gpu_query_cache[1]

    exe = _nvidia_smi_path()
    if exe is None:
        _gpu_query_cache = (now, None)
        return None

    from hermes_cli._subprocess_compat import windows_hide_flags
    with suppress(OSError, ValueError, subprocess.TimeoutExpired):
        out = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu,index,uuid,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            creationflags=windows_hide_flags(), check=False)
        if out.returncode != 0 or not out.stdout.strip():
            _gpu_query_cache = (now, None)
            return None

        def _card(row: str, index: int) -> dict | None:
            try:
                fields = next(csv.reader([row], skipinitialspace=True))
                total_mib, free_mib, name, raw_id, used_mib, util = fields[:6]
                pci_id = None
                with suppress(ValueError):
                    pci_id = int(raw_id, 16)
                total, free = int(total_mib) << 20, int(free_mib) << 20
                if total <= 0 or not 0 <= free <= total:
                    return None
                data = {
                    "index": int(fields[6]) if len(fields) > 6 else index,
                    "gpu_name": name.strip(), "total_bytes": total, "free_bytes": free,
                    "used_bytes": int(used_mib) << 20,
                    "gpu_util_percent": int(util) if util.strip().isdigit() else None,
                    "gpu_pci_id": pci_id,
                }
                if len(fields) >= 9:
                    data.update(uuid=fields[7].strip(), id=fields[7].strip(),
                                pci_bus_id=fields[8].strip())
                return data
            except (StopIteration, ValueError):
                return None

        # One row per card (#134227): a display card plus a dedicated compute card is a common
        # topology, and selection/placement need the whole list. The top-level keys stay the
        # FIRST row so single-card callers (statusbar facts) keep their historical answer.
        rows = [card for card in
                (_card(line, i) for i, line in enumerate(out.stdout.strip().splitlines()))
                if card is not None]
        if not rows:
            _gpu_query_cache = (now, None)
            return None
        data = dict(rows[0])
        data["cards"] = rows
        _gpu_query_cache = (now, data)
        return data
    _gpu_query_cache = (now, None)
    return None


def _cuda_driver_pool() -> tuple[int, bool | None] | None:
    """(allocator_total_bytes, integrated_or_None) from the CUDA driver API via ctypes against the
    driver's own DLL/SO — no toolkit, no subprocess, ~ms. INTEGRATED is the vendor's own
    unified-memory declaration; total is the pool the allocator will actually hand out (on
    carve-out devices, several times what nvidia-smi reports)."""
    import ctypes

    for name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return None
    with suppress(OSError, AttributeError):
        if cuda.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        selected = selected_nvidia_card()
        target = (selected or {}).get("pci_bus_id")
        if target:
            # PCI identity selects the same physical card despite CUDA visibility/ordinal remapping.
            get_by_bus = cuda.cuDeviceGetByPCIBusId
            get_by_bus.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
            if get_by_bus(ctypes.byref(dev), target.encode("ascii")) != 0:
                return None
        elif cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
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
    return None


def _configured_engine():
    """The engine the probes below run: the managed server's own when one is up or booting, else
    the installed engine for ``local_runtime.backend``."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.local_runtime.binaries import installed_engine
    from hermes_cli.local_runtime.bootstrap import serving_engine

    engine = serving_engine()
    if engine is not None:
        return engine
    section = load_config_readonly().get("local_runtime") or {}
    return installed_engine(section.get("backend") or "auto")


def _engine_device_pool() -> tuple[int, bool | None] | None:
    """(engine_total_bytes, None) from the installed runtime's own --list-devices, or None. The
    fallback when the driver API is unreachable: asks the exact binary that will do the
    allocating. Carries no integrated verdict — callers must gate it."""
    with suppress(Exception):  # a probe miss must never block budgeting
        engine = _configured_engine()
        if engine is None:
            return None
        exe = engine.binary
        from hermes_cli._subprocess_compat import windows_hide_flags

        child_env = dict(os.environ)
        child_env.update(gpu_pin_env(engine.backend))
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=30,
                             cwd=str(exe.parent), check=False, env=child_env,
                             creationflags=windows_hide_flags())
        if out.returncode != 0:
            return None
        for line in (out.stdout + out.stderr).splitlines():
            m = _DEVICE_LINE_RE.search(line)
            if m:
                return int(m.group(1)) << 20, None
    return None


def _device_pool_view() -> tuple[int, bool | None] | None:
    """Best available allocator-side view, cached: a hit is permanent for the process, a miss
    retries after a short TTL (the engine binary can appear mid-session via a pane install)."""
    global _pool_probe_cache, _pool_probe_key
    selected = selected_nvidia_card()
    key = ((selected or {}).get("uuid"), (selected or {}).get("index"),
           os.environ.get("CUDA_VISIBLE_DEVICES"))
    if key != _pool_probe_key:
        _pool_probe_cache = None
        _pool_probe_key = key
    now = time.monotonic()
    if _pool_probe_cache is not None:
        stamp, view = _pool_probe_cache
        if view is not None or now - stamp < _POOL_NEGATIVE_TTL_S:
            return view
    view = _cuda_driver_pool() or _engine_device_pool()
    _pool_probe_cache = (now, view)
    return view


_accelerator_cache: dict[str, tuple[float, dict | None]] = {}


def _accelerator_device(*, fresh: bool = False) -> dict | None:
    """Selected Vulkan/HIP device, with legacy auto-only probe fallback."""
    from hermes_cli.local_runtime.devices import probe_devices
    from hermes_cli.local_runtime.gpu_selection import choose

    engine = _configured_engine()
    if engine is None or engine.backend not in _GPU_ENGINE_BACKENDS:
        return None
    cards = accelerator_cards(fresh=fresh)
    if cards:
        card = choose(cards, _effective_gpu_selection())
        if card is not None:
            return {"name": card["name"], "description": card["gpu_name"],
                    "type": card["type"], "total": card["total_bytes"], "free": card["free_bytes"]}
    # Older probe results may not carry device names; only Automatic can use those safely.
    if _effective_gpu_selection() != "auto":
        choose([], _effective_gpu_selection())
    key, now = str(engine.binary), time.monotonic()
    cached = _accelerator_cache.get(key)
    if not fresh and cached and (cached[1] is not None or now - cached[0] < _POOL_NEGATIVE_TTL_S):
        return cached[1]
    devices = probe_devices(engine.binary.parent, engine.backend)
    device = max(devices, key=lambda d: (d["type"] == GGML_DEVICE_GPU, d["total"]), default=None)
    _accelerator_cache[key] = (now, device)
    return device


def _unified_pool_bytes(smi_total: int, ram_total: int) -> int | None:
    """The real pool size when this NVIDIA device is unified memory behind a carve-out, else None.

    The driver's INTEGRATED attribute decides in BOTH directions when readable; only the
    attribute-less engine fallback needs the two numeric gates, both of which must hold.
    """
    view = _device_pool_view()
    if view is None:
        return None
    pool, integrated = view
    if integrated is not None:
        return pool if integrated else None
    if (smi_total > 0 and pool >= int(smi_total * _POOL_DISAGREEMENT_FACTOR)
            and ram_total > 0 and pool >= int(ram_total * _POOL_RAM_FRACTION)):
        return pool
    return None


def _uma_budget(base: int, total: int, *, gpu_name: str = "",
                gpu_pci_id: int | None = None, lazy_reads: bool = True) -> HardwareBudget:
    usable = max(0, int(base * (1 - _UMA_HEADROOM_FRACTION)))
    return HardwareBudget(usable_vram_bytes=usable, total_device_bytes=total,
                          ram_available_bytes=0, uma=True, gpu_name=gpu_name, platform=sys.platform,
                          gpu_pci_id=gpu_pci_id, lazy_reads=lazy_reads)


def _windows_commit_room(own_bytes: int = 0) -> int | None:
    """Commit the managed server may take on Windows right now, or None off Windows.

    Windows charges every GPU allocation to commit: the RAM Windows can see plus the page file.
    Memory carved out for the GPU is invisible to Windows, so a bigger carve-out leaves less
    commit for the model it was meant to hold. ``own_bytes`` is what the managed server holds now
    and frees before the new instance loads. A desktop margin stays free.
    """
    stat = _windows_memory_status()
    if stat is None or not stat.ullTotalPageFile:
        return None
    margin = max(_MARGIN_FLOOR, int(stat.ullTotalPhys * _MARGIN_FRACTION))
    return max(0, int(stat.ullAvailPageFile) + max(0, own_bytes) - margin)


def model_commit_bytes(model_bytes: int) -> int:
    """What Windows charges to commit for a model estimated at ``model_bytes`` of device memory:
    the estimate runs ~20% under the charge, which is the UMA headroom."""
    return int(model_bytes / (1 - _UMA_HEADROOM_FRACTION))


def prompt_cache_bytes(live: HardwareBudget | None, model_bytes: int, residents: int = 1) -> int | None:
    """Size for llama-server's prompt cache beside a model of ``model_bytes`` (the estimate the
    window was planned with), or None when the engine default applies.

    Windows unified memory only: each loaded model's cache comes out of the same commit as the
    models, so each of the ``residents`` models the router may hold at once (its ``--models-max``)
    gets an even share of what they leave.
    """
    room = live.commit_room_bytes if live is not None else None
    if room is None or model_bytes <= 0:
        return None
    residents = max(1, residents)
    model_commit = model_commit_bytes(model_bytes)
    cache = min(_PROMPT_CACHE_MAX, max(0, room - residents * model_commit) // residents)
    return cache if cache >= _PROMPT_CACHE_MIN else 0


def usable_beside_prompt_cache(live: HardwareBudget, residents: int = 1) -> int:
    """Device memory a model may plan with on Windows unified memory and still leave each of the
    ``residents`` models the smallest useful prompt cache: the inverse of ``prompt_cache_bytes``."""
    share = live.commit_room_bytes // max(1, residents) - _PROMPT_CACHE_MIN
    return min(live.usable_vram_bytes, max(0, int(share * (1 - _UMA_HEADROOM_FRACTION))))


def _gpu_engine_runs_the_model() -> bool:
    """Whether a Vulkan/HIP engine runs the model: the serving or installed engine, else the one
    setup would install for ``local_runtime.backend``. False when the engine can't be resolved."""
    with suppress(Exception):
        engine = _configured_engine()
        if engine is not None:
            return engine.backend in _GPU_ENGINE_BACKENDS
        from hermes_cli.config import load_config_readonly
        from hermes_cli.local_runtime.binaries import resolve_backend

        section = load_config_readonly().get("local_runtime") or {}
        return resolve_backend(section.get("backend") or "auto") in _GPU_ENGINE_BACKENDS
    return False


def _windows_igpu_bytes(device: dict | None) -> int | None:
    """What Windows lets the integrated GPU allocate (Task Manager's GPU memory), or None.

    The adapter is the one the engine named, else the only non-NVIDIA hardware adapter when a
    Vulkan/HIP engine will run the model; the CPU engine (chosen, or the only build for a Snapdragon)
    keeps the model in system RAM. Not the engine's own figure: ggml-vulkan sums every memory heap
    of an integrated GPU, host-visible ones included, so it can exceed what Windows grants.
    """
    from hermes_platform.host.gpu_adapters import windows_gpu_adapters

    hardware = [a for a in windows_gpu_adapters() if not a.software]
    named = [a for a in hardware if device is not None and a.description == device["description"]]
    candidates = named or [a for a in hardware if a.vendor_id != _NVIDIA_VENDOR_ID]
    if len(candidates) != 1 or (device is None and not _gpu_engine_runs_the_model()):
        return None
    return candidates[0].memory_bytes


def probe_budget(*, planning: bool = False) -> HardwareBudget:
    from hermes_cli.local_runtime.gpu_selection import GpuSelectionError

    try:
        return _probe_budget(planning=planning)
    except GpuSelectionError as exc:
        logger.warning("GPU budget unavailable: %s", exc)
        return HardwareBudget(0, 0, 0)


def _probe_budget(*, planning: bool = False) -> HardwareBudget:
    """Construct the budget per the source rules above.

    ``planning=False``: LIVE budget (free VRAM now) for launch-time fit and growth re-grants.
    ``planning=True``: CAPACITY budget (total minus margin) for catalog pricing and quant
    selection — pricing against live-free while a model was loaded made every row read as too
    large. The managed server unloads/relaunches itself, so capacity is real.
    """
    ram_total, ram_avail = _ram_bytes()
    vram = _nvidia_vram()

    # Unified-memory NVIDIA: the CUDA allocator pool is the real capacity. Classification comes
    # from the driver API/engine and must not require nvidia-smi (stripped-PATH sessions lose smi
    # but nvcuda loads via the system loader). Deliberately NOT clamped to OS RAM: carved-out
    # memory is invisible to GlobalMemoryStatusEx, so a RAM clamp would throw away exactly that
    # capacity. On Windows, commit narrows the launch window instead (``launch_budget``).
    engine = _configured_engine()
    unified = None
    if engine is None or engine.backend == "cuda":
        # Missing smi can still have a usable CUDA allocator (N1X). A visibility mask
        # hiding known cards, in contrast, must not price an allocator the child cannot use.
        if vram is not None or not nvidia_cards():
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
            # Live: dedicated-free plus what the OS can still give. smi's free saturates at the
            # carve-out so this under-counts a bit — the safe direction (the pool edge is a
            # measured soft cliff: decode collapses ~3.5x when concurrent demand hits it).
            live = (vram[1] + ram_avail) if vram else ram_avail
            base = min(unified, live)
        return _uma_budget(base, unified, gpu_name=vram[2] if vram else "",
                           gpu_pci_id=vram[3] if vram else None)

    if vram is None:
        device = _accelerator_device()
        if device is not None and device["type"] == GGML_DEVICE_GPU:
            # Discrete AMD/Intel card behind a Vulkan/HIP engine: its own memory, with system RAM
            # as spill. Capacity serves the polled live view; launch_budget reads free memory.
            total = device["total"]
            margin = max(_MARGIN_FLOOR, int(total * _MARGIN_FRACTION))
            return HardwareBudget(usable_vram_bytes=max(0, total - margin), total_device_bytes=total,
                                  ram_available_bytes=ram_total if planning else ram_avail,
                                  uma=False, gpu_name=device["description"], platform=sys.platform)
        # Integrated GPU, Metal, CPU, or no answer: budget from RAM as unified memory, capped at what
        # Windows lets an integrated GPU allocate (18 GB of 31.5 on an Arc B390). llama.cpp loads
        # lazy tensors up front on an AMD/Intel integrated GPU, where reading them on demand halved
        # prefill (#28160).
        pool = min(ram_total, _windows_igpu_bytes(device) or ram_total)
        return _uma_budget(pool if planning else min(pool, ram_avail), pool,
                           lazy_reads=gpu_class() not in ("amd", "intel"))

    total, free, gpu_name, gpu_pci_id = vram
    margin = max(_MARGIN_FLOOR, int(total * _MARGIN_FRACTION))
    return HardwareBudget(usable_vram_bytes=max(0, (total if planning else free) - margin),
                          total_device_bytes=total,
                          ram_available_bytes=ram_total if planning else ram_avail,
                          uma=False, gpu_name=gpu_name, platform=sys.platform, gpu_pci_id=gpu_pci_id)


def launch_budget(capacity: HardwareBudget, *, own_bytes: int = 0) -> HardwareBudget | None:
    """Capacity less what other programs hold on the card right now, or None when that can't apply.

    Capacity assumes other programs hold no more than the fixed margin. Beside a heavier desktop the
    capacity-sized window doesn't fit, and Windows pages part of the model to host memory without
    an error: a 32 GiB card with 4.5-6.3 GiB held by other apps decoded at ~24 tok/s at the
    capacity window, and at ~90 tok/s at a window sized from free memory.

    ``own_bytes`` is the device-memory estimate of what the managed server holds now and frees
    before the new instance loads. A stopped server's memory reads as free by the time its process
    has exited (measured on Windows: free memory was fully back at the first reading after exit).
    Unified memory: free commit on Windows (the model and its prompt cache must fit there), with
    ``own_bytes`` converted to the commit Windows charged for it, else None (its live budget is
    already free memory). None too when the device query fails, so callers keep the capacity plan.
    """
    if capacity.uma:
        room = _windows_commit_room(model_commit_bytes(own_bytes) if own_bytes > 0 else 0)
        if room is None:
            return None
        usable = min(capacity.usable_vram_bytes, int(room * (1 - _UMA_HEADROOM_FRACTION)))
        return replace(capacity, usable_vram_bytes=usable, commit_room_bytes=room)
    vram = _nvidia_vram()
    if vram is not None:
        total, free = vram[0], vram[1]
    else:
        # A discrete card behind Vulkan/HIP: the driver's free count includes other programs.
        device = _accelerator_device(fresh=True)
        if device is None or device["type"] != GGML_DEVICE_GPU:
            return None
        total, free = device["total"], device["free"]
    others = max(0, total - free - max(0, own_bytes))
    usable = min(capacity.usable_vram_bytes, max(0, total - others - _LAUNCH_HEADROOM))
    return replace(capacity, usable_vram_bytes=usable)
