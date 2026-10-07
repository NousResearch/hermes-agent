"""Read the CUDA driver's allocator pool in a disposable process.

An in-process ``cuInit`` turns the caller into a CUDA client for the rest of its life: the
driver keeps device handles open until that process exits, and a client that never allocates
still keeps the GPU from reaching runtime suspend. ``hardware._cuda_driver_pool`` is reached
from the hardware endpoint the Desktop backend polls every few seconds, so the probe runs as
a short-lived child — the same boundary ``devices.py`` uses for the engine's device list —
and only its plain-data result crosses back.

Run directly: ``python -I cuda_probe.py`` prints one JSON line,
``{"pool": <bytes>|null, "integrated": <bool>|null}``.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import sys

# cuDeviceGetAttribute enum: device is integrated with host memory.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18

# A wedged driver must not stall the caller: cold dlopen measures in milliseconds, and the
# polled endpoint this feeds degrades to the engine device pool when the child gives up.
_PROBE_TIMEOUT_S = 10.0


def probe_pool() -> "tuple[int, bool | None] | None":
    """(allocator_total_bytes, integrated_or_None) from a bounded child, or None.

    None when the driver is absent, refuses, or the child fails/times out; callers fall back
    to the engine device pool.
    """
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        out = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve())],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=_PROBE_TIMEOUT_S, creationflags=windows_hide_flags())
        # The child's single JSON line is the contract; a stray driver print before it must
        # not corrupt the read.
        line = next((ln for ln in reversed(out.stdout.splitlines()) if ln.strip()), "")
        payload = json.loads(line) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if not isinstance(payload, dict):
        return None
    pool = payload.get("pool")
    if not isinstance(pool, int) or pool <= 0:
        return None
    integrated = payload.get("integrated")
    return pool, (integrated if isinstance(integrated, bool) else None)


def _read_pool() -> "tuple[int, bool | None] | None":
    """Child half: the driver API via ctypes against the driver's own DLL/SO — no toolkit.
    INTEGRATED is the vendor's own unified-memory declaration; total is the pool the allocator
    will actually hand out (on carve-out devices, several times what nvidia-smi reports)."""
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


if __name__ == "__main__":
    result = _read_pool()
    print(json.dumps({"pool": None if result is None else result[0],
                      "integrated": None if result is None else result[1]}))
