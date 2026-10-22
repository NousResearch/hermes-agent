"""Ask the installed llama.cpp build which accelerators it sees, in a disposable process.

ggml's device type is the engine's own integrated-vs-discrete verdict (Vulkan reads it from
VkPhysicalDeviceType), and llama.cpp places layers on discrete devices whenever one exists, so the
budget can follow the same rule. Loading a GPU driver can crash or hang the caller, hence a child
with a deadline that imports only the standard library (``-I``). Run directly:
``python -I devices.py <engine dir> <backend>`` prints the device list as JSON.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

# enum ggml_backend_dev_type (ggml-backend.h): CPU, GPU, IGPU, ACCEL.
GGML_DEVICE_GPU = 1
GGML_DEVICE_IGPU = 2


def probe_devices(engine_dir: Path, backend: str) -> list[dict]:
    """GPU/iGPU devices the ``backend`` library in ``engine_dir`` registers; [] on any failure."""
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        out = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), str(engine_dir), backend],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20, creationflags=windows_hide_flags(), check=False)
        devices = json.loads(out.stdout) if out.returncode == 0 else []
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    if not isinstance(devices, list):
        return []
    return [d for d in devices if isinstance(d, dict)
            and d.get("type") in (GGML_DEVICE_GPU, GGML_DEVICE_IGPU)
            and isinstance(d.get("total"), int) and d["total"] > 0
            and isinstance(d.get("description"), str)]


def _library(directory: Path, name: str) -> Path:
    return directory / (f"{name}.dll" if os.name == "nt" else f"lib{name}.so")


def _read_devices(directory: Path, backend: str) -> list[dict]:
    dll_directory = os.add_dll_directory(str(directory)) if os.name == "nt" else None
    try:
        base = ctypes.CDLL(str(_library(directory, "ggml-base")), mode=ctypes.RTLD_GLOBAL)
        core = ctypes.CDLL(str(_library(directory, "ggml")), mode=ctypes.RTLD_GLOBAL)
        core.ggml_backend_load.argtypes = [ctypes.c_char_p]
        core.ggml_backend_load.restype = ctypes.c_void_p
        registry = core.ggml_backend_load(os.fsencode(_library(directory, f"ggml-{backend}")))
        if not registry:
            return []
        signatures = {
            "ggml_backend_reg_dev_count": ([ctypes.c_void_p], ctypes.c_size_t),
            "ggml_backend_reg_dev_get": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p),
            "ggml_backend_dev_name": ([ctypes.c_void_p], ctypes.c_char_p),
            "ggml_backend_dev_description": ([ctypes.c_void_p], ctypes.c_char_p),
            "ggml_backend_dev_type": ([ctypes.c_void_p], ctypes.c_int),
            "ggml_backend_dev_memory": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_size_t)], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(base, name)
            function.argtypes, function.restype = arguments, result
        devices = []
        for index in range(min(base.ggml_backend_reg_dev_count(registry), 64)):
            device = base.ggml_backend_reg_dev_get(registry, index)
            if not device:
                continue
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            base.ggml_backend_dev_memory(device, ctypes.byref(free), ctypes.byref(total))
            devices.append({
                "name": base.ggml_backend_dev_name(device).decode("utf-8", errors="replace"),
                "description": base.ggml_backend_dev_description(device).decode("utf-8", errors="replace"),
                "type": base.ggml_backend_dev_type(device),
                "free": min(free.value, total.value), "total": total.value,
            })
        return devices
    finally:
        if dll_directory is not None:
            dll_directory.close()


def cuda_visible_uuids() -> list[str] | None:
    """Ask a fresh CUDA process to interpret inherited ordinals; parent CUDA may be initialised."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    from hermes_cli.local_runtime.processes import server_child_env

    try:
        out = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--cuda-uuids"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20, creationflags=windows_hide_flags(),
            env=server_child_env(os.environ), check=False)
        result = json.loads(out.stdout) if out.returncode == 0 else None
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def _cuda_uuids() -> list[str]:
    import uuid

    cuda = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    if cuda.cuInit(0):
        return []
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)):
        return []
    result = []
    for index in range(min(count.value, 64)):
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), index):
            continue
        raw = (ctypes.c_ubyte * 16)()
        # The original UUID API identifies the physical GPU, not a MIG instance.
        if cuda.cuDeviceGetUuid(ctypes.byref(raw), device) == 0:
            result.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return result


if __name__ == "__main__":
    print(json.dumps(_cuda_uuids() if sys.argv[1] == "--cuda-uuids"
                     else _read_devices(Path(sys.argv[1]), sys.argv[2])))
