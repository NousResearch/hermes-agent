"""Read device topology from the selected llama.cpp build in a disposable process."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def probe_devices(install_dir: Path) -> list[dict]:
    from hermes_cli.local_runtime.binaries import BinaryResolutionError, server_binary

    try:
        directory = server_binary(install_dir).parent
        result = subprocess.run(
            [getattr(sys, "_base_executable", sys.executable), "-I",
             str(Path(__file__).resolve()), str(directory), install_dir.name],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            return []
        devices = json.loads(result.stdout)
        if not isinstance(devices, list):
            return []
        return [d for d in devices if isinstance(d, dict)
                and isinstance(d.get("name"), str) and d["name"]
                and d.get("type") in (1, 2)
                and isinstance(d.get("total"), int) and d["total"] > 0
                and isinstance(d.get("free"), int) and 0 <= d["free"] <= d["total"]]
    except (BinaryResolutionError, OSError, ValueError, subprocess.TimeoutExpired):
        return []


def _read_devices(directory: Path, backend: str) -> list[dict]:
    def library(name):
        if os.name == "nt":
            return directory / f"{name}.dll"
        return directory / f"lib{name}.so"

    dll_directory = os.add_dll_directory(str(directory)) if os.name == "nt" else None
    try:
        base = ctypes.CDLL(str(library("ggml-base")), mode=ctypes.RTLD_GLOBAL)
        core = ctypes.CDLL(str(library("ggml")), mode=ctypes.RTLD_GLOBAL)
        core.ggml_backend_load.argtypes = [ctypes.c_char_p]
        core.ggml_backend_load.restype = ctypes.c_void_p
        registry = core.ggml_backend_load(os.fsencode(library(f"ggml-{backend}")))
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
            kind = base.ggml_backend_dev_type(device)
            if kind not in (1, 2):
                continue
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            base.ggml_backend_dev_memory(device, ctypes.byref(free), ctypes.byref(total))
            devices.append({
                "name": base.ggml_backend_dev_name(device).decode("utf-8", errors="replace"),
                "description": base.ggml_backend_dev_description(device).decode("utf-8", errors="replace"),
                "type": kind, "free": min(free.value, total.value), "total": total.value,
            })
        return devices
    finally:
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    print(json.dumps(_read_devices(Path(sys.argv[1]), sys.argv[2])))
