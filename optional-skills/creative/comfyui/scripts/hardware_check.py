#!/usr/bin/env python3
"""Detect whether this machine can realistically run ComfyUI locally.

Emits a structured JSON report the agent can read to decide whether to:
  - help the user install ComfyUI locally, or
  - steer them to Comfy Cloud instead.

Usage:
    python3 hardware_check.py [--json]

Exit code:
    0  → "ok"       — can run local ComfyUI at reasonable speed
    1  → "marginal" — technically works but slow / memory-tight
    2  → "cloud"    — local is not viable, recommend Comfy Cloud

The JSON report always prints to stdout regardless of exit code.

Output fields the agent should read:
    verdict:                    "ok" | "marginal" | "cloud"
    recommended_install_path:   "nvidia" | "amd" | "apple-silicon" | "intel" | "comfy-cloud"
    comfy_cli_flag:             "--nvidia" | "--amd" | "--m-series" | None
                                (pass directly to `comfy install` when verdict != cloud)
    gpu:                        detected GPU info or null
    notes:                      list of human-readable strings to surface to the user
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys


# Rough thresholds. SDXL/Flux need real VRAM; SD1.5 will scrape by on 6GB.
# Apple Silicon shares RAM with GPU — unified memory budget is total RAM.
MIN_VRAM_GB_USABLE = 6     # below this, most modern models won't load
OK_VRAM_GB = 8             # SDXL fits comfortably here
GREAT_VRAM_GB = 12         # Flux / video models start being realistic
MIN_MAC_RAM_GB = 16        # Apple Silicon unified memory; below = pain
OK_MAC_RAM_GB = 32         # smooth for SDXL / most workflows


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def detect_nvidia() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out.strip():
        return None
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    name = parts[0]
    try:
        vram_mb = int(parts[1])
    except ValueError:
        vram_mb = 0
    driver = parts[2] if len(parts) > 2 else ""
    return {
        "vendor": "nvidia",
        "name": name,
        "vram_gb": round(vram_mb / 1024, 1),
        "driver": driver,
    }


def detect_rocm() -> dict | None:
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if not out.strip():
        return None
    name_m = re.search(r"Card series:\s*(.+)", out)
    vram_m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", out)
    vram_gb = 0.0
    if vram_m:
        vram_gb = round(int(vram_m.group(1)) / (1024**3), 1)
    return {
        "vendor": "amd",
        "name": name_m.group(1).strip() if name_m else "AMD GPU",
        "vram_gb": vram_gb,
        "driver": "rocm",
    }


def detect_apple_silicon() -> dict | None:
    if platform.system() != "Darwin":
        return None
    if platform.machine() != "arm64":
        return None  # Intel Mac — no usable MPS
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    # Examples: "Apple M1", "Apple M1 Pro", "Apple M2 Max", "Apple M3 Ultra"
    m = re.search(r"Apple M(\d+)", chip)
    generation = int(m.group(1)) if m else 1
    mem_bytes = 0
    try:
        mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    except ValueError:
        pass
    ram_gb = round(mem_bytes / (1024**3), 1) if mem_bytes else 0.0
    return {
        "vendor": "apple",
        "name": chip or "Apple Silicon",
        "generation": generation,
        "unified_memory_gb": ram_gb,
    }


def detect_intel_arc() -> dict | None:
    if platform.system() != "Linux":
        return None
    if not shutil.which("clinfo"):
        return None
    out = _run(["clinfo", "--list"])
    if "Intel" in out and ("Arc" in out or "Xe" in out):
        return {"vendor": "intel", "name": "Intel Arc/Xe", "vram_gb": 0.0}
    return None


def total_system_ram_gb() -> float:
    sysname = platform.system()
    if sysname == "Darwin":
        try:
            return round(int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0) / (1024**3), 1)
        except ValueError:
            return 0.0
    if sysname == "Linux":
        try:
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        except OSError:
            return 0.0
    if sysname == "Windows":
        out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
        m = re.search(r"(\d{6,})", out)
        if m:
            return round(int(m.group(1)) / (1024**3), 1)
    return 0.0


# Map recommended_install_path → flag the agent can pass to `comfy install`
# Set to None when no local install is advised (verdict=cloud).
_COMFY_CLI_FLAG = {
    "nvidia": "--nvidia",
    "amd": "--amd",
    "apple-silicon": "--m-series",
    "intel": None,          # comfy-cli has no Intel Arc flag — manual install
    "comfy-cloud": None,
}


def classify(gpu: dict | None, ram_gb: float) -> tuple[str, str, list[str]]:
    """Return (verdict, recommended_install_path, notes)."""
    notes: list[str] = []

    if gpu is None:
        notes.append(
            "No supported accelerator found (NVIDIA CUDA / AMD ROCm / Apple Silicon / Intel Arc)."
        )
        notes.append(
            "CPU-only ComfyUI works but is unusably slow for modern models — use Comfy Cloud."
        )
        return "cloud", "comfy-cloud", notes

    if gpu["vendor"] == "apple":
        gen = gpu.get("generation", 1)
        mem = gpu.get("unified_memory_gb", 0.0)
        if mem < MIN_MAC_RAM_GB:
            notes.append(
                f"Apple Silicon with {mem} GB unified memory — below the {MIN_MAC_RAM_GB} GB practical minimum."
            )
            notes.append("SD1.5 may work; SDXL/Flux will swap or OOM. Recommend Comfy Cloud.")
            return "cloud", "comfy-cloud", notes
        if mem < OK_MAC_RAM_GB:
            notes.append(
                f"Apple Silicon M{gen} with {mem} GB — SDXL works but slow. Flux/video likely too tight."
            )
            return "marginal", "apple-silicon", notes
        notes.append(f"Apple Silicon M{gen} with {mem} GB unified memory — good for SDXL/Flux.")
        return "ok", "apple-silicon", notes

    # Discrete GPU path (nvidia/amd/intel)
    vram = gpu.get("vram_gb", 0.0)
    if gpu["vendor"] == "intel":
        notes.append("Intel Arc detected — ComfyUI IPEX support is experimental; Comfy Cloud is more reliable.")
        return "marginal", "intel", notes
    if vram < MIN_VRAM_GB_USABLE:
        notes.append(
            f"{gpu['name']} has only {vram} GB VRAM — below the {MIN_VRAM_GB_USABLE} GB practical minimum."
        )
        notes.append("Most modern models won't load. Recommend Comfy Cloud.")
        return "cloud", "comfy-cloud", notes
    if vram < OK_VRAM_GB:
        notes.append(
            f"{gpu['name']} ({vram} GB VRAM) — SD1.5 works, SDXL tight, Flux/video unlikely."
        )
        return "marginal", gpu["vendor"], notes
    if vram < GREAT_VRAM_GB:
        notes.append(f"{gpu['name']} ({vram} GB VRAM) — SDXL comfortable, Flux possible with optimizations.")
        return "ok", gpu["vendor"], notes
    notes.append(f"{gpu['name']} ({vram} GB VRAM) — can run everything including Flux/video.")
    return "ok", gpu["vendor"], notes


def build_report() -> dict:
    sysname = platform.system()
    arch = platform.machine()
    ram_gb = total_system_ram_gb()

    gpu = (
        detect_nvidia()
        or detect_rocm()
        or detect_apple_silicon()
        or detect_intel_arc()
    )

    # Intel Mac special case — fall out of apple-silicon detection with no GPU
    if gpu is None and sysname == "Darwin" and platform.machine() != "arm64":
        notes = [
            "Intel Mac detected — no MPS backend available.",
            "ComfyUI will fall back to CPU which is unusably slow. Use Comfy Cloud.",
        ]
        return {
            "os": sysname,
            "arch": arch,
            "system_ram_gb": ram_gb,
            "gpu": None,
            "verdict": "cloud",
            "recommended_install_path": "comfy-cloud",
            "comfy_cli_flag": None,
            "notes": notes,
            "install_urls": _install_urls(),
        }

    verdict, install_path, notes = classify(gpu, ram_gb)

    return {
        "os": sysname,
        "arch": arch,
        "system_ram_gb": ram_gb,
        "gpu": gpu,
        "verdict": verdict,
        "recommended_install_path": install_path,
        "comfy_cli_flag": _COMFY_CLI_FLAG.get(install_path),
        "notes": notes,
        "install_urls": _install_urls(),
    }


def _install_urls() -> dict:
    return {
        "desktop": "https://docs.comfy.org/installation/desktop",
        "manual": "https://docs.comfy.org/installation/manual_install",
        "comfy_cli": "https://docs.comfy.org/comfy-cli/getting-started",
        "cloud": "https://platform.comfy.org",
    }


def main() -> int:
    report = build_report()
    json_mode = "--json" in sys.argv

    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        print(f"OS:      {report['os']} ({report['arch']})")
        print(f"RAM:     {report['system_ram_gb']} GB")
        if report["gpu"]:
            g = report["gpu"]
            if g["vendor"] == "apple":
                print(f"GPU:     {g['name']} — {g.get('unified_memory_gb', 0)} GB unified memory")
            else:
                print(f"GPU:     {g['name']} — {g.get('vram_gb', 0)} GB VRAM")
        else:
            print("GPU:     (none detected)")
        print(f"Verdict: {report['verdict']}  → {report['recommended_install_path']}")
        if report["comfy_cli_flag"]:
            print(f"         → run: comfy --skip-prompt install {report['comfy_cli_flag']}")
        for n in report["notes"]:
            print(f"  • {n}")

    if report["verdict"] == "ok":
        return 0
    if report["verdict"] == "marginal":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
