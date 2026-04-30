#!/usr/bin/env python3
"""hardware_check.py — Detect whether this machine can realistically run ComfyUI locally.

Improvements over v1:
  - Multi-GPU detection: scans all NVIDIA / AMD GPUs, picks the best one (most VRAM)
  - Apple Silicon: detects Rosetta-via-x86_64 false negative; warns instead of misclassifying
  - Apple generation: defaults to None (unknown) instead of mis-tagging as M1
  - WSL2 detection: identifies WSL2 + nvidia-smi situation explicitly
  - ROCm: prefers `rocm-smi --json` for new ROCm 6.x output
  - Disk space check: warns if /home or workspace volume has < 25 GB free
  - PyTorch verification (optional): tries to import torch and check device availability
  - Windows: prefers PowerShell `Get-CimInstance` over deprecated `wmic`
  - More accurate VRAM thresholds and verdict reasons

Emits a structured JSON report. Exit codes match `verdict`:
    0 → ok
    1 → marginal
    2 → cloud

Usage:
    python3 hardware_check.py [--json] [--check-pytorch]
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


# Thresholds (GiB).
MIN_VRAM_GB_USABLE = 6
OK_VRAM_GB = 8
GREAT_VRAM_GB = 12
MIN_MAC_RAM_GB = 16
OK_MAC_RAM_GB = 32
MIN_FREE_DISK_GB = 25  # ComfyUI core ~5 GB + one model ~5–24 GB

_COMFY_CLI_FLAG = {
    "nvidia": "--nvidia",
    "amd": "--amd",
    "apple-silicon": "--m-series",
    "intel": None,
    "comfy-cloud": None,
    "cpu": "--cpu",
}


def _run(cmd: list[str], timeout: int = 8) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def is_wsl() -> bool:
    """Return True when running under Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
        return True
    try:
        with open("/proc/version", "r") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def is_rosetta() -> bool:
    """Return True when Python is running translated under Rosetta on Apple Silicon."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() == "arm64":
        return False
    # x86_64 on Darwin — could be Intel Mac or Rosetta. Probe sysctl.
    out = _run(["sysctl", "-in", "sysctl.proc_translated"]).strip()
    return out == "1"


def detect_nvidia() -> dict | None:
    """Detect NVIDIA GPUs. Returns the GPU with the most VRAM, plus list of all."""
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out.strip():
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram_mb = int(parts[2])
        except ValueError:
            continue
        driver = parts[3] if len(parts) > 3 else ""
        gpus.append({
            "vendor": "nvidia",
            "index": idx,
            "name": name,
            "vram_gb": round(vram_mb / 1024, 1),
            "driver": driver,
        })
    if not gpus:
        return None
    # Pick GPU with most VRAM
    best = max(gpus, key=lambda g: g["vram_gb"])
    if len(gpus) > 1:
        best["all_gpus"] = gpus
    return best


def detect_rocm() -> dict | None:
    if not shutil.which("rocm-smi"):
        return None
    # Prefer JSON output (new ROCm 6.x)
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if out.strip().startswith("{"):
        try:
            data = json.loads(out)
            cards = []
            for card_id, info in data.items():
                if not card_id.startswith("card"):
                    continue
                name = (info.get("Card series") or info.get("Card model")
                        or info.get("Marketing Name") or "AMD GPU")
                vram_b = info.get("VRAM Total Memory (B)") or info.get("vram_total_memory_b") or 0
                try:
                    vram_b = int(vram_b)
                except (ValueError, TypeError):
                    vram_b = 0
                cards.append({
                    "vendor": "amd",
                    "name": str(name).strip(),
                    "vram_gb": round(vram_b / (1024**3), 1),
                    "driver": "rocm",
                })
            if cards:
                best = max(cards, key=lambda c: c["vram_gb"])
                if len(cards) > 1:
                    best["all_gpus"] = cards
                return best
        except json.JSONDecodeError:
            pass
    # Fall back to text parsing
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if not out.strip():
        return None
    name_m = re.search(r"Card (?:series|model|Marketing Name):\s*(.+)", out)
    vram_m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", out)
    vram_gb = round(int(vram_m.group(1)) / (1024**3), 1) if vram_m else 0.0
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
        return None
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    m = re.search(r"Apple M(\d+)", chip)
    generation = int(m.group(1)) if m else None
    mem_bytes = 0
    try:
        mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    except ValueError:
        pass
    ram_gb = round(mem_bytes / (1024**3), 1) if mem_bytes else 0.0

    # Detect chip variant ("Pro", "Max", "Ultra") — affects performance even at same gen
    variant = None
    for v in ("Ultra", "Max", "Pro"):
        if v in chip:
            variant = v
            break

    return {
        "vendor": "apple",
        "name": chip or "Apple Silicon",
        "generation": generation,
        "variant": variant,
        "unified_memory_gb": ram_gb,
    }


def detect_intel_arc() -> dict | None:
    if platform.system() not in ("Linux", "Windows"):
        return None
    if shutil.which("clinfo"):
        out = _run(["clinfo", "--list"])
        if "Intel" in out and ("Arc" in out or "Xe" in out):
            return {"vendor": "intel", "name": "Intel Arc/Xe", "vram_gb": 0.0}
    # Windows: try Get-CimInstance
    if platform.system() == "Windows" and shutil.which("powershell"):
        out = _run(["powershell", "-NoProfile",
                    "Get-CimInstance Win32_VideoController | Select-Object Name | Format-List"])
        if "Intel" in out and ("Arc" in out or "Iris Xe" in out):
            return {"vendor": "intel", "name": "Intel Arc/Iris Xe", "vram_gb": 0.0}
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
        if shutil.which("powershell"):
            out = _run([
                "powershell", "-NoProfile",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            m = re.search(r"(\d{8,})", out)
            if m:
                return round(int(m.group(1)) / (1024**3), 1)
        # Fall back to wmic for older Windows
        out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
        m = re.search(r"(\d{6,})", out)
        if m:
            return round(int(m.group(1)) / (1024**3), 1)
    return 0.0


def total_free_disk_gb(path: str = ".") -> float:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 1)
    except OSError:
        return 0.0


def check_pytorch_cuda() -> dict | None:
    """Optional PyTorch availability check. Only run when --check-pytorch is set."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as e:
        return {"available": False, "reason": f"torch not importable: {e}"}
    info: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
    }
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available"] = False
    try:
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        info["mps_available"] = False
    return info


def classify(gpu: dict | None, ram_gb: float, free_disk_gb: float, *, wsl: bool, rosetta: bool) -> tuple[str, str, list[str]]:
    notes: list[str] = []

    if rosetta:
        notes.append(
            "Detected Python running under Rosetta on Apple Silicon. "
            "ComfyUI MPS support requires native ARM64 Python — install via "
            "`brew install python` or arm64 Miniforge, then re-run."
        )
        return "cloud", "comfy-cloud", notes

    if wsl and gpu and gpu["vendor"] == "nvidia":
        notes.append("Detected WSL2 + NVIDIA — confirm `nvidia-smi` works in your WSL distro before installing.")

    if free_disk_gb and free_disk_gb < MIN_FREE_DISK_GB:
        notes.append(
            f"Free disk space ({free_disk_gb} GB) is below the {MIN_FREE_DISK_GB} GB recommended minimum. "
            "ComfyUI core (~5 GB) plus one SDXL model (~6.5 GB) needs space; Flux Dev needs ~24 GB."
        )

    # Host RAM matters even for discrete-GPU systems: ComfyUI swaps model
    # weights through CPU RAM when shuffling between text encoders / VAE / UNet.
    # Apple's unified-memory check is handled below so don't double-warn.
    if ram_gb and ram_gb < 8 and gpu and gpu.get("vendor") != "apple":
        notes.append(
            f"System RAM ({ram_gb} GB) is low. ComfyUI swaps model weights through "
            "host RAM; <8 GB causes severe slowdowns. 16+ GB recommended."
        )

    if gpu is None:
        notes.append(
            "No supported accelerator found (NVIDIA CUDA / AMD ROCm / Apple Silicon / Intel Arc)."
        )
        notes.append(
            "CPU-only ComfyUI works but is unusably slow for modern models — use Comfy Cloud."
        )
        return "cloud", "comfy-cloud", notes

    if gpu["vendor"] == "apple":
        gen = gpu.get("generation")
        variant = gpu.get("variant")
        mem = gpu.get("unified_memory_gb", 0.0)
        gen_str = f"M{gen}" if gen else "Apple Silicon"
        if variant:
            gen_str += f" {variant}"
        if mem < MIN_MAC_RAM_GB:
            notes.append(
                f"{gen_str} with {mem} GB unified memory — below the {MIN_MAC_RAM_GB} GB practical minimum."
            )
            notes.append("SD1.5 may work; SDXL/Flux will swap or OOM. Recommend Comfy Cloud.")
            return "cloud", "comfy-cloud", notes
        if mem < OK_MAC_RAM_GB:
            notes.append(
                f"{gen_str} with {mem} GB — SDXL works but slow. Flux/video likely too tight."
            )
            return "marginal", "apple-silicon", notes
        notes.append(f"{gen_str} with {mem} GB unified memory — good for SDXL/Flux.")
        return "ok", "apple-silicon", notes

    if gpu["vendor"] == "intel":
        notes.append("Intel Arc detected — ComfyUI IPEX support is experimental; Comfy Cloud is more reliable.")
        return "marginal", "intel", notes

    # Discrete NVIDIA / AMD
    vram = gpu.get("vram_gb", 0.0)
    name = gpu["name"]
    if vram < MIN_VRAM_GB_USABLE:
        notes.append(
            f"{name} has only {vram} GB VRAM — below the {MIN_VRAM_GB_USABLE} GB practical minimum."
        )
        notes.append("Most modern models won't load. Recommend Comfy Cloud.")
        return "cloud", "comfy-cloud", notes
    if vram < OK_VRAM_GB:
        notes.append(
            f"{name} ({vram} GB VRAM) — SD1.5 works, SDXL tight, Flux/video unlikely."
        )
        return "marginal", gpu["vendor"], notes
    if vram < GREAT_VRAM_GB:
        notes.append(f"{name} ({vram} GB VRAM) — SDXL comfortable, Flux possible with optimizations.")
        return "ok", gpu["vendor"], notes
    notes.append(f"{name} ({vram} GB VRAM) — can run everything including Flux/video.")
    return "ok", gpu["vendor"], notes


def build_report(*, check_pytorch: bool = False) -> dict:
    sysname = platform.system()
    arch = platform.machine()
    ram_gb = total_system_ram_gb()
    free_disk_gb = total_free_disk_gb(os.path.expanduser("~"))

    rosetta = is_rosetta()
    wsl = is_wsl()

    gpu = (
        detect_nvidia()
        or detect_rocm()
        or detect_apple_silicon()
        or detect_intel_arc()
    )

    # Intel Mac: arm64 detect failed AND no other GPU paths
    if gpu is None and sysname == "Darwin" and arch != "arm64" and not rosetta:
        notes = [
            "Intel Mac detected — no MPS backend available.",
            "ComfyUI will fall back to CPU which is unusably slow. Use Comfy Cloud.",
        ]
        report = {
            "os": sysname,
            "arch": arch,
            "system_ram_gb": ram_gb,
            "free_disk_gb": free_disk_gb,
            "wsl": False,
            "rosetta": False,
            "gpu": None,
            "verdict": "cloud",
            "recommended_install_path": "comfy-cloud",
            "comfy_cli_flag": None,
            "notes": notes,
            "install_urls": _install_urls(),
        }
        if check_pytorch:
            report["pytorch"] = check_pytorch_cuda()
        return report

    verdict, install_path, notes = classify(
        gpu, ram_gb, free_disk_gb, wsl=wsl, rosetta=rosetta,
    )

    report = {
        "os": sysname,
        "arch": arch,
        "system_ram_gb": ram_gb,
        "free_disk_gb": free_disk_gb,
        "wsl": wsl,
        "rosetta": rosetta,
        "gpu": gpu,
        "verdict": verdict,
        "recommended_install_path": install_path,
        "comfy_cli_flag": _COMFY_CLI_FLAG.get(install_path),
        "notes": notes,
        "install_urls": _install_urls(),
    }
    if check_pytorch:
        report["pytorch"] = check_pytorch_cuda()
    return report


def _install_urls() -> dict:
    return {
        "desktop": "https://docs.comfy.org/installation/desktop",
        "manual": "https://docs.comfy.org/installation/manual_install",
        "comfy_cli": "https://docs.comfy.org/comfy-cli/getting-started",
        "cloud": "https://platform.comfy.org",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Check whether this machine can run ComfyUI locally.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    p.add_argument("--check-pytorch", action="store_true",
                   help="Also probe `torch` for CUDA/MPS availability (slower)")
    args = p.parse_args(argv)

    report = build_report(check_pytorch=args.check_pytorch)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OS:        {report['os']} ({report['arch']})")
        if report.get("wsl"):
            print("Env:       WSL2")
        if report.get("rosetta"):
            print("Env:       Rosetta (x86_64 Python on Apple Silicon)")
        print(f"RAM:       {report['system_ram_gb']} GB")
        print(f"Free disk: {report['free_disk_gb']} GB (~/)")
        if report["gpu"]:
            g = report["gpu"]
            if g["vendor"] == "apple":
                print(f"GPU:       {g['name']} — {g.get('unified_memory_gb', 0)} GB unified memory")
            else:
                print(f"GPU:       {g['name']} — {g.get('vram_gb', 0)} GB VRAM")
                if g.get("all_gpus") and len(g["all_gpus"]) > 1:
                    print(f"           ({len(g['all_gpus'])} GPUs total; using best by VRAM)")
        else:
            print("GPU:       (none detected)")
        print(f"Verdict:   {report['verdict']}  → {report['recommended_install_path']}")
        if report["comfy_cli_flag"]:
            print(f"           run: comfy --skip-prompt install {report['comfy_cli_flag']}")
        if report.get("pytorch"):
            pt = report["pytorch"]
            if pt.get("available"):
                line = f"PyTorch:   {pt.get('torch_version')}"
                if pt.get("cuda_available"):
                    line += f" + CUDA ({pt.get('cuda_device_0', '?')})"
                if pt.get("mps_available"):
                    line += " + MPS"
                print(line)
            else:
                print(f"PyTorch:   not available — {pt.get('reason')}")
        for n in report["notes"]:
            print(f"  • {n}")

    if report["verdict"] == "ok":
        return 0
    if report["verdict"] == "marginal":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
