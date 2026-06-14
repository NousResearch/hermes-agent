"""Find a Python executable that can actually train with CUDA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROBE = r"""
import importlib.util, json, platform
report = {"python": platform.python_version(), "executable": __import__("sys").executable}
for name in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "unsloth", "axolotl"]:
    report[name] = bool(importlib.util.find_spec(name))
try:
    import torch
    report["torch_version"] = getattr(torch, "__version__", None)
    report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_memory_free_mib"] = int(free // (1024 * 1024))
        report["cuda_memory_total_mib"] = int(total // (1024 * 1024))
except Exception as exc:
    report["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
"""


def default_candidates() -> list[Path]:
    seen: set[str] = set()
    candidates: list[Path] = []

    def add(path: str | Path | None) -> None:
        if not path:
            return
        p = Path(path).expanduser()
        key = str(p).lower()
        if key not in seen and p.exists():
            seen.add(key)
            candidates.append(p)

    add(sys.executable)
    for name in ("python", "python3", "py"):
        add(shutil.which(name))
    user = Path.home()
    for path in (
        user / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe",
        user / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "python.exe",
        user / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "python.exe",
        Path("C:/Python314/python.exe"),
        Path("C:/Python312/python.exe"),
        Path("C:/Python311/python.exe"),
        Path(".venv/Scripts/python.exe"),
    ):
        add(path)
    return candidates


def probe_python(python: Path, timeout: int) -> dict[str, object]:
    try:
        proc = subprocess.run(
            [str(python), "-c", PROBE],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"executable": str(python), "ok": False, "error": f"timeout after {timeout}s"}
    output = proc.stdout.strip().splitlines()
    payload: dict[str, object]
    if output:
        try:
            value = json.loads(output[-1])
            payload = value if isinstance(value, dict) else {"raw": output[-1]}
        except json.JSONDecodeError:
            payload = {"raw": output[-1]}
    else:
        payload = {}
    payload["executable"] = str(python)
    payload["ok"] = proc.returncode == 0
    if proc.stderr.strip():
        payload["stderr_tail"] = proc.stderr.strip().splitlines()[-3:]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locate a CUDA-capable Python for Hermes operator LoRA training.")
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    candidates = args.candidate or default_candidates()
    reports = [probe_python(path, args.timeout) for path in candidates]
    for report in reports:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if any(report.get("cuda_available") is True for report in reports) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
