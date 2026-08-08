#!/usr/bin/env python3
"""Validate a WSL Photoshop handoff and optionally open an image in Photoshop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_PHOTOSHOP = Path("/mnt/c/Program Files/Adobe/Adobe Photoshop 2025/Photoshop.exe")


def is_wsl() -> bool:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower() or "wsl" in release.lower()


def wsl_to_windows_path(path: Path) -> str:
    parts = path.resolve().parts
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])
    raise ValueError("input must be located under /mnt/<drive>/ so Windows can open it")


def prerequisites(input_path: Path, photoshop_path: Path) -> dict[str, object]:
    missing: list[str] = []
    if not is_wsl():
        missing.append("WSL is required")
    if not shutil.which("powershell.exe"):
        missing.append("powershell.exe is not reachable from WSL")
    if not photoshop_path.is_file():
        missing.append(f"Photoshop executable not found: {photoshop_path}")
    if not input_path.is_file():
        missing.append(f"input image not found: {input_path}")
    else:
        try:
            wsl_to_windows_path(input_path)
        except ValueError as exc:
            missing.append(str(exc))
    return {
        "ok": not missing,
        "input": str(input_path),
        "photoshop": str(photoshop_path),
        "missing": missing,
    }


def launch(input_path: Path, photoshop_path: Path) -> dict[str, object]:
    result = prerequisites(input_path, photoshop_path)
    if not result["ok"]:
        return result

    windows_input = wsl_to_windows_path(input_path)
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "Start-Process -FilePath $args[0] -ArgumentList $args[1]",
        str(photoshop_path).replace("/mnt/c", "C:\\").replace("/", "\\"),
        windows_input,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result.update(
        launched=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )
    result["ok"] = bool(result["launched"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image under /mnt/<drive>/ to check or open")
    parser.add_argument("--photoshop-path", type=Path, default=Path(os.environ.get("PHOTOSHOP_EXE", DEFAULT_PHOTOSHOP)))
    parser.add_argument("--check", action="store_true", help="Validate prerequisites without starting Photoshop")
    args = parser.parse_args(argv)

    result = prerequisites(args.input, args.photoshop_path) if args.check else launch(args.input, args.photoshop_path)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
