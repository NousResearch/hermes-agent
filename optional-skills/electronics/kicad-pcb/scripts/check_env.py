#!/usr/bin/env python3
"""Preflight for the headless KiCad toolchain — Linux / Windows / macOS.

Usage:
  python3 check_env.py            # report; always exit 0 (informational)
  python3 check_env.py --strict   # exit 1 if anything critical is missing (for CI)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_paths as kp  # noqa: E402

OSNAME = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}.get(
    __import__("platform").system(), __import__("platform").system())

def check_python_pcbnew() -> str:
    for p in kp.python_with_pcbnew_candidates():
        if str(p) == sys.executable or p.exists():
            v = kp.verify_pcbnew(p, timeout=60)
            if v:
                return f"{p} ({v})"
    raise kp.ResolveError(
        "python+pcbnew",
        "install KiCad 10 (includes the pcbnew bindings) or set KICAD_PYTHON")


def check_kicad_cli() -> str:
    cli = kp.kicad_cli()
    r = subprocess.run([str(cli), "version"], capture_output=True, text=True, timeout=30)
    version = (r.stdout or r.stderr).strip()
    match = re.search(r"\b(\d+)\.(\d+)", version)
    if r.returncode != 0 or not match or int(match.group(1)) < 10:
        raise kp.ResolveError("kicad-cli >= 10", f"found {cli} ({version or 'unknown version'}); "
                              "install KiCad 10 or set KICAD_CLI to its executable")
    return f"{cli} ({version})"


CHECKS = [
    ("python+pcbnew", check_python_pcbnew),
    ("footprints dir", kp.footprints_dir),
    ("kicad-cli", check_kicad_cli),
    ("freerouting", lambda: kp.freerouting()[1]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything is missing")
    args = ap.parse_args()

    print(f"System: {OSNAME} | python: {sys.version.split()[0]} ({sys.executable})\n")
    ok_all = True
    missing: set[str] = set()
    for name, fn in CHECKS:
        try:
            val = fn()
            print(f"  OK  {name:16} -> {val}")
        except Exception as e:  # noqa: BLE001 — preflight must list everything
            ok_all = False
            missing.add(name)
            print(f"  X   {name:16} -> {e}")
    print()
    if ok_all:
        print("Toolchain complete. Next step: "
              "python3 demo_autoroute.py (full pipeline through DRC).")
    else:
        print("Missing components above (see the fix hints on each line).")
        if not missing - {"freerouting"}:
            print("KiCad itself is complete: `python simple_board.py` builds a full board "
                  "(ERC/DRC/gerbers) without Freerouting.")
    return 0 if (ok_all or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
