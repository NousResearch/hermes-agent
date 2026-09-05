#!/usr/bin/env python3
"""Confirm the wavespeed CLI is installed and signed in.

Prints one JSON object on stdout:
  {"cli": "0.4.8" | null, "signed_in": true | false | null, "hint": "..."}

Exit codes: 0 ready, 1 CLI missing, 2 CLI present but not signed in.
No network calls beyond what `wavespeed status` itself does; safe to run in tests
with a stubbed binary. Cross-platform (stdlib only).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

INSTALL_HINT = "Install with: npm install -g @wavespeed/cli"
LOGIN_HINT = "Ask the user to run: wavespeed login  (or set WAVESPEED_API_KEY)"


def _run(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def check(binary: str = "wavespeed") -> dict:
    path = shutil.which(binary)
    if not path:
        return {"cli": None, "signed_in": None, "hint": INSTALL_HINT}

    code, out = _run([path, "--version"])
    version = out.strip().splitlines()[0] if code == 0 and out.strip() else None

    if os.environ.get("WAVESPEED_API_KEY"):
        return {"cli": version, "signed_in": True, "hint": "WAVESPEED_API_KEY is set"}

    code, out = _run([path, "status"])
    lowered = out.lower()
    signed_in = code == 0 and "not" not in lowered.split("signed")[0][-8:] and "signed out" not in lowered
    return {
        "cli": version,
        "signed_in": signed_in,
        "hint": "ready" if signed_in else LOGIN_HINT,
    }


def main() -> int:
    result = check()
    print(json.dumps(result))
    if result["cli"] is None:
        return 1
    if not result["signed_in"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
