#!/usr/bin/env python3
"""SD3 adapter entrypoint — syntax-only translation over the neutral packet."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    args = sys.argv[1:]
    if "--provider" not in args:
        args = ["--provider", "sd3", *args]
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / "adapt_provider.py"), *args]))


if __name__ == "__main__":
    main()
