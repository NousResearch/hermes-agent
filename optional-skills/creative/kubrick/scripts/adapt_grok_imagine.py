#!/usr/bin/env python3
"""Compatibility wrapper for the shared Grok Imagine provider adapter."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def main() -> None:
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(ROOT / "adapt_provider.py"), *sys.argv[1:], "--provider", "grok-imagine"]
        )
    )


if __name__ == "__main__":
    main()
