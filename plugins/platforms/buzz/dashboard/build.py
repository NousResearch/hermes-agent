#!/usr/bin/env python3
"""Dependency-free reproducible builder for the Buzz Dashboard assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ASSET_PAIRS = (
    (ROOT / "src" / "index.js", ROOT / "dist" / "index.js"),
    (ROOT / "src" / "style.css", ROOT / "dist" / "style.css"),
)


def check() -> int:
    stale = [
        str(output.relative_to(ROOT))
        for source, output in ASSET_PAIRS
        if not output.is_file() or output.read_bytes() != source.read_bytes()
    ]
    if stale:
        print("stale Buzz Dashboard assets: " + ", ".join(stale), file=sys.stderr)
        print("run plugins/platforms/buzz/dashboard/build.py", file=sys.stderr)
        return 1
    print("Buzz Dashboard build is reproducible and current")
    return 0


def build() -> int:
    for source, output in ASSET_PAIRS:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
    return check()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return check() if args.check else build()


if __name__ == "__main__":
    raise SystemExit(main())
