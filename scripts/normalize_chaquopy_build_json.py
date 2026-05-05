#!/usr/bin/env python3
"""Canonicalize Chaquopy's generated build.json metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: normalize_chaquopy_build_json.py PATH", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(data, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
