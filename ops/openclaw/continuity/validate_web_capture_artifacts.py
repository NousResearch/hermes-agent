#!/usr/bin/env python3
"""Validate all required web capture artifacts exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if "--index" not in sys.argv:
        print("ERROR: --index required", file=sys.stderr)
        return 1

    idx_path = Path(sys.argv[sys.argv.index("--index") + 1])
    if not idx_path.exists():
        print(f"ERROR: Index not found: {idx_path}", file=sys.stderr)
        return 1

    try:
        idx = json.loads(idx_path.read_text())
    except Exception as e:
        print(f"ERROR: Failed to parse index: {e}", file=sys.stderr)
        return 1

    base = idx_path.parent
    artifacts = idx.get("artifacts", {})
    required_keys = ["screenshot_png", "dom_snapshot_html", "execution_trace_json"]

    ok = True
    for key in required_keys:
        if key not in artifacts:
            print(f"ERROR: Required artifact key missing: {key}", file=sys.stderr)
            ok = False
            continue

        rel = artifacts[key]
        if not isinstance(rel, str) or not rel.strip():
            print(f"ERROR: Artifact path is empty for key: {key}", file=sys.stderr)
            ok = False
            continue

        path = base / rel
        if not path.exists():
            print(f"ERROR: Artifact file not found: {key} -> {path}", file=sys.stderr)
            ok = False
            continue

        if not path.is_file():
            print(f"ERROR: Artifact path is not a file: {key} -> {path}", file=sys.stderr)
            ok = False
            continue

    if not ok:
        return 1

    print(f"✓ All {len(required_keys)} required artifacts found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())