#!/usr/bin/env python3
"""Refresh the bundled models.dev snapshot (agent/models_dev_snapshot.json).

The snapshot is the offline-first fallback for ``agent.models_dev`` (P-028): it
guarantees the cost guard, capability lookup, and context-length resolution
have data even when ``https://models.dev/api.json`` is slow or blocked (e.g.
from mainland China), so model save/switch never stalls on the network.

Run this at release time (or whenever the catalog drifts) to refresh it:

    python3 scripts/refresh_models_dev_snapshot.py

Honours ``HERMES_MODELS_DEV_URL`` so a mirror can be used. Written minified +
key-sorted to match the disk-cache format and keep diffs reviewable.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

URL = os.getenv("HERMES_MODELS_DEV_URL", "https://models.dev/api.json")
DEST = Path(__file__).resolve().parent.parent / "agent" / "models_dev_snapshot.json"


def main() -> int:
    print(f"Fetching {URL} ...", file=sys.stderr)
    with urllib.request.urlopen(URL, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict) or not data:
        print("error: unexpected models.dev payload (not a non-empty dict)", file=sys.stderr)
        return 1
    with open(DEST, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    total_models = sum(
        len(p.get("models", {})) for p in data.values() if isinstance(p, dict)
    )
    print(
        f"wrote {DEST.relative_to(DEST.parent.parent)} "
        f"({len(data)} providers, {total_models} models, {DEST.stat().st_size} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
