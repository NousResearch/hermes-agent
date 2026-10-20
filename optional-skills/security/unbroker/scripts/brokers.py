"""Load and query the broker database (references/brokers/*.json).

Each broker is one JSON file for clean diffs/PRs. Files beginning with `_` are
ignored (reserved for notes/scratch).
"""
from __future__ import annotations

import json
from pathlib import Path

import paths
import storage

PRIORITY_ORDER = {"crucial": 0, "high": 1, "standard": 2, "long_tail": 3}


def _load_curated(directory: Path | None = None) -> list[dict]:
    directory = directory or paths.brokers_dir()
    out: list[dict] = []
    if not directory.exists():
        return out
    for fp in sorted(directory.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        out.append(json.loads(fp.read_text(encoding="utf-8")))
    return out


def load_live_cache() -> list[dict]:
    """Records pulled from BADBOOL via `refresh-brokers` (empty until refreshed)."""
    return storage.read_json(paths.brokers_cache_path(), []) or []


def load_registry_cache() -> list[dict]:
    """CA Data Broker Registry records (separate coverage lane; empty until refreshed).

    Kept OUT of load_all() by default: these are not people-search sites to scan, they
    are worked via the CA DROP one-shot + CCPA email. Consumers of the scan/plan/fanout
    pipeline must not receive them; use this directly for coverage counts and the DROP/
    email lanes.
    """
    return storage.read_json(paths.registry_cache_path(), []) or []


def load_all(directory: Path | None = None, include_live: bool = True) -> list[dict]:
    """Curated records, with live BADBOOL records merged underneath (curated wins)."""
    merged: dict[str, dict] = {b["id"]: b for b in _load_curated(directory)}
    if include_live:
        for b in load_live_cache():
            bid = b.get("id")
            if bid and bid not in merged:
                merged[bid] = b
    out = list(merged.values())
    out.sort(key=lambda b: (PRIORITY_ORDER.get(b.get("priority", "standard"), 9), b.get("id", "")))
    return out


def get(broker_id: str, directory: Path | None = None) -> dict | None:
    for b in load_all(directory):
        if b.get("id") == broker_id:
            return b
    return None


def by_priority(*levels: str, directory: Path | None = None) -> list[dict]:
    wanted = set(levels) if levels else None
    return [b for b in load_all(directory) if wanted is None or b.get("priority") in wanted]


def clusters(directory: Path | None = None) -> dict[str, list[str]]:
    """Map a parent broker id -> child site ids it can clear (force-multipliers)."""
    out: dict[str, list[str]] = {}
    for b in load_all(directory):
        owns = b.get("owns") or []
        if owns:
            out[b["id"]] = list(owns)
    return out
