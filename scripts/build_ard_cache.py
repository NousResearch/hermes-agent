#!/usr/bin/env python3
"""ARD Registry Cache Builder — Periodic re-indexing of external ARD registries.

Fetches all entries from configured ARD registries and caches them locally
so search works offline and is faster for subsequent queries.

Usage:
    # Manual run
    python scripts/build_ard_cache.py

    # As a cronjob (via hermes cron)
    hermes cron add --name "ard-reindex" --schedule "0 */6 * * *" \\
        --command "python scripts/build_ard_cache.py"

    # With specific registries
    python scripts/build_ard_cache.py --registry https://custom-ard.com

Output:
    ~/.hermes/.hub/ard-cache.json — cached entries from all registries
    ~/.hermes/.hub/ard-cache.meta.json — metadata (timestamp, counts, sources)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure hermes-agent is importable
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))

from tools.skills_hub import (
    ArdSource,
    _get_ard_registries,
    SkillMeta,
    ARD_TYPE_SKILL,
    ARD_TYPE_MCP_SERVER,
    ARD_TYPE_MCP_SERVER_CARD,
)
from hermes_constants import get_hermes_home

logger = logging.getLogger("ard_cache_builder")

CACHE_DIR = get_hermes_home() / ".hub"
CACHE_FILE = CACHE_DIR / "ard-cache.json"
META_FILE = CACHE_DIR / "ard-cache.meta.json"

# Broad queries to enumerate most entries from a registry
SEED_QUERIES = [
    "",  # Empty query — some registries return everything
    "tool skill agent mcp server",
    "automation integration api",
    "analysis processing data",
    "generation creation building",
    "security monitoring detection",
    "search extraction scraping",
    "transcription translation language",
]


def fetch_all_entries(registries: list[str]) -> tuple[list[dict], dict]:
    """Fetch all entries from configured registries.

    Returns (entries, metadata) where metadata tracks per-registry stats.
    """
    src = ArdSource(registries=registries)
    all_entries: dict[str, dict] = {}  # identifier → entry dict
    meta: dict[str, dict] = {}

    for reg_url in registries:
        reg_start = time.time()
        reg_count = 0
        reg_errors = 0

        for query in SEED_QUERIES:
            try:
                results = src.search(query, limit=50)
                for r in results:
                    if r.identifier not in all_entries:
                        entry = {
                            "identifier": r.identifier,
                            "displayName": r.name,
                            "description": r.description,
                            "type": r.extra.get("ard_type", ARD_TYPE_SKILL),
                            "tags": r.tags,
                            "source": r.source,
                            "mcp": r.extra.get("mcp"),
                            "source_url": r.extra.get("source_url"),
                            "registry": reg_url,
                        }
                        all_entries[r.identifier] = entry
                        reg_count += 1
            except Exception as e:
                logger.debug("Query '%s' on %s failed: %s", query, reg_url, e)
                reg_errors += 1

        elapsed = time.time() - reg_start
        meta[reg_url] = {
            "entries": reg_count,
            "errors": reg_errors,
            "elapsed_seconds": round(elapsed, 2),
        }
        print(f"  {reg_url}: {reg_count} entries ({elapsed:.1f}s)")

    return list(all_entries.values()), meta


def write_cache(entries: list[dict], meta: dict) -> Path:
    """Write cache + metadata to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "entries": entries,
    }

    CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")

    meta_data = {
        "timestamp": cache_data["timestamp"],
        "total_entries": len(entries),
        "registries": meta,
        "cache_file": str(CACHE_FILE),
        "cache_size_bytes": CACHE_FILE.stat().st_size,
    }
    META_FILE.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

    return CACHE_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Build ARD registry cache for offline/faster search"
    )
    parser.add_argument(
        "--registry",
        action="append",
        help="Specific registry URL (can repeat). Default: all from config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON summary instead of human-readable text.",
    )
    args = parser.parse_args()

    registries = args.registry or _get_ard_registries()
    if not registries:
        print("No ARD registries configured. Set skills_hub.ard_registries in config.yaml.")
        sys.exit(1)

    print(f"ARD Cache Builder — querying {len(registries)} registry(ies)")
    start = time.time()

    entries, meta = fetch_all_entries(registries)
    cache_path = write_cache(entries, meta)

    elapsed = time.time() - start
    total = len(entries)

    # Count by type
    from collections import Counter
    types = Counter(e.get("type", "unknown").split("/")[-1] for e in entries)

    if args.json:
        print(json.dumps({
            "cache_file": str(cache_path),
            "meta_file": str(META_FILE),
            "total_entries": total,
            "types": dict(types),
            "registries": meta,
            "elapsed_seconds": round(elapsed, 2),
        }, indent=2))
    else:
        print()
        print(f"ARD Cache Built:")
        print(f"  Entries: {total}")
        print(f"  Types: {dict(types)}")
        print(f"  Cache: {cache_path}")
        print(f"  Meta: {META_FILE}")
        print(f"  Size: {cache_path.stat().st_size:,} bytes")
        print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
