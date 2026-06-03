#!/usr/bin/env python3
"""Deterministic domain-surface inventory for code-scan outputs.

This module is intentionally path/metadata based. It does not execute project
code, parse framework semantics, validate security posture, or infer runtime
correctness. Every emitted item is labelled as an inventory-only fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import PurePosixPath
from typing import Iterable

CLAIM_TYPE = "deterministic_inventory"
SEMANTIC_STATUS = "not_validated"

DOMAIN_SURFACE_PATTERNS = [
    "supabase/migrations/*.sql",
    "supabase/functions/*/index.{ts,js}",
    "vite.config.{ts,js,mts,mjs,cjs}",
    "sw.js",
    "service-worker.js",
    "manifest.json",
    "manifest.webmanifest",
    "public/manifest.webmanifest",
    ".github/workflows/*.{yml,yaml}",
    "vercel.json",
    "netlify.toml",
    "package.json scripts",
]


def _iter_file_paths(scan_data: dict) -> Iterable[str]:
    """Yield normalized relative paths from a scan_project-style payload."""
    for record in scan_data.get("files", []) or []:
        raw = record.get("relative_path") or record.get("path")
        if not raw:
            continue
        yield str(PurePosixPath(str(raw).replace("\\", "/")))


def _classify_surface(path: str) -> str | None:
    """Classify a normalized path into a deterministic surface type."""
    lower = path.lower()
    name = PurePosixPath(lower).name

    if lower.startswith("supabase/migrations/") and lower.endswith(".sql"):
        return "supabase_migration"
    if lower.startswith("supabase/functions/") and name in {"index.ts", "index.js"}:
        return "supabase_edge_function"
    if name.startswith("vite.config."):
        return "vite_config"
    if lower in {"sw.js", "service-worker.js"} or lower.endswith("/service-worker.js"):
        return "service_worker"
    if name in {"manifest.json", "manifest.webmanifest"}:
        return "pwa_manifest"
    if lower.startswith(".github/workflows/") and lower.endswith((".yml", ".yaml")):
        return "ci_workflow"
    if lower == "vercel.json" or lower.endswith("/vercel.json"):
        return "vercel_config"
    if lower == "netlify.toml" or lower.endswith("/netlify.toml"):
        return "netlify_config"
    if lower == "package.json" or lower.endswith("/package.json"):
        return "package_scripts"
    return None


def build_domain_surfaces_summary(surfaces: list[dict]) -> dict:
    """Build stable summary counts for emitted surfaces."""
    counts = Counter(surface["surface"] for surface in surfaces)
    return {
        "total_surfaces": len(surfaces),
        "surface_types": dict(sorted(counts.items())),
    }


def scan_domain_surfaces(scan_data: dict) -> dict:
    """Return deterministic inventory of recognizable domain surfaces."""
    surfaces: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for path in sorted(_iter_file_paths(scan_data)):
        surface = _classify_surface(path)
        if surface is None:
            continue
        key = (surface, path)
        if key in seen:
            continue
        seen.add(key)
        surfaces.append(
            {
                "surface": surface,
                "path": path,
                "claim_type": CLAIM_TYPE,
                "semantic_status": SEMANTIC_STATUS,
            }
        )

    return {
        "surfaces": surfaces,
        "summary": build_domain_surfaces_summary(surfaces),
        "claim_type": CLAIM_TYPE,
        "semantic_status": SEMANTIC_STATUS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic domain-surface inventory from scan_project JSON."
    )
    parser.add_argument("scan_json", nargs="?", help="Path to scan_project JSON; stdin when omitted")
    args = parser.parse_args(argv)

    try:
        if args.scan_json:
            with open(args.scan_json, "r", encoding="utf-8") as fh:
                scan_data = json.load(fh)
        else:
            scan_data = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(scan_domain_surfaces(scan_data), indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
