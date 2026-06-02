#!/usr/bin/env python3
"""Phase 4 D6 — Delta reporting.

Compares two UA scan snapshots (and optional fingerprint snapshots) to
produce a structured JSON delta report on stdout.

Usage:
    python scripts/code-scan/delta_report.py <old-scan.json> <new-scan.json> \\
        [--old-fingerprints old.json] [--new-fingerprints new.json]

Stdout: JSON delta report.
Stderr: Errors and warnings only.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure scripts/code-scan is on sys.path for sibling imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Re-use Phase 3 fingerprint comparison logic
from fingerprints import compare_fingerprints

OUTPUT_SCHEMA_VERSION = "1.0.0"


def _load_json_file(path: str) -> dict:
    """Load and parse a JSON file, raising on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        raise


def _relative_paths(scan: dict) -> list[str]:
    """Extract sorted list of relative paths from a scan."""
    return sorted(f.get("relative_path", f.get("path", "")) for f in scan.get("files", []))


def _counter_map(scan: dict, key: str) -> dict[str, int]:
    """Build a sorted {label: count} map from scan[key].

    Handles both dict form (e.g. {"python": 3}) and list form (e.g. ["fastapi"]).
    For lists, returns {"item": 1, ...} for consistency.
    """
    value = scan.get(key, {})
    if isinstance(value, list):
        return {item: 1 for item in sorted(value)}
    if isinstance(value, dict):
        return {k: v for k, v in sorted(value.items())}
    return {}


def compute_delta(
    old_scan: dict,
    new_scan: dict,
    *,
    old_fingerprints: Optional[dict] = None,
    new_fingerprints: Optional[dict] = None,
) -> dict:
    """Compute a deterministic delta report between two scan snapshots.

    Args:
        old_scan: Parsed old scan.json dict.
        new_scan: Parsed new scan.json dict.
        old_fingerprints: Optional old fingerprint map dict.
        new_fingerprints: Optional new fingerprint map dict.

    Returns:
        Delta report dict matching the D6 output schema.
    """
    warnings: list[str] = []

    # ── Schema version checks ────────────────────────────────────────────
    old_schema = old_scan.get("schema_version")
    new_schema = new_scan.get("schema_version")
    if old_schema and old_schema != OUTPUT_SCHEMA_VERSION:
        warnings.append(f"Old scan has unexpected schema version: {old_schema}")
    if new_schema and new_schema != OUTPUT_SCHEMA_VERSION:
        warnings.append(f"New scan has unexpected schema version: {new_schema}")

    # ── File set comparison ───────────────────────────────────────────────
    old_paths = set(_relative_paths(old_scan))
    new_paths = set(_relative_paths(new_scan))

    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)
    common_count = len(old_paths & new_paths)

    files_delta = {
        "added": added,
        "removed": removed,
        "common_count": common_count,
    }

    # ── Language deltas ──────────────────────────────────────────────────
    old_langs = _counter_map(old_scan, "languages")
    new_langs = _counter_map(new_scan, "languages")
    all_langs = sorted(set(old_langs) | set(new_langs))

    languages_delta = {}
    for lang in all_langs:
        o = old_langs.get(lang, 0)
        n = new_langs.get(lang, 0)
        languages_delta[lang] = {"old": o, "new": n, "delta": n - o}

    # ── Category deltas ──────────────────────────────────────────────────
    old_cats = _counter_map(old_scan, "categories")
    new_cats = _counter_map(new_scan, "categories")
    all_cats = sorted(set(old_cats) | set(new_cats))

    categories_delta = {}
    for cat in all_cats:
        o = old_cats.get(cat, 0)
        n = new_cats.get(cat, 0)
        categories_delta[cat] = {"old": o, "new": n, "delta": n - o}

    # ── Framework deltas ─────────────────────────────────────────────────
    old_fws = set(_counter_map(old_scan, "frameworks").keys())
    new_fws = set(_counter_map(new_scan, "frameworks").keys())

    frameworks_delta = {
        "added": sorted(new_fws - old_fws),
        "removed": sorted(old_fws - new_fws),
    }

    # ── Propagate warnings from new scan ─────────────────────────────────
    new_warnings = new_scan.get("warnings", [])
    if isinstance(new_warnings, list):
        warnings.extend(new_warnings)

    # ── Fingerprint comparison (optional) ────────────────────────────────
    fp_summary: Optional[dict] = None
    if old_fingerprints is not None and new_fingerprints is not None:
        per_file = compare_fingerprints(old_fingerprints, new_fingerprints)
        counts = {"UNCHANGED": 0, "COSMETIC": 0, "STRUCTURAL": 0}
        for classification in per_file.values():
            counts[classification] = counts.get(classification, 0) + 1
        fp_summary = counts

    # ── Assemble result ──────────────────────────────────────────────────
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "files": files_delta,
        "languages": languages_delta,
        "categories": categories_delta,
        "frameworks": frameworks_delta,
        "fingerprints": fp_summary,
        "warnings": sorted(set(warnings)),  # deduplicated, sorted
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two UA scan snapshots and produce a JSON delta report.",
    )
    parser.add_argument("old_scan", help="Path to old scan.json")
    parser.add_argument("new_scan", help="Path to new scan.json")
    parser.add_argument(
        "--old-fingerprints",
        default=None,
        help="Path to old fingerprints JSON",
    )
    parser.add_argument(
        "--new-fingerprints",
        default=None,
        help="Path to new fingerprints JSON",
    )

    args = parser.parse_args()

    # Load scans
    try:
        old_scan = _load_json_file(args.old_scan)
        new_scan = _load_json_file(args.new_scan)
    except (FileNotFoundError, json.JSONDecodeError):
        sys.exit(1)

    # Load optional fingerprints
    old_fp: Optional[dict] = None
    new_fp: Optional[dict] = None

    if args.old_fingerprints and args.new_fingerprints:
        try:
            old_fp = _load_json_file(args.old_fingerprints)
            new_fp = _load_json_file(args.new_fingerprints)
        except (FileNotFoundError, json.JSONDecodeError):
            sys.exit(1)
    elif args.old_fingerprints or args.new_fingerprints:
        print(
            "Warning: --old-fingerprints and --new-fingerprints must both be provided; skipping fingerprint comparison.",
            file=sys.stderr,
        )

    result = compute_delta(old_scan, new_scan, old_fingerprints=old_fp, new_fingerprints=new_fp)
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
