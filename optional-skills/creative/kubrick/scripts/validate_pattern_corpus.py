#!/usr/bin/env python3
"""Validate Kubrick JSON pattern sidecars using only the standard library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = {
    "pattern_id",
    "title",
    "domain",
    "source_tier",
    "source_refs",
    "observed_structure",
    "dramatic_operations",
    "cinematic_affordances",
    "misuse_risks",
    "mutation_requirements",
    "transferable_structure",
    "non_transferable_surface",
    "production_cost",
    "confidence",
}


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: invalid JSON: {exc}"]

    missing = sorted(REQUIRED - set(data))
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(missing)}")
    if data.get("pattern_id") != path.stem:
        errors.append(f"{path}: pattern_id must equal filename stem")
    if not data.get("observed_structure"):
        errors.append(f"{path}: observed_structure must not be empty")
    mutation = data.get("mutation_requirements", {})
    if mutation.get("required") is not True or not mutation.get("variables"):
        errors.append(f"{path}: mutation must be required and define variables")
    if not data.get("source_refs"):
        errors.append(f"{path}: provenance is required")
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append(f"{path}: confidence must be between 0 and 1")
    if not data.get("transferable_structure") or not data.get("non_transferable_surface"):
        errors.append(f"{path}: transferable and non-transferable structure are required")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Skill root; defaults to parent of scripts/")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    pattern_root = root / "references" / "patterns"
    paths = sorted(pattern_root.rglob("*.json"))
    errors: list[str] = []
    seen: dict[str, Path] = {}
    for path in paths:
        errors.extend(validate_file(path))
        try:
            pattern_id = json.loads(path.read_text(encoding="utf-8")).get("pattern_id")
        except Exception:
            continue
        if pattern_id in seen:
            errors.append(f"duplicate pattern_id {pattern_id}: {seen[pattern_id]} and {path}")
        elif pattern_id:
            seen[pattern_id] = path
    report = {
        "status": "PASS" if not errors else "FAIL",
        "pattern_count": len(paths),
        "unique_pattern_ids": len(seen),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
