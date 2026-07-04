#!/usr/bin/env python3
"""Inspect an ARD entry without installing/registering it."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ard_candidate_risk_score import default_catalog_paths, find_entry_in_catalogs, score_entry


def _read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") if isinstance(data, dict) else []
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def find_entry_with_source(identifier: str, catalogs: list[Path] | None = None) -> tuple[dict[str, Any], Path] | None:
    for path in catalogs or default_catalog_paths():
        for entry in _read_entries(path):
            if entry.get("identifier") == identifier:
                return entry, path
    return None


def inspect_identifier(identifier: str, catalogs: list[Path] | None = None) -> dict[str, Any]:
    found = find_entry_with_source(identifier, catalogs)
    if not found:
        return {
            "schema": "hermes.ard.inspect.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "error": "not_found",
            "identifier": identifier,
            "searched_catalogs": [str(p) for p in (catalogs or default_catalog_paths())],
        }
    entry, source = found
    risk = score_entry(entry)
    risk.setdefault("next_action", "manual_register_after_review")
    return {
        "schema": "hermes.ard.inspect.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "identifier": identifier,
        "source_catalog": str(source),
        "entry": entry,
        "risk": risk,
        "install_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an ARD entry by URN")
    parser.add_argument("identifier")
    parser.add_argument("--catalog", type=Path, action="append", help="Catalog path. Repeatable. Defaults to profile ARD catalogs.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_identifier(args.identifier, catalogs=args.catalog)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if report["ok"]:
            risk = report["risk"]
            print(f"{args.identifier}: {risk['decision']} risk={risk['risk']} source={report['source_catalog']}")
            print(f"next_action={risk.get('next_action')}")
        else:
            print(f"{args.identifier}: not found")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
