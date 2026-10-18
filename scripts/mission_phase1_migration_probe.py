#!/usr/bin/env python3
"""Apply the additive ownership schema to a disposable Kanban copy and compare legacy tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from hermes_cli import mission_ownership
from scripts.mission_phase1_copy_board import _fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    args = parser.parse_args()
    baseline = _fingerprint(args.baseline)
    before = _fingerprint(args.working)
    if before["tables"] != baseline["tables"]:
        raise SystemExit("working copy differs from preserved baseline before migration")

    conn = sqlite3.connect(args.working)
    try:
        mission_ownership.ensure_schema(conn)
        conn.commit()
    finally:
        conn.close()

    after = _fingerprint(args.working)
    if any(after["tables"].get(name) != value for name, value in baseline["tables"].items()):
        raise SystemExit("additive mission migration changed an existing Kanban table")
    mission_tables = {name: value for name, value in after["tables"].items() if name.startswith("mission")}
    required = {"missions", "mission_history", "mission_owner_history",
                "mission_execution_leases", "mission_validations"}
    if not required.issubset(mission_tables):
        raise SystemExit(f"migration omitted mission tables: {sorted(required - set(mission_tables))}")

    # A rollback to a pre-mission consumer is additive: retain ownership data and prove the
    # legacy Kanban tables remain queryable without dropping newer rows or mission tables.
    conn = sqlite3.connect(args.working)
    try:
        task_count = conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        mission_count = conn.execute("SELECT count(*) FROM missions").fetchone()[0]
        table_names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()
    if task_count != baseline["tables"]["tasks"]["rows"] or mission_count != 0:
        raise SystemExit("rollback compatibility check changed task or new-mission counts")
    if not required.issubset(table_names):
        raise SystemExit("rollback compatibility check found a removed ownership table")
    print(json.dumps({
        "status": "passed",
        "preserved_legacy_tables": len(baseline["tables"]),
        "legacy_task_count": task_count,
        "mission_tables_added": sorted(required),
        "mission_rows": mission_count,
        "rollback": "legacy Kanban tables remain queryable; additive mission tables retained",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
