#!/usr/bin/env python3
"""Classify #90950 WAL-sidecar experiments without overclaiming causality."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

NEGATIVE_SCENARIOS = {
    "open_probe",
    "set_wal",
    "fts_rebuild",
    "drop_recreate_fts",
}
POSITIVE_CONTROL = "forced_unlink"


def split_brain_signature(row: dict[str, Any]) -> bool:
    return bool(
        row.get("deleted_fd_observed")
        or row.get("wal_inode_replacements", 0)
        or row.get("shm_inode_replacements", 0)
    )


def database_damage(row: dict[str, Any]) -> bool:
    return row.get("integrity_check") != "ok" or row.get("quick_check") != "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("probe emitted no rows")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["sqlite_version"]][row["scenario"]].append(row)

    versions: dict[str, Any] = {}
    invalid_controls: list[str] = []
    ordinary_signatures: list[dict[str, Any]] = []
    ordinary_damage: list[dict[str, Any]] = []

    for version, scenarios in sorted(grouped.items()):
        missing = (NEGATIVE_SCENARIOS | {POSITIVE_CONTROL}) - set(scenarios)
        if missing:
            raise SystemExit(f"SQLite {version} missing scenarios: {sorted(missing)}")

        controls = scenarios[POSITIVE_CONTROL]
        control_valid = any(split_brain_signature(row) for row in controls)
        if not control_valid:
            invalid_controls.append(version)

        version_summary: dict[str, Any] = {}
        for scenario, samples in sorted(scenarios.items()):
            signatures = [row for row in samples if split_brain_signature(row)]
            damage = [row for row in samples if database_damage(row)]
            version_summary[scenario] = {
                "samples": len(samples),
                "split_brain_signatures": len(signatures),
                "database_damage": len(damage),
                "writer_errors": sum(int(row.get("writer_errors", 0)) for row in samples),
                "maintenance_errors": sum(
                    int(row.get("maintenance_errors", 0)) for row in samples
                ),
                "maintenance_busy": sum(
                    int(row.get("maintenance_busy", 0)) for row in samples
                ),
            }
            if scenario in NEGATIVE_SCENARIOS:
                ordinary_signatures.extend(signatures)
                ordinary_damage.extend(damage)
        versions[version] = {
            "positive_control_valid": control_valid,
            "scenarios": version_summary,
        }

    if invalid_controls:
        conclusion = (
            "invalid harness: forced-unlink positive control did not expose a "
            f"sidecar split-brain signature for {', '.join(invalid_controls)}"
        )
    elif ordinary_signatures:
        conclusion = (
            "reproduced the deleted-sidecar/inode-replacement signature without "
            "an intentional unlink; inspect the matching raw rows before assigning causality"
        )
    else:
        conclusion = (
            "did not reproduce the deleted-sidecar/inode-replacement signature in "
            "ordinary open, WAL-set, FTS rebuild, or FTS drop/recreate scenarios; "
            "the forced-unlink control did reproduce the signature"
        )

    summary = {
        "schema": "hermes.state-db-wal-sidecar-repro.v1",
        "sample_count": len(rows),
        "negative_scenarios": sorted(NEGATIVE_SCENARIOS),
        "positive_control": POSITIVE_CONTROL,
        "positive_control_valid_for_all_versions": not invalid_controls,
        "ordinary_split_brain_signature_count": len(ordinary_signatures),
        "ordinary_database_damage_count": len(ordinary_damage),
        "conclusion": conclusion,
        "limits": [
            "A negative stress run does not disprove a production race.",
            "The forced-unlink scenario validates observability; it is not evidence that Hermes unlinked a sidecar.",
            "Database integrity and the deleted-file fingerprint are reported separately.",
        ],
        "versions": versions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 2 if invalid_controls else 0


if __name__ == "__main__":
    raise SystemExit(main())
