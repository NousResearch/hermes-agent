#!/usr/bin/env python3
"""Create a read-only SQLite backup and working copy inside the active Phase 1 sandbox."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(value):
    if isinstance(value, bytes):
        return {"blob_sha256": hashlib.sha256(value).hexdigest(), "blob_size": len(value)}
    return value


def _fingerprint(path: Path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        result = {"schema_version": conn.execute("PRAGMA schema_version").fetchone()[0], "tables": {}}
        for table in tables:
            escaped = table.replace('"', '""')
            create_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,),
            ).fetchone()[0]
            rows = [json.dumps([_value(v) for v in row], sort_keys=True, separators=(",", ":"))
                    for row in conn.execute(f'SELECT * FROM "{escaped}"')]
            digest = hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()
            result["tables"][table] = {
                "rows": len(rows), "rows_sha256": digest,
                "schema_sha256": hashlib.sha256((create_sql or "").encode()).hexdigest(),
            }
        return result
    finally:
        conn.close()


def _backup(source: Path, destination: Path) -> None:
    uri = f"file:{source.as_posix()}?mode=ro"
    src = sqlite3.connect(uri, uri=True, timeout=10)
    dst = sqlite3.connect(destination, timeout=10)
    try:
        src.execute("PRAGMA query_only=ON")
        src.backup(dst, pages=256, sleep=0.05)
    finally:
        dst.close()
        src.close()


def _same_logical_board(left: dict, right: dict) -> bool:
    # schema_version is SQLite's per-database cache invalidation counter, not a logical schema
    # identity; online backup can assign a fresh value while preserving all table definitions/data.
    return left["tables"] == right["tables"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    args = parser.parse_args()
    root_value = os.environ.get("HERMES_ISOLATION_ROOT", "").strip()
    if not root_value:
        raise SystemExit("missing HERMES_ISOLATION_ROOT")
    root = Path(root_value).resolve(strict=True)
    source = args.source.resolve(strict=True)
    baseline = args.baseline.resolve(strict=False)
    working = args.working.resolve(strict=False)
    for target in (baseline, working):
        if not target.is_relative_to(root) or target == source:
            raise SystemExit(f"refusing database output outside isolated root: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        for sidecar in (Path(str(target) + "-wal"), Path(str(target) + "-shm")):
            if sidecar.exists():
                raise SystemExit(f"refusing to overwrite an existing SQLite sidecar: {sidecar}")
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing board copy: {target}")

    baseline_fp = _fingerprint(source)
    _backup(source, baseline)
    _backup(baseline, working)
    copied_fp = _fingerprint(baseline)
    if not _same_logical_board(baseline_fp, copied_fp):
        raise SystemExit("backup content fingerprint differs from the read-only source")
    working_fp = _fingerprint(working)
    if not _same_logical_board(baseline_fp, working_fp):
        raise SystemExit("working copy differs from the preserved baseline copy")
    record = {
        "source": str(source),
        "baseline_copy": str(baseline),
        "working_copy": str(working),
        "source_sha256_at_copy": _sha(source),
        "baseline_sha256": _sha(baseline),
        "working_sha256": _sha(working),
        "source_schema_version": baseline_fp["schema_version"],
        "backup_schema_version": copied_fp["schema_version"],
        "fingerprint": baseline_fp,
    }
    record_path = root / "migration" / "board-baseline.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": str(baseline), "working": str(working),
                      "tables": len(baseline_fp["tables"]),
                      "row_counts": {k: v["rows"] for k, v in baseline_fp["tables"].items()},
                      "mission_tables_before": [k for k in baseline_fp["tables"] if k.startswith("mission")],
                      "baseline_sha256": record["baseline_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
