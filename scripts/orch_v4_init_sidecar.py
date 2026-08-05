#!/usr/bin/env python3
"""ORCH V4 Sidecar Init — create-only new orch_v4.db.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §15.0 Sidecar Architecture Decision.

Usage:
    python scripts/orch_v4_init_sidecar.py \
        --sidecar-db /path/to/orch_v4.db \
        --native-db /path/to/kanban.db \
        --receipt /path/to/receipt.json

Hard rules:
- Sidecar path must not exist (O_EXCL)
- Native DB opened mode=ro only
- No native schema mutation
- Receipt records native_sha256 + sidecar_sha256 + native_mutated=false
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize ORCH V4 sidecar DB")
    parser.add_argument("--sidecar-db", required=True, help="Path for new sidecar orch_v4.db")
    parser.add_argument("--native-db", required=True, help="Path to native kanban.db (read-only)")
    parser.add_argument("--receipt", required=True, help="Path for init receipt JSON")
    parser.add_argument("--board-instance-id", default=None, help="Board instance ID (optional)")
    args = parser.parse_args()

    sidecar_path = Path(args.sidecar_db)
    native_path = Path(args.native_db)
    receipt_path = Path(args.receipt)

    # O_EXCL: sidecar must not exist
    if sidecar_path.exists():
        print(f"ERROR: sidecar path already exists: {sidecar_path}", file=sys.stderr)
        return 1

    # Native must exist
    if not native_path.exists():
        print(f"ERROR: native DB not found: {native_path}", file=sys.stderr)
        return 1

    # Hash native BEFORE (zero-mutation baseline)
    native_sha_before = sha256_file(str(native_path))

    # Open native RO (verify it works)
    native_conn = sqlite3.connect(f"file:{native_path}?mode=ro", uri=True)
    native_conn.row_factory = sqlite3.Row
    try:
        # Verify we can read
        native_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.Error as e:
        print(f"ERROR: native DB read failed: {e}", file=sys.stderr)
        return 1
    finally:
        native_conn.close()

    # Create sidecar DB
    sidecar_conn = sqlite3.connect(str(sidecar_path))
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from hermes_cli.kanban_orch_schema_sidecar import (
            apply_sidecar_schema,
            get_sidecar_table_names,
            get_sidecar_trigger_names,
            verify_no_native_tables,
        )

        apply_sidecar_schema(sidecar_conn)

        # Verify
        tables = get_sidecar_table_names(sidecar_conn)
        triggers = get_sidecar_trigger_names(sidecar_conn)
        verify_no_native_tables(sidecar_conn)

        # Count
        table_count = len([t for t in tables if not t.startswith("sqlite_")])
        trigger_count = len(triggers)
    finally:
        sidecar_conn.close()

    # Hash native AFTER (must be unchanged)
    native_sha_after = sha256_file(str(native_path))
    sidecar_sha = sha256_file(str(sidecar_path))

    if native_sha_before != native_sha_after:
        print("FATAL: native DB hash changed during init!", file=sys.stderr)
        # Rollback: delete sidecar
        sidecar_path.unlink(missing_ok=True)
        return 1

    # Write receipt
    receipt = {
        "init_id": f"sidecar-init-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "native_path": str(native_path),
        "native_sha256_before": native_sha_before,
        "native_sha256_after": native_sha_after,
        "native_mutated": native_sha_before != native_sha_after,
        "sidecar_table_count": table_count,
        "sidecar_trigger_count": trigger_count,
        "native_opened_mode": "ro",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    print(f"Sidecar created: {sidecar_path}")
    print(f"  Tables: {table_count}, Triggers: {trigger_count}")
    print(f"  Native SHA-256: {native_sha_before[:16]}... (unchanged: {native_sha_before == native_sha_after})")
    print(f"  Receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
