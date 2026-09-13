#!/usr/bin/env python3
"""ORCH V4 Sidecar Init — create-only new orch_v4.db.

Hard rules:
- Sidecar path must not exist (atomic O_CREAT|O_EXCL|O_NOFOLLOW)
- Receipt path must not exist (atomic create-only)
- Native DB opened mode=ro only
- No native schema mutation
"""

from __future__ import annotations

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


def _atomic_create_file(path: Path, mode: int = 0o600) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, mode)
    os.close(fd)


def _atomic_write_json(path: Path, payload: dict) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(str(path), flags, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


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

    if sidecar_path.exists() or os.path.lexists(sidecar_path):
        print(f"ERROR: sidecar path already exists: {sidecar_path}", file=sys.stderr)
        return 1
    if receipt_path.exists() or os.path.lexists(receipt_path):
        print(f"ERROR: receipt path already exists: {receipt_path}", file=sys.stderr)
        return 1
    if not native_path.exists():
        print(f"ERROR: native DB not found: {native_path}", file=sys.stderr)
        return 1

    # Refuse live default native path unless operator is explicit via env.
    live = Path.home() / ".hermes" / "kanban.db"
    if native_path.resolve() == live.resolve() and os.environ.get("ORCH_V4_ALLOW_LIVE") != "1":
        print("ERROR: live native path forbidden without ORCH_V4_ALLOW_LIVE=1", file=sys.stderr)
        return 2

    native_sha_before = sha256_file(str(native_path))
    native_conn = sqlite3.connect(f"file:{native_path}?mode=ro", uri=True)
    try:
        native_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.Error as e:
        print(f"ERROR: native DB read failed: {e}", file=sys.stderr)
        return 1
    finally:
        native_conn.close()

    try:
        _atomic_create_file(sidecar_path)
    except FileExistsError:
        print(f"ERROR: sidecar path already exists: {sidecar_path}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ERROR: sidecar create failed: {e}", file=sys.stderr)
        return 1

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
        tables = get_sidecar_table_names(sidecar_conn)
        triggers = get_sidecar_trigger_names(sidecar_conn)
        verify_no_native_tables(sidecar_conn)
        table_count = len([t for t in tables if not t.startswith("sqlite_")])
        trigger_count = len(triggers)
    finally:
        sidecar_conn.close()

    native_sha_after = sha256_file(str(native_path))
    sidecar_sha = sha256_file(str(sidecar_path))
    if native_sha_before != native_sha_after:
        print("FATAL: native DB hash changed during init!", file=sys.stderr)
        sidecar_path.unlink(missing_ok=True)
        return 1

    receipt = {
        "init_id": f"sidecar-init-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "native_path": str(native_path),
        "native_sha256_before": native_sha_before,
        "native_sha256_after": native_sha_after,
        "native_mutated": False,
        "sidecar_table_count": table_count,
        "sidecar_trigger_count": trigger_count,
        "native_opened_mode": "ro",
        "create_mode": "O_CREAT|O_EXCL|O_NOFOLLOW",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(receipt_path, receipt)
    except FileExistsError:
        print(f"ERROR: receipt path already exists: {receipt_path}", file=sys.stderr)
        return 1

    print(f"Sidecar created: {sidecar_path}")
    print(f"  Tables: {table_count}, Triggers: {trigger_count}")
    print(f"  Native SHA-256: {native_sha_before[:16]}... (unchanged)")
    print(f"  Receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
