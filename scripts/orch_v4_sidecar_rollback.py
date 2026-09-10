#!/usr/bin/env python3
"""Rollback live ORCH V4 sidecar control plane (delete sidecar + disable writer).

Does NOT mutate native kanban.db.
Does NOT remove hermes-agent source modules.

Usage:
  python scripts/orch_v4_sidecar_rollback.py --dry-run
  python scripts/orch_v4_sidecar_rollback.py --apply --i-understand-destructive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SIDECAR = Path.home() / ".hermes" / "orch_v4.db"
DEFAULT_WRITER = Path.home() / ".hermes" / "orch_v4_writer.json"
NATIVE = Path.home() / ".hermes" / "kanban.db"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), flags, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rollback live ORCH V4 sidecar")
    p.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    p.add_argument("--writer-cfg", default=str(DEFAULT_WRITER))
    p.add_argument("--receipt", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    p.add_argument("--i-understand-destructive", action="store_true")
    args = p.parse_args(argv)

    sidecar = Path(args.sidecar)
    writer = Path(args.writer_cfg)
    receipt_path = Path(args.receipt)

    native_before = sha256_file(NATIVE)
    plan = {
        "action": "sidecar_rollback",
        "mode": "dry-run" if args.dry_run else "apply",
        "sidecar_path": str(sidecar),
        "sidecar_exists": sidecar.exists() or os.path.lexists(sidecar),
        "sidecar_sha256_before": sha256_file(sidecar),
        "writer_cfg_path": str(writer),
        "writer_exists": writer.exists() or os.path.lexists(writer),
        "writer_sha256_before": sha256_file(writer),
        "native_path": str(NATIVE),
        "native_sha256_before": native_before,
        "will_delete": [],
    }
    if plan["sidecar_exists"]:
        plan["will_delete"].append(str(sidecar))
        # also wal/shm if present
        for suf in ("-wal", "-shm", "-journal"):
            extra = Path(str(sidecar) + suf)
            if extra.exists() or os.path.lexists(extra):
                plan["will_delete"].append(str(extra))
    if plan["writer_exists"]:
        plan["will_delete"].append(str(writer))

    if args.dry_run:
        plan["result"] = "dry_run_only"
        plan["sealed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(receipt_path, plan)
        print(json.dumps(plan, indent=2))
        return 0

    if not args.i_understand_destructive:
        print("REFUSED: need --i-understand-destructive with --apply", file=sys.stderr)
        return 2

    deleted = []
    errors = []
    for path_s in plan["will_delete"]:
        path = Path(path_s)
        try:
            if path.is_symlink() or path.exists():
                path.unlink()
                deleted.append(path_s)
        except OSError as exc:
            errors.append(f"{path_s}:{exc}")

    native_after = sha256_file(NATIVE)
    out = {
        **plan,
        "mode": "apply",
        "deleted": deleted,
        "errors": errors,
        "native_sha256_after": native_after,
        "native_mutated": native_after != native_before,
        "sidecar_exists_after": sidecar.exists() or os.path.lexists(sidecar),
        "writer_exists_after": writer.exists() or os.path.lexists(writer),
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(time.time()),
    }
    if out["native_mutated"]:
        print("FATAL: native hash changed during rollback", file=sys.stderr)
        atomic_json(receipt_path, out)
        return 3
    if errors or out["sidecar_exists_after"] or out["writer_exists_after"]:
        atomic_json(receipt_path, out)
        print(json.dumps(out, indent=2))
        return 4
    out["result"] = "rolled_back"
    atomic_json(receipt_path, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
