#!/usr/bin/env python3
"""Fail-closed retention for verified daily backup archives.

The policy retains the newest 14 archives only when every retained archive
passes the same manifest/checksum/extractability checks as backup health.
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


BACKUP_ROOT = Path(os.path.expanduser("~/backups/daily"))
LEDGER_PATH = Path(os.path.expanduser("~/backups/retention-ledger.jsonl"))
RETENTION_COUNT = 14


def _archive_stem(archive: Path) -> str:
    return archive.name.removesuffix(".tar.gz")


def _manifest_path(archive: Path, backup_root: Path) -> Path:
    return backup_root / f"{_archive_stem(archive)}.manifest.json"


def _timestamp(archive: Path) -> datetime:
    stem = _archive_stem(archive)
    if not stem.startswith("kensei-"):
        raise ValueError(f"Cannot parse timestamp from filename: {archive.name}")
    try:
        return datetime.strptime(stem.removeprefix("kensei-"), "%Y%m%d-%H%M")
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp from filename: {archive.name}") from exc


def _health_module():
    path = Path(__file__).with_name("backup-health-check.py")
    spec = importlib.util.spec_from_file_location("backup_health_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_archive(archive: Path, backup_root: Path) -> str | None:
    try:
        health = _health_module()
        manifest_error = health.check_manifest(archive, _manifest_path(archive, backup_root))
        if manifest_error:
            return manifest_error
        return health.check_extractable(archive)
    except OSError as error:
        return f"Cannot verify archive: {error}"


def _archives(backup_root: Path) -> list[Path]:
    archives = list(backup_root.glob("kensei-*.tar.gz"))
    return sorted(archives, key=_timestamp, reverse=True)


def _append_ledger(ledger_path: Path, entry: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, sort_keys=True) + "\n")


def _failure_entry(reason: str, archives: list[Path]) -> dict:
    return {
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unverified_archives": [archive.name for archive in archives],
    }


def run_retention(backup_root: Path, ledger_path: Path, *, dry_run: bool = False) -> int:
    try:
        archives = _archives(backup_root)
    except ValueError as error:
        print(f"ALERT: {error}")
        return 2

    if len(archives) <= RETENTION_COUNT:
        return 0

    keep = archives[:RETENTION_COUNT]
    invalid_keep = [archive for archive in keep if _verify_archive(archive, backup_root)]
    if invalid_keep:
        _append_ledger(ledger_path, _failure_entry("fail_closed_verification_gap", invalid_keep))
        print("ALERT: retention set not fully verified; aborting")
        return 2

    delete = archives[RETENTION_COUNT:]
    for archive in delete:
        manifest = _manifest_path(archive, backup_root)
        try:
            size_bytes = archive.stat().st_size
        except OSError as error:
            print(f"ALERT: Failed to plan removal for {archive.name}: {error}")
            return 1
        entry = {
            "archive": archive.name,
            "dry_run": dry_run,
            "reason": "retention_policy_14_daily_dryrun" if dry_run else "retention_policy_14_daily",
            "size_bytes": size_bytes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if dry_run:
            entry["manifest_deleted"] = False
            _append_ledger(ledger_path, entry)
            continue
        try:
            archive.unlink()
            entry["manifest_deleted"] = False
            if manifest.exists():
                manifest.unlink()
                entry["manifest_deleted"] = True
            _append_ledger(ledger_path, entry)
        except OSError as error:
            print(f"ALERT: Failed to remove {archive.name}: {error}")
            return 1

    if dry_run:
        print(f"DRY RUN: would remove {len(delete)}, retain {len(keep)}")
    else:
        print(f"Retention: removed {len(delete)}, retained {len(keep)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=BACKUP_ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run_retention(args.backup_root, args.ledger, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
