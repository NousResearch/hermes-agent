#!/usr/bin/env python3
"""Migrate Torben's legacy action-ledger array to the append-only journal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()


def _repo_root() -> Path:
    current = SCRIPT_PATH
    for parent in current.parents:
        if (parent / "hermes_cli").exists():
            return parent
        if parent.name == ".hermes" and (parent / "hermes-agent" / "hermes_cli").exists():
            return parent / "hermes-agent"
    fallback = os.getenv("HERMES_REPO_ROOT")
    if fallback:
        return Path(fallback)
    return Path("/Users/ericfreeman/.hermes/hermes-agent")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.action_ledger import ActionLedger, ActionRecord  # noqa: E402

LEDGER_STEM = "torben-action-ledger"
LEGACY_LEDGER_NAME = f"{LEDGER_STEM}.json"
JOURNAL_LEDGER_NAME = f"{LEDGER_STEM}.jsonl"


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_legacy_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if not isinstance(payload, list):
        raise ValueError(f"Legacy ledger must contain a JSON array: {path}")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Legacy ledger record {index} must be an object")
        records.append(dict(item))
    return records


def _migrate_record(payload: dict[str, Any], *, migrated_at: str) -> dict[str, Any]:
    migrated = dict(payload)
    if migrated.get("status") == "approved":
        history = list(migrated.get("resolution_history") or [])
        history.append(
            {
                "at": migrated_at,
                "status": "migration_status_demoted",
                "previous_status": "approved",
                "new_status": "approval_required",
                "reason": "P0-4 migration safety gate: carried-forward approvals must be re-validated.",
            }
        )
        migrated["resolution_history"] = history
        migrated["status"] = "approval_required"
    return ActionRecord.from_dict(migrated).to_dict()


def _write_journal(records: list[dict[str, Any]], journal_path: Path, *, force: bool) -> None:
    if journal_path.exists() and not force:
        raise FileExistsError(f"Journal already exists; pass --force to replace it: {journal_path}")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = journal_path.with_name(f".{journal_path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, journal_path)


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, report_path)


def migrate_legacy_ledger(
    *,
    legacy_path: Path,
    journal_path: Path,
    report_path: Path,
    apply: bool,
    force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    migrated_at = _iso(now)
    legacy_records = _load_legacy_records(legacy_path)
    migrated_records = [_migrate_record(record, migrated_at=migrated_at) for record in legacy_records]
    status_before = Counter(str(record.get("status")) for record in legacy_records)
    status_after = Counter(str(record.get("status")) for record in migrated_records)
    approved_demoted = sum(1 for record in legacy_records if record.get("status") == "approved")
    approved_remaining = sum(1 for record in migrated_records if record.get("status") == "approved")
    handles = [str(record.get("handle")) for record in migrated_records]
    duplicate_handles = sorted(handle for handle, count in Counter(handles).items() if count > 1)
    backup_path = legacy_path.with_name(f"{legacy_path.name}.bak.{migrated_at.replace(':', '').replace('-', '')}.p0-4-migration")
    ledger = ActionLedger(journal_path)
    hold_path = ledger.hold_path

    report: dict[str, Any] = {
        "schema": "torben.ledger-migration.v1",
        "status": "planned" if not apply else "migrated",
        "generated_at": migrated_at,
        "legacy_path": str(legacy_path),
        "legacy_backup_path": str(backup_path) if apply else None,
        "journal_path": str(journal_path),
        "snapshot_path": str(ledger.snapshot_path),
        "report_path": str(report_path),
        "legacy_record_count": len(legacy_records),
        "journal_record_count": len(migrated_records),
        "record_loss": len(legacy_records) - len(migrated_records),
        "status_counts_before": dict(sorted(status_before.items())),
        "status_counts_after": dict(sorted(status_after.items())),
        "approved_demoted": approved_demoted,
        "approved_remaining": approved_remaining,
        "duplicate_handles": duplicate_handles,
        "migration_safety_gate": "pass" if approved_remaining == 0 and not duplicate_handles else "fail",
    }
    if report["record_loss"] != 0 or approved_remaining != 0 or duplicate_handles:
        report["status"] = "failed"
        if apply:
            _write_report(report, report_path)
        return report

    if not apply:
        return report

    hold_path.write_text(f"P0-4 migration started {migrated_at}\n", encoding="utf-8")
    try:
        shutil.copy2(legacy_path, backup_path)
        _write_journal(migrated_records, journal_path, force=force)
        ledger._write_snapshot([ActionRecord.from_dict(record) for record in migrated_records])
        try:
            legacy_path.chmod(0o444)
        except PermissionError:
            report["legacy_chmod_warning"] = "permission_denied"
        _write_report(report, report_path)
    finally:
        if hold_path.exists():
            hold_path.unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", help="Torben state directory")
    parser.add_argument("--legacy", help="Legacy JSON array path")
    parser.add_argument("--journal", help="Target JSONL journal path")
    parser.add_argument("--report", help="Migration reconciliation report path")
    parser.add_argument("--apply", action="store_true", help="Write the migration; default is dry-run")
    parser.add_argument("--force", action="store_true", help="Replace an existing journal")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir) if args.state_dir else get_hermes_home() / "state"
    report = migrate_legacy_ledger(
        legacy_path=Path(args.legacy) if args.legacy else state_dir / LEGACY_LEDGER_NAME,
        journal_path=Path(args.journal) if args.journal else state_dir / JOURNAL_LEDGER_NAME,
        report_path=Path(args.report) if args.report else state_dir / "torben-ledger-migration-report.json",
        apply=bool(args.apply),
        force=bool(args.force),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "\n".join(
                [
                    f"status={report['status']}",
                    f"records={report['legacy_record_count']}->{report['journal_record_count']}",
                    f"approved_demoted={report['approved_demoted']}",
                    f"approved_remaining={report['approved_remaining']}",
                    f"report={report['report_path']}",
                ]
            )
        )
    return 0 if report.get("migration_safety_gate") == "pass" and report.get("status") != "failed" else 1


if __name__ == "__main__":
    if "--json" in sys.argv:
        raise SystemExit(main())
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-ledger-migrate", main))
