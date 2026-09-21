#!/usr/bin/env python3
"""Register all 127 P13 catalogue entries as disabled cron jobs.

Reads the residual ledger (post-repair state), creates disabled-only cron jobs
via the Hermes cron API. Does NOT enable any job. Safe to run multiple times —
reuses existing disabled registrations by source_instance.

Usage:
  KENSEI_MIGRATION_AUTHORITY=!go python3 scripts/p13_register_disabled_all.py \
    --ledger PATH --receipt PATH [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

GO_AUTHORITY = "!go"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schedule_text(schedule: Any) -> str:
    if not isinstance(schedule, dict):
        return str(schedule or "daily")
    kind = schedule.get("kind", "")
    if kind == "cron" and schedule.get("expr"):
        return str(schedule["expr"])
    if kind == "interval" and schedule.get("minutes") is not None:
        return f"every {schedule['minutes']}m"
    if kind == "once":
        return str(schedule.get("run_at") or schedule.get("timestamp") or "2026-01-01T00:00:00")
    return str(schedule.get("display", "daily"))


def load_catalogue(path: Path, expected_sha256: str = "") -> list[dict[str, Any]]:
    """Load the P13 catalogue and return entries ready for registration."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not entries:
        raise ValueError("catalogue has zero entries")
    if expected_sha256:
        actual = _sha256_hex(path.read_bytes())
        if actual != expected_sha256:
            raise ValueError(f"catalogue SHA256 mismatch: expected {expected_sha256}, got {actual}")
    registrable = []
    for entry in entries:
        tj = entry.get("target_job")
        if not isinstance(tj, dict):
            print(f"SKIP {entry.get('source_instance')}: no target_job", file=sys.stderr)
            continue
        registrable.append(entry)
    return registrable


def build_kwargs(entry: dict[str, Any]) -> dict[str, Any]:
    """Build cron create kwargs from a ledger entry."""
    tj = entry["target_job"]
    si = entry.get("source_instance", "unknown")

    schedule = _schedule_text(tj.get("schedule", "daily"))
    name = tj.get("name") or si
    prompt = tj.get("prompt", "")
    script = tj.get("script")
    no_agent = bool(tj.get("no_agent", False))
    deliver = tj.get("deliver", "local")
    if deliver == "origin":
        deliver = "local"
    skills = tj.get("skills")
    if isinstance(skills, list):
        skills = [s for s in skills if s]
    model_config = None
    if tj.get("model") or tj.get("provider"):
        model_config = {"model": tj.get("model", ""), "provider": tj.get("provider", "")}

    return {
        "name": name,
        "prompt": prompt,
        "schedule": schedule,
        "script": script,
        "no_agent": no_agent,
        "deliver": deliver,
        "skills": skills or None,
        "model": model_config,
        "enabled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True,
                        help="Path to P13 Disabled Target Catalogue JSON")
    parser.add_argument("--receipt", type=Path, required=True,
                        help="Path to write the registration receipt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be done, don't register")
    args = parser.parse_args(argv)

    authority = os.environ.get("KENSEI_MIGRATION_AUTHORITY", "")
    if not args.dry_run and authority != GO_AUTHORITY:
        print("ERROR: KENSEI_MIGRATION_AUTHORITY=!go required for registration", file=sys.stderr)
        return 2

    entries = load_catalogue(args.catalogue)
    print(f"Loaded {len(entries)} registrable entries from {args.catalogue}", file=sys.stderr)

    # Import cron modules
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cron.jobs import create_job, list_jobs

    # Check existing registrations
    existing = list_jobs(include_disabled=True)
    by_source: dict[str, dict[str, Any]] = {}
    for job in existing:
        origin = job.get("origin") or {}
        if origin.get("migration") == "P13_DISABLED_CATALOGUE":
            si = origin.get("source_instance")
            if si:
                by_source[str(si)] = job

    print(f"Found {len(by_source)} existing P13 registrations", file=sys.stderr)

    created = []
    reused = 0
    skipped = 0
    errors = []

    for entry in entries:
        si = entry.get("source_instance", "unknown")
        if si in by_source:
            reused += 1
            target_id = by_source[si]["id"]
            created.append({"source_instance": si, "target_id": target_id, "action": "reused"})
            continue

        kwargs = build_kwargs(entry)
        kwargs["origin"] = {
            "migration": "P13_DISABLED_CATALOGUE",
            "source_instance": si,
        }

        if args.dry_run:
            print(f"  WOULD CREATE: {kwargs['name']} ({kwargs['schedule']})", file=sys.stderr)
            created.append({"source_instance": si, "action": "dry-run"})
            continue

        try:
            result = create_job(**kwargs)
            target_id = result.get("id", "unknown")
            if result.get("enabled"):
                print(f"WARNING: {si} created as enabled! Pausing...", file=sys.stderr)
                from cron.jobs import pause_job
                pause_job(target_id, reason="p13_disabled_registration")
            created.append({"source_instance": si, "target_id": target_id, "action": "created"})
        except Exception as e:
            errors.append({"source_instance": si, "error": str(e)})
            skipped += 1
            print(f"ERROR {si}: {e}", file=sys.stderr)

    # Write receipt
    receipt = {
        "schema_version": 1,
        "receipt_kind": "p13_registration_v1",
        "migration": "P13_DISABLED_CATALOGUE",
        "catalogue_sha256": _sha256_hex(args.catalogue.read_bytes()),
        "summary": {
            "total": len(entries),
            "created": len([c for c in created if c.get("action") == "created"]),
            "reused": reused,
            "skipped": skipped,
        },
        "jobs": created,
        "errors": errors,
    }

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print(f"\nDONE: {receipt['summary']['created']} created, {reused} reused, {skipped} skipped", file=sys.stderr)
    print(json.dumps(receipt["summary"], indent=2))

    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
