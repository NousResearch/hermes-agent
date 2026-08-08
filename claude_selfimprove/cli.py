"""Operator commands for the Claude Code self-improvement pipeline.

    python -m claude_selfimprove.cli install [--dry-run]
    python -m claude_selfimprove.cli status
    python -m claude_selfimprove.cli seed [--force] [--dry-run]
    python -m claude_selfimprove.cli scan [--dry-run] [--model M] [--provider P]
    python -m claude_selfimprove.cli consolidate [--dry-run] [--model M] [--provider P]
    python -m claude_selfimprove.cli rollback --list
    python -m claude_selfimprove.cli rollback --target /path/to/file
    python -m claude_selfimprove.cli rollback --backup-id /path/to/backup.json

``--dry-run`` is a flag on ``scan`` and ``consolidate`` (not a separate
command) — dry-running only means something in the context of one of those
two passes, so this is the only interpretation that reads unambiguously to
an operator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audit, install, notify, paths, pipeline, store, writers


def _fmt_ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "(none)"


def cmd_install(args: argparse.Namespace) -> int:
    result = install.run(dry_run=args.dry_run)
    verb = "would deploy" if args.dry_run else "deployed"
    lines = [f"install: package {verb} to {paths.hermes_home() / 'scripts' / 'claude_selfimprove'}"]

    for name, created, existed in (
        (install.NIGHTLY_JOB_NAME, result.nightly_job_created, result.nightly_job_already_existed),
        (install.WEEKLY_JOB_NAME, result.weekly_job_created, result.weekly_job_already_existed),
    ):
        if existed:
            lines.append(f"  {name}: already registered, left untouched")
        elif created:
            lines.append(f"  {name}: {'would create' if args.dry_run else 'created'}")
        else:
            lines.append(f"  {name}: NOT created (see errors below)")

    if result.errors:
        lines.append("errors:")
        lines.extend(f"  - {e}" for e in result.errors)

    print("\n".join(lines))
    return 1 if result.errors else 0


def cmd_status(args: argparse.Namespace) -> int:
    lines = ["claude-selfimprove status", "=" * 26]

    paths.ensure_state_dirs()
    lines.append(f"state dir: {paths.state_dir()}")
    lines.append(f"claude home: {paths.claude_home()}")
    lines.append(f"claude-hermes home: {paths.claude_hermes_home()}")

    if paths.lock_dir_path().exists():
        lines.append("lock: HELD (a run is in progress, or a stale lock — see rollback docs)")
    else:
        lines.append("lock: free")

    with store.CandidateStore() as db:
        counts = db.counts_by_status()
    lines.append("")
    lines.append("candidates by status:")
    if counts:
        for status, n in sorted(counts.items()):
            lines.append(f"  {status}: {n}")
    else:
        lines.append("  (none yet)")

    backups = writers.list_backups()
    lines.append("")
    lines.append(f"backups on record: {len(backups)}")

    recent = audit.read_all()[-10:]
    lines.append("")
    lines.append(f"last {len(recent)} audit events:")
    for entry in recent:
        lines.append(f"  {entry.get('ts', '?')}  {entry.get('event', '?')}")

    print("\n".join(lines))
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    result = pipeline.seed(force=args.force, dry_run=args.dry_run)
    if result.skipped_busy:
        print("seed skipped: another run holds the lock")
        return 0
    if result.error:
        print(f"seed failed: {result.error}", file=sys.stderr)
        return 1
    if result.already_seeded:
        print("already seeded — pass --force to re-seed (this discards unprocessed evidence in any partial checkpoint)")
        return 0
    mode = " (dry run)" if args.dry_run else ""
    print(f"seeded{mode}: {result.files_seeded} existing transcript files marked as already read")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    result = pipeline.scan(model=args.model, provider=args.provider, dry_run=args.dry_run)
    if result.skipped_busy:
        print("scan skipped: another run holds the lock")
        return 0
    if result.error:
        print(f"scan failed: {result.error}", file=sys.stderr)
        return 1
    mode = " (dry run)" if args.dry_run else ""
    print(
        f"scan complete{mode}: {result.turns_scanned} new turns across "
        f"{result.files_touched} tracked files, {result.candidates_flagged} "
        f"flagged, {result.candidates_classified} classified as real lessons"
    )
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    result = pipeline.consolidate(model=args.model, provider=args.provider, dry_run=args.dry_run)
    if result.skipped_busy:
        print("consolidate skipped: another run holds the lock")
        return 0
    if result.error:
        print(f"consolidate failed: {result.error}", file=sys.stderr)
        return 1
    mode = " (dry run — nothing written)" if args.dry_run else ""
    verb = "would apply" if args.dry_run else "applied"
    print(
        f"consolidate complete{mode}: {result.evaluated} pending candidates evaluated\n"
        f"  {verb}: {_fmt_ids(result.applied)}\n"
        f"  conflict-blocked: {_fmt_ids(result.conflict_blocked)}\n"
        f"  excluded (repo-specific): {_fmt_ids(result.excluded_repo_scope)}\n"
        f"  not yet eligible: {result.not_yet_eligible}\n"
        f"  failed: {_fmt_ids(result.failed)}"
    )
    return 1 if result.failed else 0


def cmd_rollback(args: argparse.Namespace) -> int:
    if args.list:
        backups = writers.list_backups()
        if not backups:
            print("no backups on record")
            return 0
        for meta in backups:
            existed = "update" if meta.get("existed_before") else "creation"
            print(f"{meta['timestamp']}  {meta['target_path']}  ({existed})  [{meta['_backup_path']}]")
        return 0

    if args.backup_id:
        result = writers.rollback_backup(Path(args.backup_id))
    elif args.target:
        meta = writers.latest_backup_for(Path(args.target))
        if meta is None:
            print(f"no backup found for {args.target}", file=sys.stderr)
            return 1
        result = writers.rollback_backup(Path(meta["_backup_path"]))
    else:
        print("rollback requires --list, --target, or --backup-id", file=sys.stderr)
        return 2

    if result.success:
        notify.notify_rollback(
            artifact=str(result.path), label=Path(result.path).name if result.path else "?",
            reason="operator-triggered rollback",
        )
        print(f"rolled back {result.path}")
        return 0
    print(f"rollback failed: {result.reason}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claude-selfimprove")
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="Deploy the package and register the two cron jobs (idempotent)")
    p_install.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Show candidate counts, recent activity, and lock state")

    p_seed = sub.add_parser("seed", help="One-time: mark existing transcripts as already read")
    p_seed.add_argument("--force", action="store_true", help="Re-seed even if already seeded")
    p_seed.add_argument("--dry-run", action="store_true")

    p_scan = sub.add_parser("scan", help="Incrementally scan transcripts and classify candidates")
    p_scan.add_argument("--dry-run", action="store_true")
    p_scan.add_argument("--model", default=None)
    p_scan.add_argument("--provider", default=None)

    p_consolidate = sub.add_parser("consolidate", help="Evaluate thresholds/conflicts and write eligible candidates")
    p_consolidate.add_argument("--dry-run", action="store_true")
    p_consolidate.add_argument("--model", default=None)
    p_consolidate.add_argument("--provider", default=None)

    p_rollback = sub.add_parser("rollback", help="List or restore backups of pipeline-written files")
    p_rollback.add_argument("--list", action="store_true")
    p_rollback.add_argument("--target", default=None, help="Restore the latest backup for this file")
    p_rollback.add_argument("--backup-id", default=None, help="Restore a specific backup metadata file")

    return parser


_HANDLERS = {
    "install": cmd_install,
    "status": cmd_status,
    "seed": cmd_seed,
    "scan": cmd_scan,
    "consolidate": cmd_consolidate,
    "rollback": cmd_rollback,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
