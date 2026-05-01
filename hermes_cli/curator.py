"""CLI subcommand: `hermes curator <subcommand>`.

Thin shell around agent/curator.py and tools/skill_usage.py. Renders a status
table, triggers a run, pauses/resumes, and pins/unpins skills.

This module intentionally has no side effects at import time — main.py wires
the argparse subparsers on demand.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return "never"
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _cmd_status(args) -> int:
    from agent import curator
    from tools import skill_usage

    state = curator.load_state()
    enabled = curator.is_enabled()
    paused = state.get("paused", False)
    last_run = state.get("last_run_at")
    summary = state.get("last_run_summary") or "(none)"
    runs = state.get("run_count", 0)

    status_line = (
        "ENABLED" if enabled and not paused else
        "PAUSED" if paused else
        "DISABLED"
    )
    print(f"curator: {status_line}")
    print(f"  runs:           {runs}")
    print(f"  last run:       {_fmt_ts(last_run)}")
    print(f"  last summary:   {summary}")
    _report = state.get("last_report_path")
    if _report:
        print(f"  last report:    {_report}")
    _ih = curator.get_interval_hours()
    _interval_label = (
        f"{_ih // 24}d" if _ih % 24 == 0 and _ih >= 24
        else f"{_ih}h"
    )
    print(f"  interval:       every {_interval_label}")
    print(f"  stale after:    {curator.get_stale_after_days()}d unused")
    print(f"  archive after:  {curator.get_archive_after_days()}d unused")

    rows = skill_usage.agent_created_report()
    if not rows:
        print("\nno agent-created skills")
        return 0

    by_state = {"active": [], "stale": [], "archived": []}
    pinned = []
    for r in rows:
        state_name = r.get("state", "active")
        by_state.setdefault(state_name, []).append(r)
        if r.get("pinned"):
            pinned.append(r["name"])

    print(f"\nagent-created skills: {len(rows)} total")
    for state_name in ("active", "stale", "archived"):
        bucket = by_state.get(state_name, [])
        print(f"  {state_name:10s} {len(bucket)}")

    if pinned:
        print(f"\npinned ({len(pinned)}): {', '.join(pinned)}")

    # Show top 5 least-recently-active skills. Views and edits are activity too:
    # curator should not report a skill as "never used" right after skill_view()
    # or skill_manage() touched it.
    active = sorted(
        by_state.get("active", []),
        key=lambda r: r.get("last_activity_at") or r.get("created_at") or "",
    )[:5]
    if active:
        print("\nleast recently active (top 5):")
        for r in active:
            last = _fmt_ts(r.get("last_activity_at"))
            print(
                f"  {r['name']:40s}  "
                f"activity={r.get('activity_count', 0):3d}  "
                f"use={r.get('use_count', 0):3d}  "
                f"view={r.get('view_count', 0):3d}  "
                f"patches={r.get('patch_count', 0):3d}  "
                f"last_activity={last}"
            )

    # Show top 5 most-active and least-active skills by activity_count
    # (use + view + patch). This is a different signal from
    # least-recently-active: activity_count reflects frequency,
    # last_activity_at reflects recency. A skill touched 30 times a year
    # ago is high-frequency but stale; a skill touched once yesterday is
    # recent but low-frequency. Both can matter.
    active_all = by_state.get("active", [])
    if active_all:
        most_active = sorted(
            active_all,
            key=lambda r: (r.get("activity_count") or 0, r.get("last_activity_at") or ""),
            reverse=True,
        )[:5]
        if most_active and (most_active[0].get("activity_count") or 0) > 0:
            print("\nmost active (top 5):")
            for r in most_active:
                last = _fmt_ts(r.get("last_activity_at"))
                print(
                    f"  {r['name']:40s}  "
                    f"activity={r.get('activity_count', 0):3d}  "
                    f"use={r.get('use_count', 0):3d}  "
                    f"view={r.get('view_count', 0):3d}  "
                    f"patches={r.get('patch_count', 0):3d}  "
                    f"last_activity={last}"
                )

        least_active = sorted(
            active_all,
            key=lambda r: (r.get("activity_count") or 0, r.get("last_activity_at") or ""),
        )[:5]
        if least_active:
            print("\nleast active (top 5):")
            for r in least_active:
                last = _fmt_ts(r.get("last_activity_at"))
                print(
                    f"  {r['name']:40s}  "
                    f"activity={r.get('activity_count', 0):3d}  "
                    f"use={r.get('use_count', 0):3d}  "
                    f"view={r.get('view_count', 0):3d}  "
                    f"patches={r.get('patch_count', 0):3d}  "
                    f"last_activity={last}"
                )

    return 0


def _cmd_run(args) -> int:
    from agent import curator
    if not curator.is_enabled():
        print("curator: disabled via config; enable with `curator.enabled: true`")
        return 1

    dry = bool(getattr(args, "dry_run", False))
    if dry:
        print("curator: running DRY-RUN (report only, no mutations)...")
    else:
        print("curator: running review pass...")

    def _on_summary(msg: str) -> None:
        print(msg)

    result = curator.run_curator_review(
        on_summary=_on_summary,
        synchronous=bool(args.synchronous),
        dry_run=dry,
    )
    auto = result.get("auto_transitions", {})
    if auto:
        if dry:
            print(
                f"auto (preview): {auto.get('checked', 0)} candidate skill(s) "
                "— no transitions applied in dry-run"
            )
        else:
            print(
                f"auto: checked={auto.get('checked', 0)} "
                f"stale={auto.get('marked_stale', 0)} "
                f"archived={auto.get('archived', 0)} "
                f"reactivated={auto.get('reactivated', 0)}"
            )
    if not args.synchronous:
        print("llm pass running in background — check `hermes curator status` later")
    if dry:
        print(
            "dry-run: no changes applied. When the report lands, read it with "
            "`hermes curator status` and run `hermes curator run` (no flag) to apply."
        )
    return 0


def _cmd_pause(args) -> int:
    from agent import curator
    curator.set_paused(True)
    print("curator: paused")
    return 0


def _cmd_resume(args) -> int:
    from agent import curator
    curator.set_paused(False)
    print("curator: resumed")
    return 0


def _cmd_pin(args) -> int:
    from tools import skill_usage
    if not skill_usage.is_agent_created(args.skill):
        print(
            f"curator: '{args.skill}' is bundled or hub-installed — cannot pin "
            "(only agent-created skills participate in curation)"
        )
        return 1
    skill_usage.set_pinned(args.skill, True)
    print(f"curator: pinned '{args.skill}' (will bypass auto-transitions)")
    return 0


def _cmd_unpin(args) -> int:
    from tools import skill_usage
    if not skill_usage.is_agent_created(args.skill):
        print(
            f"curator: '{args.skill}' is bundled or hub-installed — "
            "there's nothing to unpin (curator only tracks agent-created skills)"
        )
        return 1
    skill_usage.set_pinned(args.skill, False)
    print(f"curator: unpinned '{args.skill}'")
    return 0


def _cmd_restore(args) -> int:
    from tools import skill_usage
    ok, msg = skill_usage.restore_skill(args.skill)
    print(f"curator: {msg}")
    return 0 if ok else 1


def _cmd_backup(args) -> int:
    """Take a manual snapshot of the skills tree. Same mechanism as the
    automatic pre-run snapshot, just user-initiated."""
    from agent import curator_backup
    if not curator_backup.is_enabled():
        print(
            "curator: backups are disabled via config "
            "(`curator.backup.enabled: false`); re-enable to snapshot"
        )
        return 1
    reason = getattr(args, "reason", None) or "manual"
    snap = curator_backup.snapshot_skills(reason=reason)
    if snap is None:
        print("curator: snapshot failed — check logs (backup disabled or IO error)")
        return 1
    print(f"curator: snapshot created at ~/.hermes/skills/.curator_backups/{snap.name}")
    return 0


def _cmd_rollback(args) -> int:
    """Restore the skills tree from a snapshot. Defaults to newest.

    ``--list`` prints available snapshots and exits. ``--id <stamp>`` picks
    a specific one. Without ``-y``, prompts for confirmation. A safety
    snapshot of the current tree is always taken first, so rollbacks are
    themselves undoable.
    """
    from agent import curator_backup

    if getattr(args, "list", False):
        print(curator_backup.summarize_backups())
        return 0

    backup_id = getattr(args, "backup_id", None)
    target_path = curator_backup._resolve_backup(backup_id)
    if target_path is None:
        rows = curator_backup.list_backups()
        if not rows:
            print(
                "curator: no snapshots exist yet. Take one with "
                "`hermes curator backup` or wait for the next curator run."
            )
        else:
            print(
                f"curator: no snapshot matching "
                f"{'id ' + repr(backup_id) if backup_id else 'your query'}."
            )
            print("Available:")
            print(curator_backup.summarize_backups())
        return 1

    manifest = curator_backup._read_manifest(target_path)
    print(f"Rollback target: {target_path.name}")
    if manifest:
        print(f"  reason:      {manifest.get('reason', '?')}")
        print(f"  created_at:  {manifest.get('created_at', '?')}")
        print(f"  skill files: {manifest.get('skill_files', '?')}")
    print(
        "\nThis will replace the current ~/.hermes/skills/ tree (a safety "
        "snapshot of the current state is taken first so this is undoable)."
    )

    if not getattr(args, "yes", False):
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled")
            return 1
        if ans not in ("y", "yes"):
            print("cancelled")
            return 1

    ok, msg, _ = curator_backup.rollback(backup_id=target_path.name)
    if ok:
        print(f"curator: {msg}")
        return 0
    print(f"curator: rollback failed — {msg}")
    return 1


# ---------------------------------------------------------------------------
# argparse wiring (called from hermes_cli.main)
# ---------------------------------------------------------------------------

def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `curator` subcommands to *parent*.

    main.py calls this with the ArgumentParser returned by
    ``subparsers.add_parser("curator", ...)``.
    """
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="curator_command")

    p_status = subs.add_parser("status", help="Show curator status and skill stats")
    p_status.set_defaults(func=_cmd_status)

    p_run = subs.add_parser("run", help="Trigger a curator review now")
    p_run.add_argument(
        "--sync", "--synchronous", dest="synchronous", action="store_true",
        help="Wait for the LLM review pass to finish (default: background thread)",
    )
    p_run.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Report only — no state changes, no archives, no consolidation "
             "(use this to preview what curator would do)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_pause = subs.add_parser("pause", help="Pause the curator until resumed")
    p_pause.set_defaults(func=_cmd_pause)

    p_resume = subs.add_parser("resume", help="Resume a paused curator")
    p_resume.set_defaults(func=_cmd_resume)

    p_pin = subs.add_parser("pin", help="Pin a skill so the curator never auto-transitions it")
    p_pin.add_argument("skill", help="Skill name")
    p_pin.set_defaults(func=_cmd_pin)

    p_unpin = subs.add_parser("unpin", help="Unpin a skill")
    p_unpin.add_argument("skill", help="Skill name")
    p_unpin.set_defaults(func=_cmd_unpin)

    p_restore = subs.add_parser("restore", help="Restore an archived skill")
    p_restore.add_argument("skill", help="Skill name")
    p_restore.set_defaults(func=_cmd_restore)

    p_backup = subs.add_parser(
        "backup",
        help="Take a manual tar.gz snapshot of ~/.hermes/skills/ "
             "(curator also does this automatically before every real run)",
    )
    p_backup.add_argument(
        "--reason", default=None,
        help="Free-text label stored in manifest.json (default: 'manual')",
    )
    p_backup.set_defaults(func=_cmd_backup)

    p_rollback = subs.add_parser(
        "rollback",
        help="Restore ~/.hermes/skills/ from a curator snapshot "
             "(defaults to the newest)",
    )
    p_rollback.add_argument(
        "--list", action="store_true",
        help="List available snapshots and exit without restoring",
    )
    p_rollback.add_argument(
        "--id", dest="backup_id", default=None,
        help="Snapshot id to restore (see `--list`); default: newest",
    )
    p_rollback.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_rollback.set_defaults(func=_cmd_rollback)


def cli_main(argv=None) -> int:
    """Standalone entry (also usable by hermes_cli.main fallthrough)."""
    parser = argparse.ArgumentParser(prog="hermes curator")
    register_cli(parser)
    args = parser.parse_args(argv)
    fn = getattr(args, "func", None)
    if fn is None:
        parser.print_help()
        return 0
    return int(fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
