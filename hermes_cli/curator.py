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
from pathlib import Path
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


def _print_unmanaged_summary() -> None:
    """Report curation-eligible skills that carry no provenance marker.

    A skill only becomes curator-managed once ``created_by: agent`` lands on
    its usage record, which happens ONLY for background-review creations.
    Skills predating that marker, plus every foreground
    ``skill_manage(create)``, are eligible but unmanaged — no automatic
    transition ever considers them. Printing just the managed count made a
    large library look fully curated while a big slice was untouchable.
    """
    from tools import skill_usage

    try:
        unmanaged = skill_usage.unmanaged_report()
    except Exception:
        return
    if not unmanaged:
        return
    legacy = sum(1 for r in unmanaged if not r.get("has_provenance_key"))
    foreground = len(unmanaged) - legacy
    print(f"\nunmanaged (no provenance marker): {len(unmanaged)} total")
    print(f"  pre-dates marker    {legacy}")
    print(f"  foreground-created  {foreground}")
    print(
        "  never auto-staled or archived — "
        "`hermes curator adopt <name>` hands one over"
    )


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
    # Summary may be multi-line when the curator archived skills (the rename
    # map gets appended as `name → umbrella` lines). Indent continuation
    # lines so the block reads as one logical field.
    if "\n" in summary:
        first, *rest = summary.splitlines()
        print(f"  last summary:   {first}")
        for line in rest:
            print(f"                  {line}")
    else:
        print(f"  last summary:   {summary}")
    _report = state.get("last_report_path")
    if _report:
        suffix = "" if Path(_report).exists() else " (missing)"
        print(f"  last report:    {_report}{suffix}")
    _ih = curator.get_interval_hours()
    _interval_label = (
        f"{_ih // 24}d" if _ih % 24 == 0 and _ih >= 24
        else f"{_ih}h"
    )
    print(f"  interval:       every {_interval_label}")
    print(f"  stale after:    {curator.get_stale_after_days()}d unused")
    print(f"  archive after:  {curator.get_archive_after_days()}d unused")
    print(
        f"  consolidate:    {'on' if curator.get_consolidate() else 'off'}"
        f"{'' if curator.get_consolidate() else ' (prune-only; LLM merge pass opt-in)'}"
    )

    rows = skill_usage.curated_report()
    if not rows:
        print("\nno curator-managed skills")
        _print_unmanaged_summary()
        return 0

    by_state = {"active": [], "stale": [], "archived": []}
    pinned = []
    agent_count = 0
    bundled_count = 0
    for r in rows:
        state_name = r.get("state", "active")
        by_state.setdefault(state_name, []).append(r)
        if r.get("pinned"):
            pinned.append(r["name"])
        prov = r.get("provenance", "agent")
        if prov == "agent":
            agent_count += 1
        elif prov == "bundled":
            bundled_count += 1

    print(f"\ncurator-managed skills: {len(rows)} total  "
          f"(agent-created={agent_count}  bundled={bundled_count})")
    for state_name in ("active", "stale", "archived"):
        bucket = by_state.get(state_name, [])
        print(f"  {state_name:10s} {len(bucket)}")

    if pinned:
        print(f"\npinned ({len(pinned)}): {', '.join(pinned)}")

    # Surface the curation blind spot on the managed path too.
    _print_unmanaged_summary()

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
    background = bool(getattr(args, "background", False))
    synchronous = bool(getattr(args, "synchronous", False)) or not background
    # --consolidate forces the LLM umbrella-building pass on for this run,
    # overriding the config default (off). When the flag is absent, pass None
    # so run_curator_review reads curator.consolidate from config.
    consolidate = True if bool(getattr(args, "consolidate", False)) else None
    if dry:
        print("curator: running DRY-RUN (report only, no mutations)...")
    else:
        print("curator: running review pass...")
    if consolidate is None and not curator.get_consolidate():
        print(
            "curator: consolidation is off — running prune-only "
            "(deterministic stale/archive). Pass --consolidate or set "
            "`curator.consolidate: true` to enable the LLM merge pass."
        )

    def _on_summary(msg: str) -> None:
        print(msg)

    result = curator.run_curator_review(
        on_summary=_on_summary,
        synchronous=synchronous,
        dry_run=dry,
        consolidate=consolidate,
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
    if not synchronous:
        print("llm pass running in background — check `hermes curator status` later")
    if dry:
        if synchronous:
            print(
                "dry-run: no changes applied. Read the report with "
                "`hermes curator status` and run `hermes curator run` (no flag) to apply."
            )
        else:
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
    if not skill_usage.set_pinned(args.skill, True):
        print(
            f"curator: could not pin '{args.skill}' — the skill is not "
            "curation-eligible (protected built-in or external). "
            "`hermes curator list-unmanaged` shows which skills the curator tracks."
        )
        return 1
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
    if not skill_usage.set_pinned(args.skill, False):
        print(
            f"curator: could not unpin '{args.skill}' — the skill is not "
            "curation-eligible (protected built-in or external)."
        )
        return 1
    print(f"curator: unpinned '{args.skill}'")
    return 0


def _cmd_list_unmanaged(args) -> int:
    """List curation-eligible skills that carry no provenance marker.

    The same population `status` summarizes, itemized. Useful before deciding
    what to hand over with `adopt`.
    """
    from tools import skill_usage

    rows = skill_usage.unmanaged_report()
    if not rows:
        print("curator: no unmanaged skills — every eligible skill is managed")
        return 0

    print(f"unmanaged skills ({len(rows)}):")
    for r in sorted(rows, key=lambda x: x["name"]):
        why = "created_by:null" if r.get("has_provenance_key") else "no marker"
        last = _fmt_ts(r.get("last_activity_at"))
        print(
            f"  {r['name']:44s} "
            f"activity={r.get('activity_count', 0):4d}  "
            f"last_activity={last:14s}  "
            f"({why})"
        )
    print("\nadopt one with `hermes curator adopt <name>`, "
          "or all with `hermes curator adopt --all-unmanaged`")
    return 0


def _cmd_adopt(args) -> int:
    """Hand unmanaged skills to the curator by explicit user declaration.

    Provenance cannot be inferred from telemetry: a high patch count proves
    the agent MAINTAINS a skill, not that it AUTHORED it (the agent edits
    user-written skills on the user's behalf constantly). So adoption is never
    automatic — the user names what they're handing over, or passes
    ``--all-unmanaged`` to hand over every eligible skill at once.
    """
    from tools import skill_usage

    names = list(getattr(args, "skill", None) or [])
    adopt_all = bool(getattr(args, "all_unmanaged", False))
    if adopt_all:
        if names:
            print("curator: pass either skill names or --all-unmanaged, not both")
            return 1
        names = skill_usage.list_unmanaged_skill_names()
        if not names:
            print("curator: no unmanaged skills to adopt")
            return 0
    if not names:
        print("curator: name a skill to adopt, or pass --all-unmanaged")
        return 1

    dry_run = bool(getattr(args, "dry_run", False))
    if dry_run:
        print(f"curator: would adopt {len(names)} skill(s) (dry run):")
        for n in names:
            print(f"  + {n}")
        return 0

    # Bulk adoption is a real lifecycle change (adopted skills become
    # archivable), so confirm unless the caller opted out.
    if adopt_all and not bool(getattr(args, "yes", False)):
        print(f"curator: adopt {len(names)} unmanaged skill(s) into curator management?")
        print("  they become eligible for automatic staleness + archival")
        try:
            reply = input("  proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        if reply not in {"y", "yes"}:
            print("curator: aborted")
            return 1

    failed = 0
    for n in names:
        ok, msg = skill_usage.adopt_skill(n)
        print(f"curator: {msg}")
        if not ok:
            failed += 1
    if len(names) > 1:
        print(f"curator: adopted {len(names) - failed}/{len(names)}")
    return 1 if failed else 0


def _cmd_restore(args) -> int:
    from tools import skill_ledger, skill_usage
    tok = skill_ledger.set_ledger_actor("user")
    try:
        ok, msg = skill_usage.restore_skill(args.skill)
    finally:
        skill_ledger.reset_ledger_actor(tok)
    print(f"curator: {msg}")
    return 0 if ok else 1


def _cmd_archive(args) -> int:
    """Manually archive an agent-created skill. Refuses if pinned.

    The auto-curator archives stale skills on its own schedule; this verb is
    for the user who wants to archive *now* without waiting for a run.
    """
    from tools import skill_ledger, skill_usage
    if skill_usage.get_record(args.skill).get("pinned"):
        print(
            f"curator: '{args.skill}' is pinned — unpin first with "
            f"`hermes curator unpin {args.skill}`"
        )
        return 1
    tok = skill_ledger.set_ledger_actor("user")
    try:
        ok, msg = skill_usage.archive_skill(args.skill)
    finally:
        skill_ledger.reset_ledger_actor(tok)
    print(f"curator: {msg}")
    return 0 if ok else 1


def _idle_days(record: dict) -> Optional[int]:
    """Days since the skill's last activity (view / use / patch).

    Falls back to ``created_at`` so a skill that was authored but never used
    can still be pruned — otherwise never-touched skills would be immortal.
    Returns None only when both fields are missing or unparseable.
    """
    ts = record.get("last_activity_at") or record.get("created_at")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _cmd_prune(args) -> int:
    """Bulk-archive curator-managed skills idle for >= N days.

    Pinned skills are exempt. Already-archived skills are skipped. Default
    ``--days 90`` matches a conservative read of the curator's own archive
    threshold; adjust with ``--days``. Use ``--dry-run`` to preview.
    """
    from tools import skill_usage
    days = getattr(args, "days", 90)
    if days < 1:
        print(f"curator: --days must be >= 1 (got {days})", file=sys.stderr)
        return 2

    dry_run = bool(getattr(args, "dry_run", False))
    skip_confirm = bool(getattr(args, "yes", False))

    candidates = []
    for r in skill_usage.curated_report():
        if r.get("pinned"):
            continue
        if r.get("state") == skill_usage.STATE_ARCHIVED:
            continue
        idle = _idle_days(r)
        if idle is None or idle < days:
            continue
        candidates.append((r["name"], idle))

    if not candidates:
        print(f"curator: nothing to prune (no unpinned skills idle >= {days}d)")
        return 0

    candidates.sort(key=lambda c: -c[1])
    print(f"curator: {len(candidates)} skill(s) idle >= {days}d:")
    for name, idle in candidates:
        print(f"  {name:40s} idle {idle}d")

    if dry_run:
        print("\n(dry run — no changes made)")
        return 0

    if not skip_confirm:
        try:
            reply = input(f"\nArchive {len(candidates)} skill(s)? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ncurator: aborted")
            return 1
        if reply not in {"y", "yes"}:
            print("curator: aborted")
            return 1

    archived = 0
    failures = []
    for name, _ in candidates:
        ok, msg = skill_usage.archive_skill(name)
        if ok:
            archived += 1
        else:
            failures.append((name, msg))

    print(f"\ncurator: archived {archived}/{len(candidates)}")
    if failures:
        print("failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


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


def _cmd_ledger(args) -> int:
    """List per-mutation audit ledger entries (newest first)."""
    from tools import skill_ledger

    rows = skill_ledger.list_entries(
        skill=getattr(args, "skill", None),
        limit=getattr(args, "limit", None) or 20,
    )
    if not rows:
        print("curator: ledger is empty (or skills.ledger is disabled).")
        return 0
    print(f"{'id':<14} {'when':<12} {'actor':<8} {'action':<12} skill")
    for r in rows:
        evidence = r.get("evidence") or {}
        extra = ""
        if evidence.get("absorbed_into"):
            extra = f"  → absorbed into '{evidence['absorbed_into']}'"
        elif evidence.get("rollback_target"):
            extra = f"  → rollback of {evidence['rollback_target']}"
        print(
            f"{r.get('id', '?'):<14} {_fmt_ts(r.get('ts')):<12} "
            f"{r.get('actor', '?'):<8} {r.get('action', '?'):<12} "
            f"{r.get('skill', '?')}{extra}"
        )
    print(
        "\nRoll back a single mutation with `hermes curator rollback <id>`; "
        "whole-tree snapshots remain available via `hermes curator rollback --list`."
    )
    return 0


def _cmd_purge(args) -> int:
    """Delete archived skills older than curator.archive_ttl_days.

    Explicit command only — never runs automatically. Respects the ledger:
    each purged skill is captured (before-blobs) and recorded as a 'purge'
    entry, so even a purge is auditable and blob-recoverable.
    """
    from hermes_cli.config import cfg_get, load_config
    from tools import skill_ledger
    from tools.skill_usage import _archive_dir

    ttl_days = getattr(args, "days", None)
    if ttl_days is None:
        ttl_days = int(cfg_get(load_config(), "curator", "archive_ttl_days", default=0) or 0)
    if ttl_days <= 0:
        print(
            "curator: purge disabled (curator.archive_ttl_days is 0). Set the "
            "config key or pass --days N to purge archives older than N days."
        )
        return 1

    archive_root = _archive_dir()
    if not archive_root.exists():
        print("curator: no archive directory — nothing to purge.")
        return 0

    import shutil
    import time

    cutoff = time.time() - ttl_days * 86400
    candidates = [
        p for p in archive_root.iterdir()
        if p.is_dir() and p.stat().st_mtime < cutoff
    ]
    if not candidates:
        print(f"curator: no archived skills older than {ttl_days}d.")
        return 0

    print(f"Archived skills older than {ttl_days}d:")
    for p in sorted(candidates):
        print(f"  {p.name}")
    if getattr(args, "dry_run", False):
        print("(dry run — nothing deleted)")
        return 0
    if not getattr(args, "yes", False):
        try:
            ans = input(f"Permanently delete {len(candidates)} archived skill(s)? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled")
            return 1
        if ans not in {"y", "yes"}:
            print("cancelled")
            return 1

    purged = 0
    for p in sorted(candidates):
        before = skill_ledger.capture_before(p)
        try:
            shutil.rmtree(p)
        except OSError as e:
            print(f"curator: failed to purge {p.name}: {e}")
            continue
        skill_ledger.append_entry(
            "purge",
            p.name,
            before=before or [],
            after=[],
            actor="user",
            evidence={"ttl_days": ttl_days},
        )
        purged += 1
    print(f"curator: purged {purged} archived skill(s). Ledger entries recorded.")
    return 0


def _cmd_rollback(args) -> int:
    """Restore the skills tree from a snapshot, or a single mutation from
    the audit ledger.

    With a positional ``entry_id``, restores exactly the files touched by
    that one ledger entry (from content-addressed blobs), taking a
    pre-rollback safety ledger entry first — and failing closed when that
    safety capture fails. Without it, behaves as before: whole-tree tarball
    restore. ``--list`` prints available snapshots and exits. ``--id
    <stamp>`` picks a specific snapshot. Without ``-y``, prompts for
    confirmation. A safety snapshot of the current tree is always taken
    first, so rollbacks are themselves undoable.
    """
    from agent import curator_backup

    entry_id = getattr(args, "entry_id", None)
    if entry_id:
        from tools import skill_ledger

        entry = skill_ledger.get_entry(entry_id)
        if entry is None:
            print(
                f"curator: no ledger entry '{entry_id}'. "
                "See `hermes curator ledger` for entry ids, or use "
                "`--id <snapshot>` for whole-tree snapshot rollback."
            )
            return 1
        print(f"Rollback target: ledger entry {entry_id}")
        print(f"  action: {entry.get('action', '?')}")
        print(f"  skill:  {entry.get('skill', '?')}")
        print(f"  actor:  {entry.get('actor', '?')}")
        print(f"  when:   {entry.get('ts', '?')}")
        touched = {i.get("path") for i in (entry.get("before") or []) + (entry.get("after") or [])}
        print(f"  files:  {len(touched)}")
        if not getattr(args, "yes", False):
            try:
                ans = input("Restore this mutation's before-state? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\ncancelled")
                return 1
            if ans not in {"y", "yes"}:
                print("cancelled")
                return 1
        ok, msg = skill_ledger.rollback_entry(entry_id)
        if ok:
            print(f"curator: {msg}")
            return 0
        print(f"curator: rollback failed — {msg}")
        return 1

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
        cron = manifest.get("cron_jobs") or {}
        if isinstance(cron, dict):
            if cron.get("backed_up"):
                print(
                    f"  cron jobs:   {cron.get('jobs_count', 0)} "
                    f"(will be restored for skill-link fields only)"
                )
            else:
                reason = cron.get("reason", "not captured")
                print(f"  cron jobs:   not in snapshot ({reason})")
    print(
        "\nThis will replace the current ~/.hermes/skills/ tree (a safety "
        "snapshot of the current state is taken first so this is undoable). "
        "Cron jobs that still exist will have their skills/skill fields "
        "restored from the snapshot; all other cron fields are left alone."
    )

    if not getattr(args, "yes", False):
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled")
            return 1
        if ans not in {"y", "yes"}:
            print("cancelled")
            return 1

    ok, msg, _ = curator_backup.rollback(backup_id=target_path.name)
    if ok:
        print(f"curator: {msg}")
        return 0
    print(f"curator: rollback failed — {msg}")
    return 1


def _cmd_list_archived(args) -> int:
    """List archived (recoverable) skills."""
    from tools import skill_usage
    names = skill_usage.list_archived_skill_names()
    if not names:
        print("curator: no archived skills")
        return 0
    for name in names:
        print(name)
    return 0


def _cmd_usage(args) -> int:
    """Show usage telemetry for ALL skills, with provenance.

    Unlike `status` (curator-scoped to curated candidates), this lists
    every skill on disk — bundled built-ins and hub-installed included — so you
    can see how often each is actually used regardless of curation.
    """
    import json as _json
    from tools import skill_usage

    rows = skill_usage.usage_report()

    prov_filter = getattr(args, "provenance", None)
    if prov_filter:
        rows = [r for r in rows if r.get("provenance") == prov_filter]

    sort_key = getattr(args, "sort", "activity")
    if sort_key == "name":
        rows.sort(key=lambda r: r["name"])
    elif sort_key == "recent":
        # Most-recently-active first; never-active sinks to the bottom.
        rows.sort(key=lambda r: r.get("last_activity_at") or "", reverse=True)
    else:  # "activity" (default): most-used first
        rows.sort(key=lambda r: r.get("activity_count", 0), reverse=True)

    if getattr(args, "json", False):
        print(_json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("curator: no skills found")
        return 0

    # Provenance tallies for a quick header.
    counts = {"agent": 0, "bundled": 0, "hub": 0}
    for r in rows:
        counts[r.get("provenance", "agent")] = counts.get(r.get("provenance", "agent"), 0) + 1
    print(
        f"skills: {len(rows)} total  "
        f"(agent={counts['agent']}  bundled={counts['bundled']}  hub={counts['hub']})"
    )
    print()
    print(
        f"  {'skill':40s}  {'origin':8s}  "
        f"{'use':>4s}  {'view':>4s}  {'patch':>5s}  {'act':>4s}  last_activity"
    )
    for r in rows:
        last = _fmt_ts(r.get("last_activity_at"))
        print(
            f"  {r['name'][:40]:40s}  "
            f"{r.get('provenance', 'agent'):8s}  "
            f"{r.get('use_count', 0):>4d}  "
            f"{r.get('view_count', 0):>4d}  "
            f"{r.get('patch_count', 0):>5d}  "
            f"{r.get('activity_count', 0):>4d}  "
            f"{last}"
        )
    return 0


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

    p_usage = subs.add_parser(
        "usage",
        help="Show usage telemetry for ALL skills (built-in, hub, agent) with provenance",
    )
    p_usage.add_argument(
        "--sort", choices=("activity", "recent", "name"), default="activity",
        help="Sort order: activity (most-used first, default), recent "
             "(most-recently-active first), or name (alphabetical)",
    )
    p_usage.add_argument(
        "--provenance", choices=("agent", "bundled", "hub"), default=None,
        help="Only show skills of this origin",
    )
    p_usage.add_argument(
        "--json", action="store_true",
        help="Emit the full report as JSON instead of a table",
    )
    p_usage.set_defaults(func=_cmd_usage)

    p_run = subs.add_parser("run", help="Trigger a curator review now")
    p_run.add_argument(
        "--sync", "--synchronous", dest="synchronous", action="store_true",
        help="Wait for the LLM review pass to finish (default for manual runs)",
    )
    p_run.add_argument(
        "--background", dest="background", action="store_true",
        help="Start the LLM review pass in a background thread and return immediately",
    )
    p_run.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Report only — no state changes, no archives, no consolidation "
             "(use this to preview what curator would do)",
    )
    p_run.add_argument(
        "--consolidate", dest="consolidate", action="store_true",
        help="Force the LLM umbrella-building consolidation pass on for this "
             "run, overriding the config default (off). Without this flag the "
             "run is prune-only unless `curator.consolidate: true` is set.",
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

    subs.add_parser(
        "list-unmanaged",
        help="List curation-eligible skills with no provenance marker",
    ).set_defaults(func=_cmd_list_unmanaged)

    p_adopt = subs.add_parser(
        "adopt",
        help="Hand unmanaged skills to the curator (provenance is a user declaration)",
    )
    p_adopt.add_argument(
        "skill", nargs="*",
        help="Skill name(s) to adopt. Omit when using --all-unmanaged.",
    )
    p_adopt.add_argument(
        "--all-unmanaged", action="store_true",
        help="Adopt every curation-eligible skill that has no provenance marker",
    )
    p_adopt.add_argument(
        "--dry-run", action="store_true",
        help="List what would be adopted without writing anything",
    )
    p_adopt.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt for --all-unmanaged",
    )
    p_adopt.set_defaults(func=_cmd_adopt)

    p_restore = subs.add_parser("restore", help="Restore an archived skill")
    p_restore.add_argument("skill", help="Skill name")
    p_restore.set_defaults(func=_cmd_restore)

    subs.add_parser("list-archived", help="List archived skills") \
        .set_defaults(func=_cmd_list_archived)

    p_archive = subs.add_parser(
        "archive",
        help="Manually archive a skill (move to .archive/, excluded from prompt)",
    )
    p_archive.add_argument("skill", help="Skill name")
    p_archive.set_defaults(func=_cmd_archive)

    p_prune = subs.add_parser(
        "prune",
        help="Bulk-archive curator-managed skills idle for >= N days (default 90)",
    )
    p_prune.add_argument(
        "--days", type=int, default=90,
        help="Archive skills idle for at least N days (default: 90)",
    )
    p_prune.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt",
    )
    p_prune.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Show what would be archived without doing it",
    )
    p_prune.set_defaults(func=_cmd_prune)

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
        help="Restore ~/.hermes/skills/ from a curator snapshot, or a single "
             "mutation by ledger entry id (see `hermes curator ledger`)",
    )
    p_rollback.add_argument(
        "entry_id", nargs="?", default=None,
        help="Ledger entry id for single-mutation rollback (from "
             "`hermes curator ledger`). Omit for whole-tree snapshot rollback.",
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

    p_ledger = subs.add_parser(
        "ledger",
        help="List the per-mutation skill audit ledger (all actors: "
             "curator/agent/user)",
    )
    p_ledger.add_argument(
        "--skill", default=None,
        help="Only show entries for this skill",
    )
    p_ledger.add_argument(
        "--limit", type=int, default=20,
        help="Max entries to show (default: 20)",
    )
    p_ledger.set_defaults(func=_cmd_ledger)

    p_purge = subs.add_parser(
        "purge",
        help="Delete archived skills older than curator.archive_ttl_days "
             "(explicit only — never automatic; recorded in the ledger)",
    )
    p_purge.add_argument(
        "--days", type=int, default=None,
        help="Override curator.archive_ttl_days for this invocation",
    )
    p_purge.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Show what would be purged without deleting",
    )
    p_purge.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt",
    )
    p_purge.set_defaults(func=_cmd_purge)


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
