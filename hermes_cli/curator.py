"""CLI subcommand: `hermes curator <subcommand>`."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _parse_ts(ts) -> Optional[datetime]:
    """ISO timestamp -> aware UTC datetime, or None when unparseable."""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return "never"
    dt = _parse_ts(ts)
    if dt is None:
        return str(ts)
    secs = int((datetime.now(timezone.utc) - dt).total_seconds())
    for unit, div, limit in (("s", 1, 60), ("m", 60, 3600), ("h", 3600, 86400)):
        if secs < limit:
            return f"{secs // div}{unit} ago"
    return f"{secs // 86400}d ago"


def _confirm(prompt: str, cancel: str = "cancelled", eof_prefix: str = "\n") -> bool:
    """Ask ``prompt``; print ``cancel`` (prefixed on EOF/Ctrl-C) and return False unless y/yes."""
    try:
        if input(prompt).strip().lower() in {"y", "yes"}:
            return True
    except (EOFError, KeyboardInterrupt):
        print(eof_prefix, end="")
    print(cancel)
    return False


def _print_skill_rows(title: str, rows: list) -> None:
    print(f"\n{title}:")
    for r in rows:
        print(
            f"  {r['name']:40s}  "
            f"activity={r.get('activity_count', 0):3d}  "
            f"use={r.get('use_count', 0):3d}  "
            f"view={r.get('view_count', 0):3d}  "
            f"patches={r.get('patch_count', 0):3d}  "
            f"last_activity={_fmt_ts(r.get('last_activity_at'))}")


def _print_unmanaged_summary() -> None:
    """Report curation-eligible skills that carry no provenance marker: only background-review
    creations get ``created_by: agent``; older skills and every foreground ``skill_manage(create)``
    are eligible but unmanaged, so no automatic transition touches them."""
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
    print("  never auto-staled or archived — `hermes curator adopt <name>` hands one over")


def _print_curator_config(curator) -> None:
    state = curator.load_state()
    status_line = (
        "PAUSED" if state.get("paused", False)
        else "ENABLED" if curator.is_enabled() else "DISABLED")
    print(f"curator: {status_line}")
    print(f"  runs:           {state.get('run_count', 0)}")
    print(f"  last run:       {_fmt_ts(state.get('last_run_at'))}")
    # Multi-line when the curator archived skills (rename map appended as `name → umbrella`
    # lines); indent continuation lines so the block reads as one field.
    summary = state.get("last_run_summary") or "(none)"
    first, *rest = summary.splitlines() if "\n" in summary else [summary]
    print(f"  last summary:   {first}")
    for line in rest:
        print(f"                  {line}")
    report = state.get("last_report_path")
    if report:
        print(f"  last report:    {report}{'' if Path(report).exists() else ' (missing)'}")
    ih = curator.get_interval_hours()
    print(f"  interval:       every {f'{ih // 24}d' if ih % 24 == 0 and ih >= 24 else f'{ih}h'}")
    print(f"  stale after:    {curator.get_stale_after_days()}d unused")
    print(f"  archive after:  {curator.get_archive_after_days()}d unused")
    consolidate = curator.get_consolidate()
    print(
        f"  consolidate:    {'on' if consolidate else 'off'}"
        f"{'' if consolidate else ' (prune-only; LLM merge pass opt-in)'}")


def _cmd_status(args) -> int:
    from agent import curator
    from tools import skill_usage
    _print_curator_config(curator)
    rows = skill_usage.curated_report()
    if not rows:
        print("\nno curator-managed skills")
        _print_unmanaged_summary()
        return 0
    by_state: dict = {}
    for r in rows:
        by_state.setdefault(r.get("state", "active"), []).append(r)
    pinned = [r["name"] for r in rows if r.get("pinned")]
    provenance = [r.get("provenance", "agent") for r in rows]
    print(f"\ncurator-managed skills: {len(rows)} total  "
          f"(agent-created={provenance.count('agent')}  bundled={provenance.count('bundled')})")
    for state_name in ("active", "stale", "archived"):
        print(f"  {state_name:10s} {len(by_state.get(state_name, []))}")
    if pinned:
        print(f"\npinned ({len(pinned)}): {', '.join(pinned)}")
    _print_unmanaged_summary()  # the curation blind spot matters on the managed path too
    # Views and edits count as activity: a skill must not read as "never used" right after
    # skill_view()/skill_manage() touched it. Recency (last_activity_at) and frequency
    # (activity_count) are different signals, so both top-5s are shown.
    active_all = by_state.get("active", [])
    if not active_all:
        return 0
    recency = sorted(
        active_all, key=lambda r: r.get("last_activity_at") or r.get("created_at") or "")
    _print_skill_rows("least recently active (top 5)", recency[:5])

    def _freq(r):
        return (r.get("activity_count") or 0, r.get("last_activity_at") or "")

    most_active = sorted(active_all, key=_freq, reverse=True)[:5]
    if (most_active[0].get("activity_count") or 0) > 0:
        _print_skill_rows("most active (top 5)", most_active)
    _print_skill_rows("least active (top 5)", sorted(active_all, key=_freq)[:5])
    return 0


def _cmd_run(args) -> int:
    from agent import curator
    if not curator.is_enabled():
        print("curator: disabled via config; enable with `curator.enabled: true`")
        return 1
    dry = bool(getattr(args, "dry_run", False))
    background = bool(getattr(args, "background", False))
    synchronous = bool(getattr(args, "synchronous", False)) or not background
    # --consolidate forces the LLM umbrella pass on; absent => None so run_curator_review reads
    # curator.consolidate from config.
    consolidate = True if getattr(args, "consolidate", False) else None
    print(
        "curator: running DRY-RUN (report only, no mutations)..." if dry
        else "curator: running review pass...")
    if consolidate is None and not curator.get_consolidate():
        print(
            "curator: consolidation is off — running prune-only "
            "(deterministic stale/archive). Pass --consolidate or set "
            "`curator.consolidate: true` to enable the LLM merge pass.")
    result = curator.run_curator_review(
        on_summary=print, synchronous=synchronous, dry_run=dry, consolidate=consolidate)
    auto = result.get("auto_transitions", {})
    if auto and dry:
        print(
            f"auto (preview): {auto.get('checked', 0)} candidate skill(s) "
            "— no transitions applied in dry-run")
    elif auto:
        print(
            f"auto: checked={auto.get('checked', 0)} "
            f"stale={auto.get('marked_stale', 0)} "
            f"archived={auto.get('archived', 0)} "
            f"reactivated={auto.get('reactivated', 0)}")
    if not synchronous:
        print("llm pass running in background — check `hermes curator status` later")
    if dry:
        print(
            "dry-run: no changes applied. Read the report with "
            "`hermes curator status` and run `hermes curator run` (no flag) to apply."
            if synchronous else
            "dry-run: no changes applied. When the report lands, read it with "
            "`hermes curator status` and run `hermes curator run` (no flag) to apply.")
    return 0


def _set_paused(paused: bool) -> int:
    from agent import curator
    curator.set_paused(paused)
    print("curator: paused" if paused else "curator: resumed")
    return 0


def _cmd_pause(args) -> int: return _set_paused(True)
def _cmd_resume(args) -> int: return _set_paused(False)


_PIN_MESSAGES = {
    True: (
        "cannot pin (only agent-created skills participate in curation)",
        "could not pin '{skill}' — the skill is not curation-eligible (protected built-in or "
        "external). `hermes curator list-unmanaged` shows which skills the curator tracks.",
        # Unmanaged skills are never auto-transitioned, so the pin is recorded but only
        # becomes protective once the skill is adopted — say so and point at `adopt`.
        "pinned '{skill}' (recorded; this skill is unmanaged — auto-transitions never consider "
        "it. Run `hermes curator adopt {skill}` to put it under curator management)",
        "pinned '{skill}' (will bypass auto-transitions)"),
    False: (
        "there's nothing to unpin (curator only tracks agent-created skills)",
        "could not unpin '{skill}' — the skill is not curation-eligible (protected built-in or "
        "external).",
        "unpinned '{skill}' (recorded; this skill is unmanaged — it was never under "
        "auto-transitions to begin with)",
        "unpinned '{skill}'")}


def _set_pin(args, pinned: bool) -> int:
    from tools import skill_usage
    not_agent, not_eligible, unmanaged, done = _PIN_MESSAGES[pinned]
    skill = args.skill
    if not skill_usage.is_agent_created(skill):
        print(f"curator: '{skill}' is bundled or hub-installed — {not_agent}")
        return 1
    if not skill_usage.set_pinned(skill, pinned):
        print("curator: " + not_eligible.replace("{skill}", skill))
        return 1
    if not skill_usage.is_curator_managed(skill):
        print("curator: " + unmanaged.replace("{skill}", skill))
        return 0
    print("curator: " + done.replace("{skill}", skill))
    return 0


def _cmd_pin(args) -> int: return _set_pin(args, True)
def _cmd_unpin(args) -> int: return _set_pin(args, False)


def _cmd_list_unmanaged(args) -> int:
    """Itemize the unmanaged population that `status` summarizes (input for `adopt`)."""
    from tools import skill_usage
    rows = skill_usage.unmanaged_report()
    if not rows:
        print("curator: no unmanaged skills — every eligible skill is managed")
        return 0
    print(f"unmanaged skills ({len(rows)}):")
    for r in sorted(rows, key=lambda x: x["name"]):
        why = "created_by:null" if r.get("has_provenance_key") else "no marker"
        print(
            f"  {r['name']:44s} activity={r.get('activity_count', 0):4d}  "
            f"last_activity={_fmt_ts(r.get('last_activity_at')):14s}  ({why})")
    print("\nadopt one with `hermes curator adopt <name>`, "
          "or all with `hermes curator adopt --all-unmanaged`")
    return 0


def _cmd_adopt(args) -> int:
    """Hand unmanaged skills to the curator by explicit user declaration: provenance cannot be
    inferred from telemetry (a high patch count proves the agent MAINTAINS a skill, not that it
    AUTHORED it)."""
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
    if getattr(args, "dry_run", False):
        print(f"curator: would adopt {len(names)} skill(s) (dry run):")
        for n in names:
            print(f"  + {n}")
        return 0
    # Bulk adoption is a lifecycle change (adopted skills become archivable): confirm.
    if adopt_all and not getattr(args, "yes", False):
        print(f"curator: adopt {len(names)} unmanaged skill(s) into curator management?")
        print("  they become eligible for automatic staleness + archival")
        if not _confirm("  proceed? [y/N] ", "curator: aborted", eof_prefix=""):
            return 1
    failed = 0
    for n in names:
        ok, msg = skill_usage.adopt_skill(n)
        print(f"curator: {msg}")
        failed += not ok
    if len(names) > 1:
        print(f"curator: adopted {len(names) - failed}/{len(names)}")
    return 1 if failed else 0


def _as_user(fn, skill: str) -> int:
    """Run a skill mutation with the ledger actor set to ``user``; print and map its result."""
    from tools import skill_ledger
    tok = skill_ledger.set_ledger_actor("user")
    try:
        ok, msg = fn(skill)
    finally:
        skill_ledger.reset_ledger_actor(tok)
    print(f"curator: {msg}")
    return 0 if ok else 1


def _cmd_restore(args) -> int:
    from tools import skill_usage
    return _as_user(skill_usage.restore_skill, args.skill)


def _cmd_archive(args) -> int:
    """Manually archive an agent-created skill. Refuses if pinned."""
    from tools import skill_usage
    if skill_usage.get_record(args.skill).get("pinned"):
        print(
            f"curator: '{args.skill}' is pinned — unpin first with "
            f"`hermes curator unpin {args.skill}`")
        return 1
    return _as_user(skill_usage.archive_skill, args.skill)


def _idle_days(record: dict) -> Optional[int]:
    """Days since last activity, falling back to ``created_at`` so never-used skills aren't
    immortal; None only when both fields are missing or unparseable."""
    ts = record.get("last_activity_at") or record.get("created_at")
    dt = _parse_ts(str(ts)) if ts else None
    return None if dt is None else max(0, (datetime.now(timezone.utc) - dt).days)


def _cmd_prune(args) -> int:
    """Bulk-archive curator-managed skills idle for >= N days (pinned exempt, archived skipped)."""
    from tools import skill_usage
    days = getattr(args, "days", 90)
    if days < 1:
        print(f"curator: --days must be >= 1 (got {days})", file=sys.stderr)
        return 2
    candidates = [
        (r["name"], idle) for r in skill_usage.curated_report()
        if not (r.get("pinned") or r.get("state") == skill_usage.STATE_ARCHIVED)
        and (idle := _idle_days(r)) is not None and idle >= days]
    if not candidates:
        print(f"curator: nothing to prune (no unpinned skills idle >= {days}d)")
        return 0
    candidates.sort(key=lambda c: -c[1])
    print(f"curator: {len(candidates)} skill(s) idle >= {days}d:")
    for name, idle in candidates:
        print(f"  {name:40s} idle {idle}d")
    if getattr(args, "dry_run", False):
        print("\n(dry run — no changes made)")
        return 0
    if not getattr(args, "yes", False) and not _confirm(
        f"\nArchive {len(candidates)} skill(s)? [y/N] ", "curator: aborted"):
        return 1
    results = [(name, *skill_usage.archive_skill(name)) for name, _ in candidates]
    failures = [(name, msg) for name, ok, msg in results if not ok]
    print(f"\ncurator: archived {len(results) - len(failures)}/{len(candidates)}")
    if failures:
        print("failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


def _cmd_backup(args) -> int:
    """Manual skills-tree snapshot (same mechanism as the automatic pre-run snapshot)."""
    from agent import curator_backup
    if not curator_backup.is_enabled():
        print(
            "curator: backups are disabled via config "
            "(`curator.backup.enabled: false`); re-enable to snapshot")
        return 1
    snap = curator_backup.snapshot_skills(reason=getattr(args, "reason", None) or "manual")
    if snap is None:
        print("curator: snapshot failed — check logs (backup disabled or IO error)")
        return 1
    print(f"curator: snapshot created at ~/.hermes/skills/.curator_backups/{snap.name}")
    return 0


def _cmd_ledger(args) -> int:
    """List per-mutation audit ledger entries (newest first)."""
    from tools import skill_ledger
    rows = skill_ledger.list_entries(
        skill=getattr(args, "skill", None), limit=getattr(args, "limit", None) or 20)
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
            f"{r.get('skill', '?')}{extra}")
    print(
        "\nRoll back a single mutation with `hermes curator rollback <id>`; "
        "whole-tree snapshots remain available via `hermes curator rollback --list`.")
    return 0


def _cmd_purge(args) -> int:
    """Delete archived skills older than curator.archive_ttl_days. Explicit only — never
    automatic; each purged skill is captured (before-blobs) and recorded as a 'purge' ledger
    entry, so even a purge is auditable and blob-recoverable."""
    import shutil
    import time
    from hermes_cli.config import cfg_get, load_config
    from tools import skill_ledger
    from tools.skill_usage import _archive_dir
    ttl_days = getattr(args, "days", None)
    if ttl_days is None:
        ttl_days = int(cfg_get(load_config(), "curator", "archive_ttl_days", default=0) or 0)
    if ttl_days <= 0:
        print(
            "curator: purge disabled (curator.archive_ttl_days is 0). Set the "
            "config key or pass --days N to purge archives older than N days.")
        return 1
    archive_root = _archive_dir()
    if not archive_root.exists():
        print("curator: no archive directory — nothing to purge.")
        return 0
    cutoff = time.time() - ttl_days * 86400
    candidates = sorted(
        p for p in archive_root.iterdir() if p.is_dir() and p.stat().st_mtime < cutoff)
    if not candidates:
        print(f"curator: no archived skills older than {ttl_days}d.")
        return 0
    print(f"Archived skills older than {ttl_days}d:")
    for p in candidates:
        print(f"  {p.name}")
    if getattr(args, "dry_run", False):
        print("(dry run — nothing deleted)")
        return 0
    if not getattr(args, "yes", False) and not _confirm(
        f"Permanently delete {len(candidates)} archived skill(s)? [y/N] "):
        return 1
    purged = 0
    for p in candidates:
        before = skill_ledger.capture_before(p, complete_package=True, skill=p.name)
        try:
            shutil.rmtree(p)
        except OSError as e:
            print(f"curator: failed to purge {p.name}: {e}")
            continue
        skill_ledger.append_entry(
            "purge", p.name, before=before or [], after=[], actor="user",
            evidence={"ttl_days": ttl_days})
        purged += 1
    print(f"curator: purged {purged} archived skill(s). Ledger entries recorded.")
    return 0


def _report_rollback(ok: bool, msg: str) -> int:
    print(f"curator: {msg}" if ok else f"curator: rollback failed — {msg}")
    return 0 if ok else 1


def _rollback_ledger_entry(args, entry_id: str) -> int:
    """Restore exactly the files touched by one ledger entry (from content-addressed blobs); a
    pre-rollback safety ledger entry is taken first and it fails closed if that capture fails."""
    from tools import skill_ledger
    entry = skill_ledger.get_entry(entry_id)
    if entry is None:
        print(
            f"curator: no ledger entry '{entry_id}'. "
            "See `hermes curator ledger` for entry ids, or use "
            "`--id <snapshot>` for whole-tree snapshot rollback.")
        return 1
    print(f"Rollback target: ledger entry {entry_id}")
    print(f"  action: {entry.get('action', '?')}")
    print(f"  skill:  {entry.get('skill', '?')}")
    print(f"  actor:  {entry.get('actor', '?')}")
    print(f"  when:   {entry.get('ts', '?')}")
    touched = {i.get("path") for i in (entry.get("before") or []) + (entry.get("after") or [])}
    print(f"  files:  {len(touched)}")
    if not getattr(args, "yes", False) and not _confirm(
        "Restore this mutation's before-state? [y/N] "):
        return 1
    return _report_rollback(*skill_ledger.rollback_entry(entry_id))


def _cmd_rollback(args) -> int:
    """Restore the skills tree from a snapshot, or a single mutation from the audit ledger."""
    from agent import curator_backup
    entry_id = getattr(args, "entry_id", None)
    if entry_id:
        return _rollback_ledger_entry(args, entry_id)
    if getattr(args, "list", False):
        print(curator_backup.summarize_backups())
        return 0
    backup_id = getattr(args, "backup_id", None)
    target_path = curator_backup._resolve_backup(backup_id)
    if target_path is None:
        if not curator_backup.list_backups():
            print(
                "curator: no snapshots exist yet. Take one with "
                "`hermes curator backup` or wait for the next curator run.")
        else:
            print(
                f"curator: no snapshot matching "
                f"{'id ' + repr(backup_id) if backup_id else 'your query'}.")
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
                    f"(will be restored for skill-link fields only)")
            else:
                print(f"  cron jobs:   not in snapshot ({cron.get('reason', 'not captured')})")
    print(
        "\nThis will replace the current ~/.hermes/skills/ tree (a safety "
        "snapshot of the current state is taken first so this is undoable). "
        "Cron jobs that still exist will have their skills/skill fields "
        "restored from the snapshot; all other cron fields are left alone.")
    if not getattr(args, "yes", False) and not _confirm("Proceed? [y/N] "):
        return 1
    ok, msg, _ = curator_backup.rollback(backup_id=target_path.name)
    return _report_rollback(ok, msg)


def _cmd_list_archived(args) -> int:
    """List archived (recoverable) skills."""
    from tools import skill_usage
    names = skill_usage.list_archived_skill_names()
    print("\n".join(names) if names else "curator: no archived skills")
    return 0


_USAGE_SORTS = {
    "name": (lambda r: r["name"], False),
    "recent": (lambda r: r.get("last_activity_at") or "", True),
    "activity": (lambda r: r.get("activity_count", 0), True)}


def _cmd_usage(args) -> int:
    """Usage telemetry for ALL skills on disk (bundled + hub included), with provenance."""
    import json as _json
    from tools import skill_usage
    rows = skill_usage.usage_report()
    prov_filter = getattr(args, "provenance", None)
    if prov_filter:
        rows = [r for r in rows if r.get("provenance") == prov_filter]
    # name: alphabetical; recent: most-recently-active first (never-active sinks); activity
    # (default): most-used first.
    key, reverse = _USAGE_SORTS.get(getattr(args, "sort", "activity"), _USAGE_SORTS["activity"])
    rows.sort(key=key, reverse=reverse)
    if getattr(args, "json", False):
        print(_json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("curator: no skills found")
        return 0
    provenance = [r.get("provenance", "agent") for r in rows]
    counts = {k: provenance.count(k) for k in ("agent", "bundled", "hub")}
    print(
        f"skills: {len(rows)} total  "
        f"(agent={counts['agent']}  bundled={counts['bundled']}  hub={counts['hub']})\n")
    print(
        f"  {'skill':40s}  {'origin':8s}  "
        f"{'use':>4s}  {'view':>4s}  {'patch':>5s}  {'act':>4s}  last_activity")
    for r in rows:
        print(
            f"  {r['name'][:40]:40s}  "
            f"{r.get('provenance', 'agent'):8s}  "
            f"{r.get('use_count', 0):>4d}  "
            f"{r.get('view_count', 0):>4d}  "
            f"{r.get('patch_count', 0):>5d}  "
            f"{r.get('activity_count', 0):>4d}  "
            f"{_fmt_ts(r.get('last_activity_at'))}")
    return 0


def _arg(*flags, **kwargs):
    return flags, kwargs


_SKILL = _arg("skill", help="Skill name")
_YES = _arg("-y", "--yes", action="store_true", help="Skip the confirmation prompt")
_STORE_TRUE = dict(action="store_true")

# argparse wiring: (name, help, handler, *(flags, add_argument kwargs))
_SUBCOMMANDS = (
    ("status", "Show curator status and skill stats", _cmd_status),
    (
        "usage", "Show usage telemetry for ALL skills (built-in, hub, agent) with provenance",
        _cmd_usage,
        _arg("--sort", choices=("activity", "recent", "name"), default="activity",
             help="Sort order: activity (most-used first, default), recent "
                  "(most-recently-active first), or name (alphabetical)"),
        _arg("--provenance", choices=("agent", "bundled", "hub"), default=None,
             help="Only show skills of this origin"),
        _arg("--json", **_STORE_TRUE, help="Emit the full report as JSON instead of a table")),
    (
        "run", "Trigger a curator review now", _cmd_run,
        _arg("--sync", "--synchronous", dest="synchronous", **_STORE_TRUE,
             help="Wait for the LLM review pass to finish (default for manual runs)"),
        _arg("--background", dest="background", **_STORE_TRUE,
             help="Start the LLM review pass in a background thread and return immediately"),
        _arg("--dry-run", dest="dry_run", **_STORE_TRUE,
             help="Report only — no state changes, no archives, no consolidation "
                  "(use this to preview what curator would do)"),
        _arg("--consolidate", dest="consolidate", **_STORE_TRUE,
             help="Force the LLM umbrella-building consolidation pass on for this "
                  "run, overriding the config default (off). Without this flag the "
                  "run is prune-only unless `curator.consolidate: true` is set.")),
    ("pause", "Pause the curator until resumed", _cmd_pause),
    ("resume", "Resume a paused curator", _cmd_resume),
    ("pin", "Pin a skill so the curator never auto-transitions it", _cmd_pin, _SKILL),
    ("unpin", "Unpin a skill", _cmd_unpin, _SKILL),
    ("list-unmanaged", "List curation-eligible skills with no provenance marker",
     _cmd_list_unmanaged),
    (
        "adopt", "Hand unmanaged skills to the curator (provenance is a user declaration)",
        _cmd_adopt,
        _arg("skill", nargs="*", help="Skill name(s) to adopt. Omit when using --all-unmanaged."),
        _arg("--all-unmanaged", **_STORE_TRUE,
             help="Adopt every curation-eligible skill that has no provenance marker"),
        _arg("--dry-run", **_STORE_TRUE,
             help="List what would be adopted without writing anything"),
        _arg("--yes", **_STORE_TRUE, help="Skip the confirmation prompt for --all-unmanaged")),
    ("restore", "Restore an archived skill", _cmd_restore, _SKILL),
    ("list-archived", "List archived skills", _cmd_list_archived),
    ("archive", "Manually archive a skill (move to .archive/, excluded from prompt)", _cmd_archive,
     _SKILL),
    (
        "prune", "Bulk-archive curator-managed skills idle for >= N days (default 90)", _cmd_prune,
        _arg("--days", type=int, default=90,
             help="Archive skills idle for at least N days (default: 90)"),
        _YES,
        _arg("--dry-run", dest="dry_run", **_STORE_TRUE,
             help="Show what would be archived without doing it")),
    (
        "backup",
        "Take a manual tar.gz snapshot of ~/.hermes/skills/ "
        "(curator also does this automatically before every real run)",
        _cmd_backup,
        _arg("--reason", default=None,
             help="Free-text label stored in manifest.json (default: 'manual')")),
    (
        "rollback",
        "Restore ~/.hermes/skills/ from a curator snapshot, or a single "
        "mutation by ledger entry id (see `hermes curator ledger`)",
        _cmd_rollback,
        _arg("entry_id", nargs="?", default=None,
             help="Ledger entry id for single-mutation rollback (from "
                  "`hermes curator ledger`). Omit for whole-tree snapshot rollback."),
        _arg("--list", **_STORE_TRUE, help="List available snapshots and exit without restoring"),
        _arg("--id", dest="backup_id", default=None,
             help="Snapshot id to restore (see `--list`); default: newest"),
        _arg("-y", "--yes", **_STORE_TRUE, help="Skip confirmation prompt")),
    (
        "ledger", "List the per-mutation skill audit ledger (all actors: curator/agent/user)",
        _cmd_ledger,
        _arg("--skill", default=None, help="Only show entries for this skill"),
        _arg("--limit", type=int, default=20, help="Max entries to show (default: 20)")),
    (
        "purge",
        "Delete archived skills older than curator.archive_ttl_days "
        "(explicit only — never automatic; recorded in the ledger)",
        _cmd_purge,
        _arg("--days", type=int, default=None,
             help="Override curator.archive_ttl_days for this invocation"),
        _arg("--dry-run", dest="dry_run", **_STORE_TRUE,
             help="Show what would be purged without deleting"),
        _YES))


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `curator` subcommands to *parent*."""
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="curator_command")
    for name, help_text, handler, *arguments in _SUBCOMMANDS:
        sub = subs.add_parser(name, help=help_text)
        for flags, arg_kwargs in arguments:
            sub.add_argument(*flags, **arg_kwargs)
        sub.set_defaults(func=handler)


def cli_main(argv=None) -> int:
    """Standalone entry (also usable by hermes_cli.main fallthrough)."""
    parser = argparse.ArgumentParser(prog="hermes curator")
    register_cli(parser)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
