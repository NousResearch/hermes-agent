"""
Cron subcommand for hermes CLI.

Handles standalone cron management commands like list, create, edit,
pause/resume/run/remove, status, and tick.
"""

import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color


def _normalize_skills(single_skill=None, skills: Optional[Iterable[str]] = None) -> Optional[List[str]]:
    if skills is None:
        if single_skill is None:
            return None
        raw_items = [single_skill]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _cron_api(**kwargs):
    from tools.cronjob_tools import cronjob as cronjob_tool

    return json.loads(cronjob_tool(**kwargs))


def cron_list(show_all: bool = False, verbose: bool = False):
    """List all scheduled jobs."""
    from cron.jobs import list_jobs

    jobs = list_jobs(include_disabled=show_all)

    if not jobs:
        print(color("No scheduled jobs.", Colors.DIM))
        print(color("Create one with 'hermes cron create ...' or the /cron command in chat.", Colors.DIM))
        return

    print()
    print(color("┌─────────────────────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                         Scheduled Jobs                                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()

    for job in jobs:
        job_id = job.get("id", "?")
        name = job.get("name", "(unnamed)")
        schedule = job.get("schedule_display", job.get("schedule", {}).get("value", "?"))
        state = job.get("state", "scheduled" if job.get("enabled", True) else "paused")
        next_run = job.get("next_run_at", "?")

        repeat_info = job.get("repeat", {})
        repeat_times = repeat_info.get("times")
        repeat_completed = repeat_info.get("completed", 0)
        repeat_str = f"{repeat_completed}/{repeat_times}" if repeat_times else "∞"

        deliver = job.get("deliver", ["local"])
        if isinstance(deliver, str):
            deliver = [deliver]
        deliver_str = ", ".join(deliver)

        skills = job.get("skills") or ([job["skill"]] if job.get("skill") else [])
        if state == "paused":
            status = color("[paused]", Colors.YELLOW)
        elif state == "completed":
            status = color("[completed]", Colors.BLUE)
        elif job.get("enabled", True):
            status = color("[active]", Colors.GREEN)
        else:
            status = color("[disabled]", Colors.RED)

        print(f"  {color(job_id, Colors.YELLOW)} {status}")
        print(f"    Name:      {name}")
        print(f"    Schedule:  {schedule}")
        print(f"    Repeat:    {repeat_str}")
        print(f"    Next run:  {next_run}")
        print(f"    Deliver:   {deliver_str}")
        if skills:
            print(f"    Skills:    {', '.join(skills)}")
        script = job.get("script")
        if script:
            print(f"    Script:    {script}")
        model_override = job.get("model")
        provider_override = job.get("provider")
        if model_override or provider_override:
            model_str = model_override or "(default)"
            provider_str = provider_override or "(default)"
            print(f"    Model:     {model_str}  via  {provider_str}")

        # Execution history — show terminal outcome, not just 'dispatched'
        # last_status values: None (never run), 'delivered', 'silent', 'error'
        # Legacy: 'ok' is treated as synonym for 'delivered' (backward compat)
        last_status = job.get("last_status")
        if last_status:
            last_run = job.get("last_run_at", "?")
            if last_status in ("delivered", "ok"):
                # 'ok' is the legacy term for a delivered run (pre-#126 jobs)
                status_display = color("delivered", Colors.GREEN)
            elif last_status == "silent":
                status_display = color("silent (suppressed)", Colors.DIM)
            else:
                status_display = color(f"{last_status}", Colors.RED)
            print(f"    Last run:  {last_run}  {status_display}")

            # --verbose: surface last_error even on non-error outcomes (e.g. agent warnings)
            last_err = job.get("last_error")
            if verbose and last_err:
                print(f"    Last error: {color(last_err, Colors.RED)}")

        delivery_err = job.get("last_delivery_error")
        if delivery_err:
            print(f"    {color('⚠ Delivery failed:', Colors.YELLOW)} {delivery_err}")
            if verbose:
                print(f"    {color('  Detail:', Colors.DIM)} {delivery_err}")

        print()

    from hermes_cli.gateway import find_gateway_pids
    if not find_gateway_pids():
        print(color("  ⚠  Gateway is not running — jobs won't fire automatically.", Colors.YELLOW))
        print(color("     Start it with: hermes gateway install", Colors.DIM))
        print(color("                    sudo hermes gateway install --system  # Linux servers", Colors.DIM))
        print()


def cron_tick():
    """Run due jobs once and exit."""
    from cron.scheduler import tick
    tick(verbose=True)


def cron_status():
    """Show cron execution status."""
    from cron.jobs import list_jobs
    from hermes_cli.gateway import find_gateway_pids

    print()

    pids = find_gateway_pids()
    if pids:
        print(color("✓ Gateway is running — cron jobs will fire automatically", Colors.GREEN))
        print(f"  PID: {', '.join(map(str, pids))}")
    else:
        print(color("✗ Gateway is not running — cron jobs will NOT fire", Colors.RED))
        print()
        print("  To enable automatic execution:")
        print("    hermes gateway install    # Install as a user service")
        print("    sudo hermes gateway install --system  # Linux servers: boot-time system service")
        print("    hermes gateway            # Or run in foreground")

    print()

    jobs = list_jobs(include_disabled=False)
    if jobs:
        next_runs = [j.get("next_run_at") for j in jobs if j.get("next_run_at")]
        print(f"  {len(jobs)} active job(s)")
        if next_runs:
            print(f"  Next run: {min(next_runs)}")
    else:
        print("  No active jobs")

    print()


def cron_create(args):
    result = _cron_api(
        action="create",
        schedule=args.schedule,
        prompt=args.prompt,
        name=getattr(args, "name", None),
        deliver=getattr(args, "deliver", None),
        repeat=getattr(args, "repeat", None),
        skill=getattr(args, "skill", None),
        skills=_normalize_skills(getattr(args, "skill", None), getattr(args, "skills", None)),
        script=getattr(args, "script", None),
    )
    if not result.get("success"):
        print(color(f"Failed to create job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    print(color(f"Created job: {result['job_id']}", Colors.GREEN))
    print(f"  Name: {result['name']}")
    print(f"  Schedule: {result['schedule']}")
    if result.get("skills"):
        print(f"  Skills: {', '.join(result['skills'])}")
    job_data = result.get("job", {})
    if job_data.get("script"):
        print(f"  Script: {job_data['script']}")
    print(f"  Next run: {result['next_run_at']}")
    return 0


def cron_edit(args):
    from cron.jobs import get_job

    job = get_job(args.job_id)
    if not job:
        print(color(f"Job not found: {args.job_id}", Colors.RED))
        return 1

    existing_skills = list(job.get("skills") or ([] if not job.get("skill") else [job.get("skill")]))
    replacement_skills = _normalize_skills(getattr(args, "skill", None), getattr(args, "skills", None))
    add_skills = _normalize_skills(None, getattr(args, "add_skills", None)) or []
    remove_skills = set(_normalize_skills(None, getattr(args, "remove_skills", None)) or [])

    final_skills = None
    if getattr(args, "clear_skills", False):
        final_skills = []
    elif replacement_skills is not None:
        final_skills = replacement_skills
    elif add_skills or remove_skills:
        final_skills = [skill for skill in existing_skills if skill not in remove_skills]
        for skill in add_skills:
            if skill not in final_skills:
                final_skills.append(skill)

    # Resolve model/provider: empty string means explicit clear (set to None)
    raw_model = getattr(args, "model", None)
    raw_provider = getattr(args, "provider", None)
    model_update = raw_model if raw_model is None else (raw_model.strip() or None)
    provider_update = raw_provider if raw_provider is None else (raw_provider.strip() or None)

    result = _cron_api(
        action="update",
        job_id=args.job_id,
        schedule=getattr(args, "schedule", None),
        prompt=getattr(args, "prompt", None),
        name=getattr(args, "name", None),
        deliver=getattr(args, "deliver", None),
        repeat=getattr(args, "repeat", None),
        skills=final_skills,
        script=getattr(args, "script", None),
        model=model_update,
        provider=provider_update,
    )
    if not result.get("success"):
        print(color(f"Failed to update job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1

    updated = result["job"]
    print(color(f"Updated job: {updated['job_id']}", Colors.GREEN))
    print(f"  Name: {updated['name']}")
    print(f"  Schedule: {updated['schedule']}")
    if updated.get("skills"):
        print(f"  Skills: {', '.join(updated['skills'])}")
    else:
        print("  Skills: none")
    if updated.get("script"):
        print(f"  Script: {updated['script']}")
    if updated.get("model"):
        print(f"  Model: {updated['model']}  via  {updated.get('provider') or '(default provider)'}")
    elif raw_model is not None:
        print("  Model: (cleared — using global default)")
    return 0


def _job_action(action: str, job_id: str, success_verb: str) -> int:
    result = _cron_api(action=action, job_id=job_id)
    if not result.get("success"):
        print(color(f"Failed to {action} job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    job = result.get("job") or result.get("removed_job") or {}
    print(color(f"{success_verb} job: {job.get('name', job_id)} ({job_id})", Colors.GREEN))
    if action in {"resume", "run"} and result.get("job", {}).get("next_run_at"):
        print(f"  Next run: {result['job']['next_run_at']}")
    if action == "run":
        print("  It will run on the next scheduler tick.")
    return 0


def cron_migrate_model(args) -> int:
    """Bulk-update model/provider across jobs matching a filter."""
    from cron.jobs import list_jobs, update_job

    from_model = getattr(args, "from_model", None)
    from_provider = getattr(args, "from_provider", None)
    new_model = getattr(args, "model", None)
    new_provider = getattr(args, "provider", None)
    dry_run = getattr(args, "dry_run", False)

    if not new_model and not new_provider:
        print(color("Error: specify at least --model or --provider to set new values.", Colors.RED))
        return 1

    jobs = list_jobs(include_disabled=True)
    matched = []
    for job in jobs:
        current_model = (job.get("model") or "").strip()
        current_provider = (job.get("provider") or "").strip()
        if from_model and from_model.lower() not in current_model.lower():
            continue
        if from_provider and from_provider.lower() not in current_provider.lower():
            continue
        matched.append(job)

    if not matched:
        print(color("No jobs matched the filter.", Colors.YELLOW))
        return 0

    print()
    print(f"  {'[DRY RUN] ' if dry_run else ''}Matching jobs: {len(matched)}")
    for job in matched:
        jid = job.get("id", "?")
        jname = job.get("name", "(unnamed)")
        print(f"    {color(jid, Colors.YELLOW)} — {jname}")
        print(f"      model:    {job.get('model') or '(default)'}  →  {new_model or job.get('model') or '(default)'}")
        print(f"      provider: {job.get('provider') or '(default)'}  →  {new_provider or job.get('provider') or '(default)'}")
        if not dry_run:
            updates: dict = {}
            if new_model is not None:
                updates["model"] = new_model.strip() or None
            if new_provider is not None:
                updates["provider"] = new_provider.strip() or None
            result = update_job(jid, updates)
            if result:
                print(f"      {color('updated', Colors.GREEN)}")
            else:
                print(f"      {color('failed', Colors.RED)}")
    print()
    if dry_run:
        print(color("  Dry run complete — no changes written.", Colors.DIM))
    else:
        print(color(f"  {len(matched)} job(s) updated.", Colors.GREEN))
    return 0


def cron_command(args):
    """Handle cron subcommands."""
    subcmd = getattr(args, 'cron_command', None)

    if subcmd is None or subcmd == "list":
        show_all = getattr(args, 'all', False)
        verbose = getattr(args, 'verbose', False)
        cron_list(show_all, verbose=verbose)
        return 0

    if subcmd == "status":
        cron_status()
        return 0

    if subcmd == "tick":
        cron_tick()
        return 0

    if subcmd in {"create", "add"}:
        return cron_create(args)

    if subcmd == "edit":
        return cron_edit(args)

    if subcmd == "pause":
        return _job_action("pause", args.job_id, "Paused")

    if subcmd == "resume":
        return _job_action("resume", args.job_id, "Resumed")

    if subcmd == "run":
        return _job_action("run", args.job_id, "Triggered")

    if subcmd in {"remove", "rm", "delete"}:
        return _job_action("remove", args.job_id, "Removed")

    if subcmd == "migrate-model":
        return cron_migrate_model(args)

    print(f"Unknown cron command: {subcmd}")
    print("Usage: hermes cron [list|create|edit|pause|resume|run|remove|status|tick|migrate-model]")
    sys.exit(1)
