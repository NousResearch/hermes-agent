"""Cron subcommand for hermes CLI."""

import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color


def _normalize_skills(single_skill=None, skills: Optional[Iterable[str]] = None) -> Optional[List[str]]:
    """Deduped, stripped skill names; None when neither argument was given."""
    if skills is None and single_skill is None:
        return None
    normalized: List[str] = []
    for item in list(skills) if skills is not None else [single_skill]:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _cron_api(**kwargs):
    from tools.cronjob_tools import cronjob as cronjob_tool
    return json.loads(cronjob_tool(**kwargs))


def _active_cron_provider_name() -> str:
    """Resolved cron scheduler provider name ('builtin', 'chronos', …); 'builtin' on failure."""
    try:
        from cron.scheduler_provider import resolve_cron_scheduler
        return resolve_cron_scheduler().name or "builtin"
    except Exception:
        return "builtin"


def _builtin_gateway_liveness() -> Optional[bool]:
    """Tri-state liveness of the builtin cron scheduler's trigger (None = unknown).

    The builtin ticker only runs inside the gateway process, so a scheduled job with no live
    gateway can never fire; non-builtin providers fire jobs without the gateway.

    Chronos) fire through their own machinery and are deliberately exempt — a missing gateway process means
    nothing for them, so they report active. ``None`` = probe failed; callers must not claim either way. See
    #87033.
    """
    try:
        if _active_cron_provider_name() != "builtin":
            return True
        # The runtime lock is held for exactly the gateway's lifetime — more reliable than PID
        # scanning (find_gateway_pids transiently misses the gateway right after a restart, and
        # inside the gateway it must never say "not running"). A crashing probe is "unknown".
        with contextlib.suppress(Exception):
            from gateway.status import is_gateway_runtime_lock_active
            if is_gateway_runtime_lock_active():
                return True
        from hermes_cli.gateway import (
            find_gateway_pids, named_profile_served_by_running_multiplexer)
        # Satellite profile: no local gateway.pid, but the default multiplexer ticks its store.
        return bool(find_gateway_pids()) or named_profile_served_by_running_multiplexer()
    except Exception:
        return None


def _warn_if_gateway_not_running() -> None:
    """Warn that scheduled jobs won't fire unless the gateway is running (the #1 cron report).

    False is the only warn-worthy liveness state (None = unknown).

    The cron ticker only runs inside the gateway (``_start_cron_ticker`` in gateway/run.py); there is no
    standalone cron daemon. Without a running gateway, ``next_run_at`` passes but jobs never fire and
    ``last_run_at`` stays null — the most common cron support report (#51038). Surfacing this at create/list
    time, when the user is right there, prevents it.
    """
    if _builtin_gateway_liveness() is not False:
        return
    print(color("  ⚠  Gateway is not running — jobs won't fire automatically.", Colors.YELLOW))
    print(color("     Start it with: hermes gateway install\n"
                "                    sudo hermes gateway install --system  # Linux servers\n"
                "     Check status:  hermes cron status", Colors.DIM))


def _format_lateness(seconds: float) -> str:
    """Render a lateness duration compactly: '31m', '2h 30m', '45s'."""
    try:
        seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = [(days, "d"), (hours, "h"), (minutes if not days else 0, "m")]
    return " ".join(f"{n}{unit}" for n, unit in parts if n) or "0m"


def _dispatch_display(dispatch: dict) -> Optional[str]:
    """One-line scheduled-vs-actual dispatch summary; None when the stamp is malformed.

    On-time dispatches render dim; late/catch-up dispatches render loudly so a run fired long
    after gateway downtime doesn't look like an ordinary success.

    See #99879.
    """
    if not isinstance(dispatch, dict):
        return None
    scheduled, actual, kind = (dispatch.get(k) for k in ("scheduled_at", "dispatched_at", "kind"))
    if not scheduled or not actual or not kind:
        return None
    lateness = _format_lateness(dispatch.get("lateness_seconds", 0))
    if kind == "on_time":
        return color(f"on time (scheduled {scheduled})", Colors.DIM)
    label = "catch-up after missed fire" if kind == "catch_up" else "late"
    return (color(f"⚠ {label}: ", Colors.YELLOW) + f"scheduled {scheduled}, ran {actual} "
            + color(f"({lateness} late)", Colors.YELLOW))


def _print_banner(title: str) -> None:
    """Boxed cyan section header shared by ``cron list`` and ``cron incidents``."""
    print()
    rule = "─" * 73
    for line in (f"┌{rule}┐", "│" + " " * 25 + title.ljust(48) + "│", f"└{rule}┘"):
        print(color(line, Colors.CYAN))
    print()


def _unverified_targets(unverified) -> str:
    return ", ".join(map(str, unverified)) if isinstance(unverified, list) else str(unverified)


_STATE_BADGES = {"paused": ("[paused]", Colors.YELLOW), "completed": ("[completed]", Colors.BLUE)}


def cron_list(show_all: bool = False):
    """List all scheduled jobs."""
    from cron.jobs import effective_job_state, list_jobs
    jobs = list_jobs(include_disabled=show_all)

    if not jobs:
        print(color("No scheduled jobs.\nCreate one with 'hermes cron create ...' "
                    "or the /cron command in chat.", Colors.DIM))
        return

    _print_banner("Scheduled Jobs")

    for job in jobs:
        # effective_job_state honours the scheduler flag — never [paused] when enabled=true.
        badge = _STATE_BADGES.get(effective_job_state(job)) or (
            ("[active]", Colors.GREEN) if job.get("enabled", True) else ("[disabled]", Colors.RED))
        print(f"  {color(job.get('id', '?'), Colors.YELLOW)} {color(*badge)}")
        for label, value in _job_rows(job):
            print(f"    {label + ':':<11}{value}")
        for line in _job_warnings(job):
            print(f"    {line}")
        print()

    _warn_if_gateway_not_running()


def _last_run_display(job: Dict[str, Any]) -> str:
    last_status = job["last_status"]
    if last_status == "ok":
        return color("ok", Colors.GREEN)
    if last_status == "delivery_failed":
        # Agent succeeded but the result never reached the user — not green; last_error is None.
        return color(f"delivery_failed: {job.get('last_delivery_error') or '?'}", Colors.YELLOW)
    display = color(f"{last_status}: {job.get('last_error', '?')}", Colors.RED)
    streak = int(job.get("failure_streak") or 0)
    if streak >= 2:
        display += color(f"  ({streak} failures in a row)", Colors.RED)
    return display


def _job_rows(job: Dict[str, Any]) -> List[tuple[str, str]]:
    """``(label, value)`` detail rows for one job in ``cron list``."""
    # `repeat` / `deliver` may be present-but-null (dict-default only covers a missing key).
    repeat_info = job.get("repeat") or {}
    repeat_times = repeat_info.get("times")
    # `deliver` may be present-but-null in the job record (same pitfall as `repeat` above), so coalesce to
    # the default rather than relying on the dict-default, which only applies to a missing key. A null value
    # would otherwise reach `", ".join(None)` and crash the whole listing (#32896).
    deliver = job.get("deliver") or ["local"]
    skills = job.get("skills") or ([job["skill"]] if job.get("skill") else [])
    monitor_source = job.get("monitor_script") or job.get("monitor_url")
    mon_state = job.get("monitor_state") or {}
    latest_execution = job.get("latest_execution") or {}
    optional = [
        ("Skills", ", ".join(skills) if skills else ""),
        ("Script", job.get("script")),
        ("Monitor", f"{monitor_source} (agent runs only on output change)" if monitor_source
         else ""),
        ("Changed", mon_state.get("last_changed_at") if monitor_source else ""),
        ("Mode", color("no-agent", Colors.DIM) + " (script stdout delivered directly)"
         if job.get("no_agent") else ""),
        ("Workdir", job.get("workdir")),
        ("Last run", f"{job.get('last_run_at', '?')}  {_last_run_display(job)}"
         if job.get("last_status") else ""),
        ("Dispatch", _dispatch_display(job.get("last_dispatch"))),
        ("Execution", f"{latest_execution.get('status', '?')}  {latest_execution.get('id', '?')}"
         if latest_execution else "")]
    return [
        ("Name", job.get("name", "(unnamed)")),
        ("Schedule", job.get("schedule_display", job.get("schedule", {}).get("value", "?"))),
        ("Repeat", f"{repeat_info.get('completed', 0)}/{repeat_times}" if repeat_times else "∞"),
        ("Next run", job.get("next_run_at", "?")),
        ("Deliver", deliver if isinstance(deliver, str) else ", ".join(deliver)),
    ] + [(label, value) for label, value in optional if value]


def _job_warnings(job: Dict[str, Any]) -> List[str]:
    """Delivery / fire warning lines for one job in ``cron list``."""
    lines = []
    if job.get("last_delivery_error"):
        lines.append(f"{color('⚠ Delivery failed:', Colors.YELLOW)} {job['last_delivery_error']}")
    # A live adapter acked the last send but returned no message_id / raw_response
    # (Slack/Matrix/Mattermost shape): accepted as delivered, but say so here.
    if unverified := job.get("last_delivery_unverified"):
        lines.append(f"{color('⚠ Delivery UNVERIFIED:', Colors.YELLOW)} adapter acked "
                     f"{_unverified_targets(unverified)} without message_id/raw_response")
    fire_err = job.get("last_fire_error")
    if isinstance(fire_err, dict) and fire_err.get("detail"):
        lines.append(f"{color('⚠ Missed scheduled fire:', Colors.RED)} "
                     f"{fire_err.get('at', '?')}  {fire_err['detail']}")
    return lines


def cron_tick():
    """Run due jobs once and exit."""
    from cron.scheduler import CronTickYielded, tick
    try:
        tick(verbose=True)
    except CronTickYielded as exc:
        # Inert for a one-shot CLI (no boot fingerprint); report cleanly rather than traceback.
        print(color(f"✗ {exc}", Colors.YELLOW))
        print("  A fresher gateway process owns the runtime lock and will fire due jobs; this "
              "stale process yielded its tick.")
        return 1
    except OSError as exc:
        # Real lock-acquisition failures (EMFILE, EACCES) propagate; they are not contention.
        # For the one-shot CLI surface, report cleanly instead of dumping a traceback; the gateway ticker
        # loop handles its own retry. See #87644.
        print(color(f"✗ Cron tick failed: {exc}", Colors.RED))
        print("  Check `hermes cron status` and the gateway log for details.")
        return 1
    return 0


def cron_runs(job_id: Optional[str] = None, limit: int = 20):
    """Show indexed durable cron execution history."""
    from cron.executions import list_executions
    records = list_executions(job_id=job_id, limit=limit)
    if not records:
        print("No cron execution attempts recorded.")
        return
    for record in records:
        print(f"{record.get('id', '?')}  {record.get('status', '?'):<9}  "
              f"job={record.get('job_id', '?')}  source={record.get('source', '?')}  "
              f"{record.get('claimed_at', '?')}")
        if record.get("error"):
            print(f"    {record['error']}")


_INCIDENT_STATE_COLORS = {"detected": Colors.RED, "alerted": Colors.YELLOW, "closed": Colors.GREEN}


def cron_incidents(args) -> int:
    """List (``[--state <s>]``) or ``ack <id>`` durable cron failure incidents.

    Acking closes an incident so its failure ping stays silent until the error signature changes.
    """
    from cron.incidents import ack_incident, list_incidents
    action = getattr(args, "incident_action", "list")
    if action == "ack":
        incident_id = getattr(args, "incident_id", None)
        if not incident_id:
            print(color("✗ Incident ID required: hermes cron incidents ack <incident_id>", Colors.RED))
            return 1
        if ack_incident(incident_id):
            print(color(f"✓ Incident {incident_id} acknowledged (closed).", Colors.GREEN))
        else:
            print(color(f"Incident {incident_id} not found or already closed.", Colors.YELLOW))
        return 0

    state = getattr(args, "state", None)
    incidents = list_incidents(state=state)
    if not incidents:
        print(color("No cron failure incidents recorded.", Colors.DIM))
        if state:
            print(color(f"  (filtered by state '{state}')", Colors.DIM))
        return 0

    _print_banner("Cron Failure Incidents")
    for inc in incidents:
        state_display = color(inc["state"], _INCIDENT_STATE_COLORS.get(inc["state"], Colors.DIM))
        error_text = re.sub(r"\s+", " ", inc.get("error") or "").strip()
        if len(error_text) > 160:
            error_text = error_text[:157].rstrip() + "..."
        rows = [("Job", inc["job_id"]), ("Type", inc.get("failure_type", "unknown")),
                ("First seen", inc.get("first_seen_at", "?")),
                ("Last seen", inc.get("last_seen_at", "?")), ("Error", error_text),
                ("Output", inc.get("output_file"))]
        print(f"  {color(inc['id'], Colors.YELLOW)}  {state_display}")
        for label, value in rows:
            if label != "Output" or value:
                print(f"    {label + ':':<12}{value}")
        print()
    print(color(f"  {len(incidents)} incident(s)  |  ack one with: hermes cron incidents ack <id>",
                Colors.DIM))
    return 0


_PERMISSION_HINT = ("  Hint: jobs.json may be owned by another user (e.g. rewritten by a root "
                    "`docker exec hermes hermes cron ...`). Fix ownership to match the gateway "
                    "user, and prefer `docker exec -u <uid>:<gid>`.")
_FD_EXHAUSTION_HINT = ("  Hint: the ticker hit file-descriptor exhaustion (EMFILE). The scheduler "
                       "now retries with backoff and attempts fd reclamation, but if the leak "
                       "persists, restart the gateway to recover scheduling.")


def _print_ticker_health(pids: list) -> None:
    """Report builtin-ticker liveness for a gateway process known to be alive.

    The ticker THREAD can die silently or stay alive while every tick fails, so check both
    the liveness heartbeat and the last-successful-tick marker before saying "will fire".
    """
    # See #32612, #32895.
    from cron.jobs import (
        get_ticker_heartbeat_age, get_ticker_last_error, get_ticker_success_age,
        TICKER_INTERVAL_SECONDS)
    from cron.scheduler import _is_fd_exhaustion_text as _cron_is_fd_exhaustion_text
    STALE_AFTER = TICKER_INTERVAL_SECONDS * 3 + 20  # ~3 missed iterations + slack (200s @ 60s)
    hb_age = get_ticker_heartbeat_age()
    ok_age = get_ticker_success_age()
    pid_line = f"  PID: {', '.join(map(str, pids))}" if pids else None

    def _warn(headline: str) -> None:
        print(color(headline, Colors.YELLOW))
        if pid_line:
            print(pid_line)

    if hb_age is None:
        # Ticker never started (non-cron profile, gateway just started, or a config issue).
        _warn("⚠ Gateway is running but the cron ticker has not reported a heartbeat.")
        print("  Cron jobs will NOT fire until the ticker writes its first heartbeat.\n"
              "  If the gateway just started, wait ~60s and re-run `hermes cron status`.\n"
              "  If heartbeat never appears, restart: hermes gateway restart")
    elif hb_age > STALE_AFTER:  # ticker thread is gone
        _warn("⚠ Gateway is running but the cron ticker looks STALLED — "
              f"no heartbeat for {int(hb_age)}s (expected every ~60s).")
        print("  Cron jobs may NOT be firing. Restart: hermes gateway restart")
    elif ok_age is not None and ok_age > STALE_AFTER:  # loop alive but every tick fails
        _warn("⚠ Gateway and cron ticker are running, but no tick has "
              f"succeeded in {int(ok_age)}s — ticks may be failing.")
        last_error = get_ticker_last_error()
        if last_error:
            # WHY ticks fail: root-rewritten jobs.json (PermissionError) or fd exhaustion.
            # Show WHY ticks fail — e.g. a root-rewritten jobs.json (PermissionError) that silently locked
            # out the ticker's uid for ~14h in the field (#68483), or fd exhaustion (EMFILE) that used to
            # stall the scheduler invisibly (#87644).
            print(color(f"  Last tick error: {last_error}", Colors.RED))
            if "Permission denied" in last_error:
                print(color(_PERMISSION_HINT, Colors.YELLOW))
            elif _cron_is_fd_exhaustion_text(last_error):
                print(color(_FD_EXHAUSTION_HINT, Colors.YELLOW))
        print("  Check the gateway log for 'Cron tick error'.")
    else:
        print(color("✓ Gateway is running — cron jobs will fire automatically", Colors.GREEN))
        if pid_line:
            print(pid_line)
        if hb_age is not None:
            print(f"  Ticker heartbeat: {int(hb_age)}s ago")


def cron_status():
    """Show cron execution status."""
    from cron.jobs import list_jobs
    from hermes_cli.gateway import find_gateway_pids
    print()

    provider = _active_cron_provider_name()
    if provider != "builtin":
        # External providers fire via webhook: no ticker thread / heartbeat file by design, so
        # the liveness heuristics would always say "stalled".
        print(color(f"✓ Cron provider: {provider} — jobs fire via the managed scheduler, "
                    "not the in-process ticker.", Colors.GREEN))
        print(color("  (No ticker heartbeat is expected for an external provider; "
                    "due jobs are delivered by an authenticated webhook.)", Colors.DIM))
    else:
        pids = find_gateway_pids()
        gateway_alive_via_lock = False
        if not pids:
            # The pid scan transiently misses a live gateway right after a restart; the runtime
            # lock proves the process is alive. Declare "not running" only when both agree.
            with contextlib.suppress(Exception):
                # Same false-alarm class the cronjob tool fixed (#95947): the pid scan can transiently miss
                # a live gateway (just after a restart) while the runtime lock — held for exactly the
                # gateway's lifetime — proves the ticker's process is alive.
                from gateway.status import get_running_pid, is_gateway_runtime_lock_active
                gateway_alive_via_lock = is_gateway_runtime_lock_active()
                lock_pid = get_running_pid() if gateway_alive_via_lock else None
                pids = [lock_pid] if lock_pid else pids
        if pids or gateway_alive_via_lock:
            _print_ticker_health(pids)
        else:
            print(color("✗ Gateway is not running — cron jobs will NOT fire", Colors.RED))
            print("\n  To enable automatic execution:\n"
                  "    hermes gateway install    # Install as a user service\n"
                  "    sudo hermes gateway install --system  "
                  "# Linux servers: boot-time system service\n"
                  "    hermes gateway            # Or run in foreground")

    print()
    _print_active_jobs_summary(list_jobs(include_disabled=False))
    print()


def _print_active_jobs_summary(jobs) -> None:
    """Print the '<N> active job(s)' + next-run line shared by every status path."""
    if not jobs:
        print("  No active jobs")
        return
    next_runs = [j.get("next_run_at") for j in jobs if j.get("next_run_at")]
    print(f"  {len(jobs)} active job(s)")
    if next_runs:
        print(f"  Next run: {min(next_runs)}")
    # Post-downtime late fires show at status level, not just per-job in `cron list`.
    late = [j for j in jobs if isinstance(j.get("last_dispatch"), dict)
            and j["last_dispatch"].get("kind") in ("late", "catch_up")]
    if late:
        print()
        print(color(f"  ⚠ {len(late)} job(s) last fired late (missed-fire catch-up):",
                    Colors.YELLOW))
        for j in late:
            d = j["last_dispatch"]
            late_by = _format_lateness(d.get("lateness_seconds", 0))
            print(f"    {j.get('id', '?')}  {j.get('name', '(unnamed)')}: "
                  f"scheduled {d.get('scheduled_at', '?')}, ran {d.get('dispatched_at', '?')} "
                  + color(f"({late_by} late)", Colors.YELLOW))


def _scripts_dir_for_cron() -> Path:
    """Scripts dir for cron jobs — via ``CRON_DIR`` so monkeypatched cron storage is honoured."""
    from cron.jobs import CRON_DIR
    return CRON_DIR.parent / "scripts"


def _script_health_issue(script: str) -> Optional[str]:
    """Human-readable script issue, or ``None`` when the path is OK."""
    scripts_dir = _scripts_dir_for_cron().resolve()
    raw = Path(script).expanduser()
    path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()
    try:
        path.relative_to(scripts_dir)
    except ValueError:
        return f"script resolves outside HERMES_HOME/scripts: {script!r}"
    if not path.exists():
        return f"script not found: {path}"
    if not path.is_file():
        return f"script path is not a file: {path}"
    return None


# A busy tick can push dispatch a few minutes late; only a next_run_at parked well in the past
# means the job is silently not firing (ticker dead, gateway down, wedged fire-claim).
_OVERDUE_GRACE_SECONDS = 15 * 60


def _next_run_overdue_issue(next_run: str) -> Optional[str]:
    """Issue string when ``next_run_at`` is parked in the past."""
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(next_run.replace("Z", "+00:00"))
    except ValueError:
        return f"next_run_at is not a valid timestamp: {next_run!r}"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    overdue_s = (datetime.now(timezone.utc) - dt).total_seconds()
    if overdue_s <= _OVERDUE_GRACE_SECONDS:
        return None
    amount = f"{overdue_s / 3600:.1f}h" if overdue_s >= 3600 else f"{overdue_s / 60:.0f}m"
    return f"next_run_at is {amount} overdue — job is not firing (is the scheduler running?)"


def _cron_doctor_issues_for_job(job: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    last_status = str(job.get("last_status") or "").strip().lower()
    # "delivery_failed" = the agent run succeeded; the delivery issue below reports it.
    if last_status and last_status not in {"ok", "delivery_failed"}:
        issues.append(f"last run failed: {str(job.get('last_error') or 'unknown error').strip()}")
    if delivery_err := str(job.get("last_delivery_error") or "").strip():
        issues.append(f"last delivery failed: {delivery_err}")
    if unverified := job.get("last_delivery_unverified"):
        issues.append("last delivery unverified (adapter acked without evidence): "
                      + _unverified_targets(unverified))
    if job.get("enabled", True) and job.get("state") not in {"paused", "completed"}:
        next_run = str(job.get("next_run_at") or "").strip()
        issue = _next_run_overdue_issue(next_run) if next_run else "active job has no next_run_at"
        if issue:
            issues.append(issue)
    script = str(job.get("script") or "").strip()
    if job.get("no_agent") and not script:
        issues.append("no-agent job has no script")
    if script and (script_issue := _script_health_issue(script)):
        issues.append(script_issue)
    workdir = str(job.get("workdir") or "").strip()
    if workdir and not Path(workdir).expanduser().exists():
        issues.append(f"workdir not found: {workdir}")
    return issues


def cron_doctor() -> int:
    """Run read-only cron health checks and return a shell-friendly status."""
    from cron.jobs import list_jobs
    jobs = list_jobs(include_disabled=False)
    findings = [(job, issues) for job in jobs if (issues := _cron_doctor_issues_for_job(job))]
    if not findings:
        print(color("✓ Cron doctor found no issues", Colors.GREEN))
        note = f"  Checked {len(jobs)} active job(s)." if jobs else "  No active jobs configured."
        print(color(note, Colors.DIM))
        return 0
    issue_count = sum(len(issues) for _, issues in findings)
    print(color(f"Cron doctor found {issue_count} issue(s) across {len(findings)} job(s):", Colors.YELLOW))
    print()
    for job, issues in findings:
        print(f"  {color(job.get('id', '?'), Colors.YELLOW)} {job.get('name', '(unnamed)')}")
        for issue in issues:
            print(f"    - {issue}")
    print()
    print(color("Next: fix the listed job config, then run `hermes cron doctor` again.", Colors.DIM))
    return 1


_JOB_ARG_FIELDS = (("name", "name"), ("deliver", "deliver"), ("failure_deliver", "failure_deliver"),
                   ("repeat", "repeat"), ("script", "script"), ("workdir", "workdir"),
                   ("model", "model"), ("provider", "model_provider"),
                   ("monitor_script", "monitor_script"), ("monitor_url", "monitor_url"),
                   ("continuity", "continuity"), ("reasoning_effort", "reasoning_effort"))


def _job_api_kwargs(args) -> Dict[str, Any]:
    """Collect the create/update kwargs shared by ``cron create`` and ``cron edit``."""
    return {api_key: getattr(args, attr, None) for api_key, attr in _JOB_ARG_FIELDS}


_JOB_DETAIL_LINES = (
    ("script", "  Script: {}"),
    ("monitor_script", "  Monitor: {} (agent runs only on output change)"),
    ("monitor_url", "  Monitor: {} (agent runs only on output change)"),
    ("no_agent", "  Mode: no-agent (script stdout delivered directly)"),
    ("continuity", "  Continuity: on (each run sees the previous run's output)"),
    ("workdir", "  Workdir: {}"))


def _print_job_details(job_data: Dict[str, Any]) -> None:
    """Print the optional Script/Monitor/Mode/Continuity/Workdir lines of a job record."""
    for key, template in _JOB_DETAIL_LINES:
        if job_data.get(key):
            print(template.format(job_data[key]))


def cron_create(args):
    # The gateway-lifecycle guard lives in cron.jobs.create_job (every creation path); a block
    # surfaces as result["error"].
    result = _cron_api(
        action="create", schedule=args.schedule, prompt=args.prompt,
        skill=getattr(args, "skill", None),
        skills=_normalize_skills(getattr(args, "skill", None), getattr(args, "skills", None)),
        no_agent=getattr(args, "no_agent", False) or None, **_job_api_kwargs(args))
    if not result.get("success"):
        print(color(f"Failed to create job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    print(color(f"Created job: {result['job_id']}", Colors.GREEN))
    print(f"  Name: {result['name']}\n  Schedule: {result['schedule']}")
    if result.get("skills"):
        print(f"  Skills: {', '.join(result['skills'])}")
    _print_job_details(result.get("job", {}))
    print(f"  Next run: {result['next_run_at']}")
    _warn_if_gateway_not_running()
    return 0


def cron_edit(args):
    from cron.jobs import AmbiguousJobReference, resolve_job_ref
    try:
        job = resolve_job_ref(args.job_id)
    except AmbiguousJobReference as exc:
        print(color(str(exc), Colors.RED))
        for m in exc.matches:
            print(f"  {m['id']}  (name: {m.get('name')!r})")
        return 1
    if not job:
        print(color(f"Job not found: {args.job_id}", Colors.RED))
        return 1
    existing_skills = list(job.get("skills") or ([job["skill"]] if job.get("skill") else []))
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
        final_skills += [skill for skill in add_skills if skill not in final_skills]
    result = _cron_api(action="update", job_id=args.job_id,
                       schedule=getattr(args, "schedule", None),
                       prompt=getattr(args, "prompt", None), skills=final_skills,
                       no_agent=getattr(args, "no_agent", None), **_job_api_kwargs(args))
    if not result.get("success"):
        print(color(f"Failed to update job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    updated = result["job"]
    print(color(f"Updated job: {updated['job_id']}", Colors.GREEN))
    print(f"  Name: {updated['name']}\n  Schedule: {updated['schedule']}")
    print(f"  Skills: {', '.join(updated['skills'])}" if updated.get("skills") else
          "  Skills: none")
    _print_job_details(updated)
    return 0


def _job_action(action: str, job_id: str, success_verb: str) -> int:
    _stateless_token = None
    if action == "run":
        # One-shot CLI: a background-dispatched run (daemon thread, triggered when the CLI
        # inherits HERMES_SESSION_KEY) would be orphaned mid-LLM-call, leaving the execution row
        # stuck 'claimed'. Declaring the channel stateless forces a synchronous run; scoped to
        # this call so in-process callers (tests, embedding apps) are not tainted.
        with contextlib.suppress(Exception):
            # The background path in ``_try_dispatch_background_run`` triggers when the CLI inherits a
            # gateway/desktop session env (HERMES_SESSION_KEY); declare the channel stateless so
            # ``async_delivery_supported()`` gates it off and the run executes synchronously to completion
            # instead. See #86721.
            from gateway.session_context import _SESSION_ASYNC_DELIVERY
            _stateless_token = _SESSION_ASYNC_DELIVERY.set(False)
    try:
        result = _cron_api(action=action, job_id=job_id)
    finally:
        if _stateless_token is not None:
            _SESSION_ASYNC_DELIVERY.reset(_stateless_token)
    if not result.get("success"):
        print(color(f"Failed to {action} job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    job = result.get("job") or result.get("removed_job") or {}
    print(color(f"{success_verb} job: {job.get('name', job_id)} ({job_id})", Colors.GREEN))
    if action in {"resume", "run"} and result.get("job", {}).get("next_run_at"):
        print(f"  Next run: {result['job']['next_run_at']}")
    if action == "run":
        print(f"  {_run_outcome(result.get('job', {}))}")
    return 0


def _run_outcome(job: Dict[str, Any]) -> str:
    """One-line verdict for a manual run.

    A background-dispatched run (execution_mode="background" / delegation_id) keeps running
    after this CLI exits, so report the dispatch rather than a success/failure verdict.
    """
    if job.get("delegation_id"):
        return f"Running in background (delegation {job['delegation_id']})."
    if job.get("execution_mode") == "background":
        return "Running in background."
    if job.get("executed"):
        return f"Ran now: {'succeeded' if job.get('execution_success') else 'failed'}."
    return job.get("execution_skipped") or "It will run on the next scheduler tick."


def cron_resume(args) -> int:
    """Resume a paused job or explicitly re-arm a completed one-shot."""
    run_at = getattr(args, "run_at", None)
    run_now = getattr(args, "run_now", False)
    if run_at and run_now:
        print(color("Use exactly one of --at or --run-now.", Colors.RED))
        return 1
    if not run_at and not run_now:
        return _job_action("resume", args.job_id, "Resumed")
    from cron.jobs import AmbiguousJobReference, _hermes_now, rearm_oneshot
    try:
        job = rearm_oneshot(args.job_id, _hermes_now().isoformat() if run_now else run_at)
    except (AmbiguousJobReference, ValueError) as exc:
        print(color(f"Failed to re-arm job: {exc}", Colors.RED))
        return 1
    if not job:
        print(color(f"Job not found: {args.job_id}", Colors.RED))
        return 1
    print(color(f"Re-armed job: {job.get('name', args.job_id)} ({args.job_id})", Colors.GREEN)
          + f"\n  Next run: {job.get('next_run_at')}")
    return 0


def cron_notepad(args) -> int:
    """Handle ``hermes cron notepad <job_id> [get|set|delete|list]`` (per-job durable KV).

    A running cron agent updates its own notepad via its terminal tool; the scheduler injects
    non-empty notepads into the job prompt on each run.
    """
    from cron import notepad
    job_id = str(getattr(args, "job_id", "") or "")
    action = getattr(args, "notepad_action", None) or "list"
    key = getattr(args, "key", None)
    value = getattr(args, "value", None)
    if not job_id:
        print(color("A job ID is required.", Colors.RED))
        return 1
    try:
        if action not in ("set", "get", "delete"):  # list (default)
            notes = notepad.list_notes(job_id)
            if not notes:
                print(color(f"Notepad for job {job_id} is empty.", Colors.DIM))
            for note in notes:
                print(f"  {color(note['key'], Colors.YELLOW)} = {note['value']}\n"
                      f"    {color('updated: ' + str(note['updated_at']), Colors.DIM)}")
            return 0
        usage_args = "set <key> <value>" if action == "set" else f"{action} <key>"
        if key is None or (action == "set" and value is None):
            print(color(f"Usage: hermes cron notepad <job_id> {usage_args}", Colors.RED))
            return 1
        if action == "set":
            notepad.set_note(job_id, key, value)
            print(color(f"Set notepad key '{key}' for job {job_id}.", Colors.GREEN))
            return 0
        if action == "get":
            stored = notepad.get_note(job_id, key)
            if stored is not None:
                print(stored)
                return 0
        elif notepad.delete_note(job_id, key):
            print(color(f"Deleted notepad key '{key}' for job {job_id}.", Colors.GREEN))
            return 0
        print(color(f"No notepad key '{key}' for job {job_id}.", Colors.YELLOW))
        return 1
    except ValueError as exc:
        print(color(f"Notepad error: {exc}", Colors.RED))
        return 1


# Late-bound lambdas keep module-level monkeypatching working; list/status/runs return None -> 0.
_CRON_SUBCOMMANDS = {
    "list": lambda a: cron_list(getattr(a, "all", False)) or 0,
    "status": lambda a: cron_status() or 0,
    "doctor": lambda a: cron_doctor(),
    "tick": lambda a: cron_tick(),
    "runs": lambda a: cron_runs(getattr(a, "job_id", None), getattr(a, "limit", 20)) or 0,
    "incidents": lambda a: cron_incidents(a),
    "notepad": lambda a: cron_notepad(a),
    "create": lambda a: cron_create(a),
    "edit": lambda a: cron_edit(a),
    "pause": lambda a: _job_action("pause", a.job_id, "Paused"),
    "resume": lambda a: cron_resume(a),
    "run": lambda a: _job_action("run", a.job_id, "Triggered"),
    "remove": lambda a: _job_action("remove", a.job_id, "Removed")}
_CRON_SUBCOMMANDS["history"] = _CRON_SUBCOMMANDS["runs"]
_CRON_SUBCOMMANDS["add"] = _CRON_SUBCOMMANDS["create"]
_CRON_SUBCOMMANDS["rm"] = _CRON_SUBCOMMANDS["delete"] = _CRON_SUBCOMMANDS["remove"]


def cron_command(args):
    """Handle cron subcommands."""
    subcmd = getattr(args, 'cron_command', None)
    handler = _CRON_SUBCOMMANDS.get("list" if subcmd is None else subcmd)
    if handler is not None:
        return handler(args)
    print(f"Unknown cron command: {subcmd}\n"
          "Usage: hermes cron [list|create|edit|pause|resume|run|remove|status|runs|doctor|tick]")
    sys.exit(1)
