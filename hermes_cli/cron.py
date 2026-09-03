"""
Cron subcommand for hermes CLI.

Handles standalone cron management commands like list, create, edit,
pause/resume/run/remove, status, and tick.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color

# Gateway-lifecycle command detection lives in ``cron.lifecycle_guard`` so it
# can be shared across every job-creation path (CLI + the agent's ``cronjob``
# model tool via ``cron.jobs.create_job``) without a circular import. Re-export
# ``_contains_gateway_lifecycle_command`` here for back-compat: ``tools/
# terminal_tool.py`` imports it from this module to hard-block the same
# commands at execution time when ``_HERMES_GATEWAY=1``.
from cron.lifecycle_guard import (  # noqa: F401  (re-exported for terminal_tool)
    contains_gateway_lifecycle_command as _contains_gateway_lifecycle_command,
)


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


def _active_cron_provider_name() -> str:
    """Name of the resolved cron scheduler provider ('builtin', 'chronos', …).

    Best-effort + offline (``resolve_cron_scheduler`` reads config and the
    provider's ``is_available()`` contract forbids network). Returns 'builtin'
    on any failure so callers fall back to the historical ticker-based checks.
    """
    try:
        from cron.scheduler_provider import resolve_cron_scheduler

        return resolve_cron_scheduler().name or "builtin"
    except Exception:
        return "builtin"


def _builtin_gateway_liveness() -> Optional[bool]:
    """Tri-state liveness of the builtin cron scheduler's trigger.

    Single source of truth shared by the CLI (``_warn_if_gateway_not_running``)
    and the ``cronjob`` model tool (#87033): the builtin ticker only runs
    inside the gateway process, so a scheduled job with no live gateway can
    never fire. Non-builtin providers (e.g. Chronos) fire through their own
    machinery and are deliberately exempt — a missing gateway process means
    nothing for them, so they report active. ``None`` = probe failed; callers
    must not claim either way.
    """
    try:
        if _active_cron_provider_name() != "builtin":
            return True  # external provider fires jobs without the gateway
        # The gateway runtime lock is held for exactly the gateway's lifetime, so it
        # is a more reliable "is the ticker's process alive" signal than PID scanning
        # — and inside the gateway process it short-circuits to True, so the in-gateway
        # cron tool never emits a false "gateway not running" (find_gateway_pids can
        # transiently miss the gateway just after a restart).
        try:
            from gateway.status import is_gateway_runtime_lock_active

            if is_gateway_runtime_lock_active():
                return True
        except Exception:
            # A crashing lock probe is "unknown", not "dead" — let the pid
            # scan below still decide instead of collapsing the whole
            # tri-state to None.
            pass
        from hermes_cli.gateway import (
            find_gateway_pids,
            named_profile_served_by_running_multiplexer,
        )

        if find_gateway_pids():
            return True
        # Satellite profile: no local gateway.pid, but the default multiplexer
        # ticks this profile's cron store (#97120).
        return named_profile_served_by_running_multiplexer()
    except Exception:
        return None


def _warn_if_gateway_not_running() -> None:
    """Warn that scheduled jobs won't fire unless the gateway is running.

    The cron ticker only runs inside the gateway (``_start_cron_ticker`` in
    gateway/run.py); there is no standalone cron daemon. Without a running
    gateway, ``next_run_at`` passes but jobs never fire and ``last_run_at``
    stays null — the most common cron support report (#51038). Surfacing this
    at create/list time, when the user is right there, prevents it.

    An external provider (e.g. Chronos) fires jobs via a NAS-mediated webhook,
    NOT the in-process ticker, so a momentarily-absent gateway process does not
    mean jobs won't fire — the warning would be a false alarm. Stay quiet for
    any non-builtin provider; the gateway-process heuristic only speaks to the
    built-in ticker's trigger.
    """
    # _builtin_gateway_liveness never raises (it maps probe failures to None),
    # so no guard is needed here — False is the only warn-worthy state.
    if _builtin_gateway_liveness() is not False:
        return

    print(color("  ⚠  Gateway is not running — jobs won't fire automatically.", Colors.YELLOW))
    print(color("     Start it with: hermes gateway install", Colors.DIM))
    print(color("                    sudo hermes gateway install --system  # Linux servers", Colors.DIM))
    print(color("     Check status:  hermes cron status", Colors.DIM))


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
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    return " ".join(parts) or "0m"


def _dispatch_display(dispatch: dict) -> Optional[str]:
    """One-line scheduled-vs-actual dispatch summary for a job (#99879).

    Returns None when the stamp is malformed. On-time dispatches render a
    dim confirmation; late/catch-up dispatches render loudly so a run that
    fired 30–150 min after gateway downtime no longer looks like an
    ordinary on-time success.
    """
    if not isinstance(dispatch, dict):
        return None
    scheduled = dispatch.get("scheduled_at")
    actual = dispatch.get("dispatched_at")
    kind = dispatch.get("kind")
    if not scheduled or not actual or not kind:
        return None
    lateness = _format_lateness(dispatch.get("lateness_seconds", 0))
    if kind == "on_time":
        return color(f"on time (scheduled {scheduled})", Colors.DIM)
    label = "catch-up after missed fire" if kind == "catch_up" else "late"
    return (
        color(f"⚠ {label}: ", Colors.YELLOW)
        + f"scheduled {scheduled}, ran {actual} "
        + color(f"({lateness} late)", Colors.YELLOW)
    )


def cron_list(show_all: bool = False):
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

    from cron.jobs import effective_job_state

    for job in jobs:
        job_id = job.get("id", "?")
        name = job.get("name", "(unnamed)")
        schedule = job.get("schedule_display", job.get("schedule", {}).get("value", "?"))
        # Derive from the scheduler-honoured flag — never show [paused] when
        # enabled=true (half-paused contradiction must not look frozen).
        state = effective_job_state(job)
        next_run = job.get("next_run_at", "?")

        # `repeat` may be present-but-null in the job record (e.g. a one-shot
        # job persisted with "repeat": null), so coalesce to {} rather than
        # relying on the dict-default, which only applies to a missing key.
        repeat_info = job.get("repeat") or {}
        repeat_times = repeat_info.get("times")
        repeat_completed = repeat_info.get("completed", 0)
        repeat_str = f"{repeat_completed}/{repeat_times}" if repeat_times else "∞"

        # `deliver` may be present-but-null in the job record (same pitfall as
        # `repeat` above), so coalesce to the default rather than relying on the
        # dict-default, which only applies to a missing key. A null value would
        # otherwise reach `", ".join(None)` and crash the whole listing (#32896).
        deliver = job.get("deliver") or ["local"]
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
        monitor_source = job.get("monitor_script") or job.get("monitor_url")
        if monitor_source:
            print(f"    Monitor:   {monitor_source} (agent runs only on output change)")
            mon_state = job.get("monitor_state") or {}
            if mon_state.get("last_changed_at"):
                print(f"    Changed:   {mon_state['last_changed_at']}")
        if job.get("no_agent"):
            print(f"    Mode:      {color('no-agent', Colors.DIM)} (script stdout delivered directly)")
        workdir = job.get("workdir")
        if workdir:
            print(f"    Workdir:   {workdir}")

        # Execution history
        last_status = job.get("last_status")
        if last_status:
            last_run = job.get("last_run_at", "?")
            if last_status == "ok":
                status_display = color("ok", Colors.GREEN)
            elif last_status == "delivery_failed":
                # The agent succeeded but the result never reached the user —
                # not green, and the detail lives in last_delivery_error
                # (last_error is None for these runs).
                detail = job.get("last_delivery_error") or "?"
                status_display = color(f"delivery_failed: {detail}", Colors.YELLOW)
            else:
                status_display = color(f"{last_status}: {job.get('last_error', '?')}", Colors.RED)
                streak = int(job.get("failure_streak") or 0)
                if streak >= 2:
                    status_display += color(f"  ({streak} failures in a row)", Colors.RED)
            print(f"    Last run:  {last_run}  {status_display}")

        dispatch_line = _dispatch_display(job.get("last_dispatch"))
        if dispatch_line:
            print(f"    Dispatch:  {dispatch_line}")

        latest_execution = job.get("latest_execution")
        if latest_execution:
            print(
                f"    Execution: {latest_execution.get('status', '?')}  "
                f"{latest_execution.get('id', '?')}"
            )

        delivery_err = job.get("last_delivery_error")
        if delivery_err:
            print(f"    {color('⚠ Delivery failed:', Colors.YELLOW)} {delivery_err}")

        # A live adapter acked the last send but returned no message_id /
        # raw_response (Slack/Matrix/Mattermost shape): accepted as delivered,
        # but say so here rather than only in a WARNING log line.
        unverified = job.get("last_delivery_unverified")
        if unverified:
            targets = ", ".join(str(t) for t in unverified) if isinstance(unverified, list) else str(unverified)
            print(
                f"    {color('⚠ Delivery UNVERIFIED:', Colors.YELLOW)} "
                f"adapter acked {targets} without message_id/raw_response"
            )

        fire_err = job.get("last_fire_error")
        if isinstance(fire_err, dict) and fire_err.get("detail"):
            print(
                f"    {color('⚠ Missed scheduled fire:', Colors.RED)} "
                f"{fire_err.get('at', '?')}  {fire_err['detail']}"
            )

        print()

    _warn_if_gateway_not_running()


def cron_tick():
    """Run due jobs once and exit."""
    from cron.scheduler import CronTickYielded, tick
    try:
        tick(verbose=True)
    except CronTickYielded as exc:
        # Not expected on this surface (a one-shot CLI process has no boot
        # fingerprint, so the yield gate is inert) — but if a future caller
        # records one, report cleanly instead of a traceback.
        print(color(f"✗ {exc}", Colors.YELLOW))
        print(
            "  A fresher gateway process owns the runtime lock and will fire "
            "due jobs; this stale process yielded its tick."
        )
        return 1
    except OSError as exc:
        # tick() now propagates real lock-acquisition failures (EMFILE,
        # EACCES on open, ...) instead of swallowing them as contention
        # (#87644). For the one-shot CLI surface, report cleanly instead of
        # dumping a traceback; the gateway ticker loop handles its own retry.
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
        print(
            f"{record.get('id', '?')}  {record.get('status', '?'):<9}  "
            f"job={record.get('job_id', '?')}  source={record.get('source', '?')}  "
            f"{record.get('claimed_at', '?')}"
        )
        if record.get("error"):
            print(f"    {record['error']}")


_INCIDENT_STATE_COLORS = {
    "detected": Colors.RED,
    "alerted": Colors.YELLOW,
    "closed": Colors.GREEN,
}


def cron_incidents(args) -> int:
    """List or acknowledge durable cron failure incidents.

    ``hermes cron incidents [--state <s>]`` lists incidents (the stored error
    is redacted and truncated at write time, safe for terminal display);
    ``hermes cron incidents ack <id>`` closes one so its failure ping stays
    silent until the error signature changes.
    """
    from cron.incidents import ack_incident, list_incidents

    action = getattr(args, "incident_action", "list")
    if action == "ack":
        incident_id = getattr(args, "incident_id", None)
        if not incident_id:
            print(
                color(
                    "✗ Incident ID required: hermes cron incidents ack <incident_id>",
                    Colors.RED,
                )
            )
            return 1
        if ack_incident(incident_id):
            print(
                color(
                    f"✓ Incident {incident_id} acknowledged (closed).",
                    Colors.GREEN,
                )
            )
        else:
            print(
                color(
                    f"Incident {incident_id} not found or already closed.",
                    Colors.YELLOW,
                )
            )
        return 0

    state = getattr(args, "state", None)
    incidents = list_incidents(state=state)
    if not incidents:
        print(color("No cron failure incidents recorded.", Colors.DIM))
        if state:
            print(color(f"  (filtered by state '{state}')", Colors.DIM))
        return 0

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────────────────────┐",
            Colors.CYAN,
        )
    )
    print(
        color(
            "│                         Cron Failure Incidents                          │",
            Colors.CYAN,
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────────────────────┘",
            Colors.CYAN,
        )
    )
    print()
    for inc in incidents:
        state_display = color(
            inc["state"], _INCIDENT_STATE_COLORS.get(inc["state"], Colors.DIM)
        )
        print(f"  {color(inc['id'], Colors.YELLOW)}  {state_display}")
        print(f"    Job:        {inc['job_id']}")
        print(f"    Type:       {inc.get('failure_type', 'unknown')}")
        print(f"    First seen: {inc.get('first_seen_at', '?')}")
        print(f"    Last seen:  {inc.get('last_seen_at', '?')}")
        error_text = re.sub(r"\s+", " ", inc.get("error") or "").strip()
        if len(error_text) > 160:
            error_text = error_text[:157].rstrip() + "..."
        print(f"    Error:      {error_text}")
        if inc.get("output_file"):
            print(f"    Output:     {inc['output_file']}")
        print()
    print(
        color(
            f"  {len(incidents)} incident(s)  |  ack one with: "
            "hermes cron incidents ack <id>",
            Colors.DIM,
        )
    )
    return 0


def cron_status():
    """Show cron execution status."""
    from cron.jobs import list_jobs
    from hermes_cli.gateway import find_gateway_pids

    print()

    provider = _active_cron_provider_name()
    if provider != "builtin":
        # An external provider (e.g. Chronos) does NOT run the in-process 60s
        # ticker — it arms one external one-shot per job and is fired by a
        # NAS-mediated webhook, so between fires there is intentionally NO
        # ticker thread and NO heartbeat file. Reporting the ticker-heartbeat
        # staleness here would always say "stalled / not firing" on a perfectly
        # healthy Chronos instance. Report the provider instead and skip the
        # ticker-liveness heuristics entirely.
        print(color(
            f"✓ Cron provider: {provider} — jobs fire via the managed scheduler, "
            "not the in-process ticker.",
            Colors.GREEN,
        ))
        print(color(
            "  (No ticker heartbeat is expected for an external provider; "
            "due jobs are delivered by an authenticated webhook.)",
            Colors.DIM,
        ))
        print()
        _print_active_jobs_summary(list_jobs(include_disabled=False))
        print()
        return

    pids = find_gateway_pids()
    gateway_alive_via_lock = False
    if not pids:
        # Same false-alarm class the cronjob tool fixed (#95947): the pid scan
        # can transiently miss a live gateway (just after a restart) while the
        # runtime lock — held for exactly the gateway's lifetime — proves the
        # ticker's process is alive. Only declare "not running" when both the
        # scan AND the lock say so.
        try:
            from gateway.status import get_running_pid, is_gateway_runtime_lock_active

            if is_gateway_runtime_lock_active():
                gateway_alive_via_lock = True
                lock_pid = get_running_pid()
                if lock_pid:
                    pids = [lock_pid]
        except Exception:
            pass
    if pids or gateway_alive_via_lock:
        # The gateway PROCESS is alive — but the cron ticker THREAD inside it
        # can die silently, or stay alive while every tick fails. Check both
        # the liveness heartbeat and the last-successful-tick marker so we
        # don't report "will fire" when the ticker is dead or failing
        # (#32612, #32895).
        from cron.jobs import (
            get_ticker_heartbeat_age,
            get_ticker_last_error,
            get_ticker_success_age,
            TICKER_INTERVAL_SECONDS,
        )
        from cron.scheduler import _is_fd_exhaustion_text as _cron_is_fd_exhaustion_text

        # Allow ~3 missed ticker iterations (+ a little slack) before declaring
        # trouble. Derived from the shared interval constant so this threshold
        # tracks the ticker cadence instead of assuming a hardcoded 60s.
        STALE_AFTER = TICKER_INTERVAL_SECONDS * 3 + 20  # = 200s at the 60s default
        hb_age = get_ticker_heartbeat_age()
        ok_age = get_ticker_success_age()

        if hb_age is None:
            # No heartbeat file means the ticker thread has never started.
            # This can occur when:
            # - Gateway is running but not in a profile with cron enabled,
            # - Gateway was started moments ago (heartbeat is written after startup),
            # - Or a configuration issue is blocking the ticker from starting at all.
            print(color(
                "⚠ Gateway is running but the cron ticker has not reported a heartbeat.",
                Colors.YELLOW,
            ))
            if pids:
                print(f"  PID: {', '.join(map(str, pids))}")
            print("  Cron jobs will NOT fire until the ticker writes its first heartbeat.")
            print("  If the gateway just started, wait ~60s and re-run `hermes cron status`.")
            print("  If heartbeat never appears, restart: hermes gateway restart")
        elif hb_age > STALE_AFTER:
            # No heartbeat at all → the ticker thread is gone.
            print(color(
                "⚠ Gateway is running but the cron ticker looks STALLED — "
                f"no heartbeat for {int(hb_age)}s (expected every ~60s).",
                Colors.YELLOW,
            ))
            if pids:
                print(f"  PID: {', '.join(map(str, pids))}")
            print("  Cron jobs may NOT be firing. Restart: hermes gateway restart")
        elif ok_age is not None and ok_age > STALE_AFTER:
            # Loop is alive (fresh heartbeat) but no tick has SUCCEEDED in a
            # long time → ticks are failing every iteration.
            print(color(
                "⚠ Gateway and cron ticker are running, but no tick has "
                f"succeeded in {int(ok_age)}s — ticks may be failing.",
                Colors.YELLOW,
            ))
            if pids:
                print(f"  PID: {', '.join(map(str, pids))}")
            last_error = get_ticker_last_error()
            if last_error:
                # Show WHY ticks fail — e.g. a root-rewritten jobs.json
                # (PermissionError) that silently locked out the ticker's
                # uid for ~14h in the field (#68483), or fd exhaustion
                # (EMFILE) that used to stall the scheduler invisibly
                # (#87644).
                print(color(f"  Last tick error: {last_error}", Colors.RED))
                if "Permission denied" in last_error:
                    print(color(
                        "  Hint: jobs.json may be owned by another user "
                        "(e.g. rewritten by a root `docker exec hermes "
                        "hermes cron ...`). Fix ownership to match the "
                        "gateway user, and prefer `docker exec -u <uid>:<gid>`.",
                        Colors.YELLOW,
                    ))
                elif _cron_is_fd_exhaustion_text(last_error):
                    print(color(
                        "  Hint: the ticker hit file-descriptor exhaustion "
                        "(EMFILE). The scheduler now retries with backoff and "
                        "attempts fd reclamation, but if the leak persists, "
                        "restart the gateway to recover scheduling.",
                        Colors.YELLOW,
                    ))
            print("  Check the gateway log for 'Cron tick error'.")
        else:
            print(color("✓ Gateway is running — cron jobs will fire automatically", Colors.GREEN))
            if pids:
                print(f"  PID: {', '.join(map(str, pids))}")
            if hb_age is not None:
                print(f"  Ticker heartbeat: {int(hb_age)}s ago")
    else:
        print(color("✗ Gateway is not running — cron jobs will NOT fire", Colors.RED))
        print()
        print("  To enable automatic execution:")
        print("    hermes gateway install    # Install as a user service")
        print("    sudo hermes gateway install --system  # Linux servers: boot-time system service")
        print("    hermes gateway            # Or run in foreground")

    print()

    _print_active_jobs_summary(list_jobs(include_disabled=False))

    print()


def _print_active_jobs_summary(jobs) -> None:
    """Print the '<N> active job(s)' + next-run line shared by every status
    path (built-in ticker AND external provider)."""
    if jobs:
        next_runs = [j.get("next_run_at") for j in jobs if j.get("next_run_at")]
        print(f"  {len(jobs)} active job(s)")
        if next_runs:
            print(f"  Next run: {min(next_runs)}")
        # Missed-run visibility (#99879): call out jobs whose LAST dispatch
        # was late or a catch-up so post-downtime late fires are visible at
        # status level, not just buried per-job in `hermes cron list`.
        late = [
            j for j in jobs
            if isinstance(j.get("last_dispatch"), dict)
            and j["last_dispatch"].get("kind") in ("late", "catch_up")
        ]
        if late:
            print()
            print(color(
                f"  ⚠ {len(late)} job(s) last fired late (missed-fire catch-up):",
                Colors.YELLOW,
            ))
            for j in late:
                d = j["last_dispatch"]
                print(
                    f"    {j.get('id', '?')}  {j.get('name', '(unnamed)')}: "
                    f"scheduled {d.get('scheduled_at', '?')}, "
                    f"ran {d.get('dispatched_at', '?')} "
                    + color(
                        f"({_format_lateness(d.get('lateness_seconds', 0))} late)",
                        Colors.YELLOW,
                    )
                )
    else:
        print("  No active jobs")


def _scripts_dir_for_cron() -> Path:
    """Return the scripts directory used by cron jobs.

    Prefer ``cron.jobs.CRON_DIR.parent`` over a fresh ``get_hermes_home()`` call
    so tests and profile-aware callers that monkeypatch cron storage inspect the
    same Hermes home the jobs were loaded from.
    """
    from cron.jobs import CRON_DIR

    return CRON_DIR.parent / "scripts"


def _script_health_issue(script: str) -> Optional[str]:
    """Return a human-readable script issue, or ``None`` when the path is OK."""
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


# Grace period before an overdue ``next_run_at`` is reported. The ticker runs
# once a minute and a busy tick can push dispatch a few minutes late; only a
# next_run_at parked well in the past means the job is silently not firing
# (ticker dead, gateway down, or a wedged fire-claim).
_OVERDUE_GRACE_SECONDS = 15 * 60


def _next_run_overdue_issue(next_run: str) -> Optional[str]:
    """Return an issue string when ``next_run_at`` is parked in the past."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(next_run.replace("Z", "+00:00"))
    except ValueError:
        return f"next_run_at is not a valid timestamp: {next_run!r}"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    overdue_s = (datetime.now(timezone.utc) - dt).total_seconds()
    if overdue_s > _OVERDUE_GRACE_SECONDS:
        hours = overdue_s / 3600
        if hours >= 1:
            return f"next_run_at is {hours:.1f}h overdue — job is not firing (is the scheduler running?)"
        return f"next_run_at is {overdue_s / 60:.0f}m overdue — job is not firing (is the scheduler running?)"
    return None


def _cron_doctor_issues_for_job(job: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    last_status = str(job.get("last_status") or "").strip().lower()
    # "delivery_failed" means the agent run itself succeeded, so it is not a
    # failed last run — the dedicated delivery issue below reports it (and
    # last_error is None, which would render as "unknown error" here).
    if last_status and last_status not in {"ok", "delivery_failed"}:
        err = str(job.get("last_error") or "unknown error").strip()
        issues.append(f"last run failed: {err}")

    delivery_err = str(job.get("last_delivery_error") or "").strip()
    if delivery_err:
        issues.append(f"last delivery failed: {delivery_err}")

    unverified = job.get("last_delivery_unverified")
    if unverified:
        targets = ", ".join(str(t) for t in unverified) if isinstance(unverified, list) else str(unverified)
        issues.append(f"last delivery unverified (adapter acked without evidence): {targets}")

    if job.get("enabled", True) and job.get("state") not in {"paused", "completed"}:
        next_run = str(job.get("next_run_at") or "").strip()
        if not next_run:
            issues.append("active job has no next_run_at")
        else:
            overdue = _next_run_overdue_issue(next_run)
            if overdue:
                issues.append(overdue)

    script = str(job.get("script") or "").strip()
    if job.get("no_agent") and not script:
        issues.append("no-agent job has no script")
    if script:
        script_issue = _script_health_issue(script)
        if script_issue:
            issues.append(script_issue)

    workdir = str(job.get("workdir") or "").strip()
    if workdir and not Path(workdir).expanduser().exists():
        issues.append(f"workdir not found: {workdir}")

    return issues


def cron_doctor() -> int:
    """Run read-only cron health checks and return a shell-friendly status."""
    from cron.jobs import list_jobs

    jobs = list_jobs(include_disabled=False)
    findings: List[tuple[Dict[str, Any], List[str]]] = []
    for job in jobs:
        issues = _cron_doctor_issues_for_job(job)
        if issues:
            findings.append((job, issues))

    if not findings:
        print(color("✓ Cron doctor found no issues", Colors.GREEN))
        if jobs:
            print(color(f"  Checked {len(jobs)} active job(s).", Colors.DIM))
        else:
            print(color("  No active jobs configured.", Colors.DIM))
        return 0

    issue_count = sum(len(issues) for _, issues in findings)
    print(color(f"Cron doctor found {issue_count} issue(s) across {len(findings)} job(s):", Colors.YELLOW))
    print()
    for job, issues in findings:
        job_id = job.get("id", "?")
        name = job.get("name", "(unnamed)")
        print(f"  {color(job_id, Colors.YELLOW)} {name}")
        for issue in issues:
            print(f"    - {issue}")
    print()
    print(color("Next: fix the listed job config, then run `hermes cron doctor` again.", Colors.DIM))
    return 1


def cron_create(args):
    # The gateway-lifecycle guard lives in cron.jobs.create_job so it fires on
    # every job-creation path (this CLI subcommand AND the agent's `cronjob`
    # model tool, which calls create_job directly). When it blocks, create_job
    # raises GatewayLifecycleBlocked, the `cronjob` tool wrapper catches it and
    # returns it as result["error"], and the `if not result.get("success")`
    # branch below prints it in red and exits 1 — same UX as before.
    result = _cron_api(
        action="create",
        schedule=args.schedule,
        prompt=args.prompt,
        name=getattr(args, "name", None),
        deliver=getattr(args, "deliver", None),
        failure_deliver=getattr(args, "failure_deliver", None),
        repeat=getattr(args, "repeat", None),
        skill=getattr(args, "skill", None),
        skills=_normalize_skills(getattr(args, "skill", None), getattr(args, "skills", None)),
        script=getattr(args, "script", None),
        workdir=getattr(args, "workdir", None),
        model=getattr(args, "model", None),
        provider=getattr(args, "model_provider", None),
        no_agent=getattr(args, "no_agent", False) or None,
        monitor_script=getattr(args, "monitor_script", None),
        monitor_url=getattr(args, "monitor_url", None),
        continuity=getattr(args, "continuity", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
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
    if job_data.get("monitor_script"):
        print(f"  Monitor: {job_data['monitor_script']} (agent runs only on output change)")
    if job_data.get("monitor_url"):
        print(f"  Monitor: {job_data['monitor_url']} (agent runs only on output change)")
    if job_data.get("no_agent"):
        print("  Mode: no-agent (script stdout delivered directly)")
    if job_data.get("continuity"):
        print("  Continuity: on (each run sees the previous run's output)")
    if job_data.get("workdir"):
        print(f"  Workdir: {job_data['workdir']}")
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

    result = _cron_api(
        action="update",
        job_id=args.job_id,
        schedule=getattr(args, "schedule", None),
        prompt=getattr(args, "prompt", None),
        name=getattr(args, "name", None),
        deliver=getattr(args, "deliver", None),
        failure_deliver=getattr(args, "failure_deliver", None),
        repeat=getattr(args, "repeat", None),
        skills=final_skills,
        script=getattr(args, "script", None),
        workdir=getattr(args, "workdir", None),
        model=getattr(args, "model", None),
        provider=getattr(args, "model_provider", None),
        no_agent=getattr(args, "no_agent", None),
        monitor_script=getattr(args, "monitor_script", None),
        monitor_url=getattr(args, "monitor_url", None),
        continuity=getattr(args, "continuity", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
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
    if updated.get("monitor_script"):
        print(f"  Monitor: {updated['monitor_script']} (agent runs only on output change)")
    if updated.get("monitor_url"):
        print(f"  Monitor: {updated['monitor_url']} (agent runs only on output change)")
    if updated.get("no_agent"):
        print("  Mode: no-agent (script stdout delivered directly)")
    if updated.get("continuity"):
        print("  Continuity: on (each run sees the previous run's output)")
    if updated.get("workdir"):
        print(f"  Workdir: {updated['workdir']}")
    return 0


def _job_action(action: str, job_id: str, success_verb: str) -> int:
    _stateless_reset = None
    if action == "run":
        # One-shot CLI: this process exits as soon as the command returns, so
        # a background-dispatched run (daemon thread of THIS process) would be
        # orphaned mid-LLM-call — the delegation dies 'unknown' and the job's
        # execution row is stuck 'claimed', blocking future runs (#86721).
        # The background path in ``_try_dispatch_background_run`` triggers when
        # the CLI inherits a gateway/desktop session env (HERMES_SESSION_KEY);
        # declare the channel stateless so ``async_delivery_supported()`` gates
        # it off and the run executes synchronously to completion instead.
        # The declaration is scoped to this call (token reset in ``finally``)
        # so in-process callers (tests, embedding apps) are not tainted.
        try:
            from gateway.session_context import _SESSION_ASYNC_DELIVERY

            _stateless_token = _SESSION_ASYNC_DELIVERY.set(False)

            def _stateless_reset() -> None:
                _SESSION_ASYNC_DELIVERY.reset(_stateless_token)
        except Exception:
            _stateless_reset = None
    try:
        result = _cron_api(action=action, job_id=job_id)
    finally:
        if _stateless_reset is not None:
            _stateless_reset()
    if not result.get("success"):
        print(color(f"Failed to {action} job: {result.get('error', 'unknown error')}", Colors.RED))
        return 1
    job = result.get("job") or result.get("removed_job") or {}
    print(color(f"{success_verb} job: {job.get('name', job_id)} ({job_id})", Colors.GREEN))
    if action in {"resume", "run"} and result.get("job", {}).get("next_run_at"):
        print(f"  Next run: {result['job']['next_run_at']}")
    if action == "run":
        job = result.get("job", {})
        # A manual run can be dispatched to the gateway daemon's background
        # delegation worker instead of executing inline (e.g. when the CLI
        # process inherits a gateway/desktop session env and the run
        # resolves a session key). Such responses carry
        # execution_mode="background" and/or a delegation_id, and the job
        # keeps running AFTER this CLI process exits — a terminal
        # success/failure verdict would be a lie (#83340). Report the
        # background dispatch instead of claiming the run failed.
        delegation_id = job.get("delegation_id")
        if job.get("execution_mode") == "background" or delegation_id:
            if delegation_id:
                print(f"  Running in background (delegation {delegation_id}).")
            else:
                print("  Running in background.")
        elif job.get("executed"):
            outcome = "succeeded" if job.get("execution_success") else "failed"
            print(f"  Ran now: {outcome}.")
        elif job.get("execution_skipped"):
            print(f"  {job['execution_skipped']}")
        else:
            print("  It will run on the next scheduler tick.")
    return 0


def cron_resume(args) -> int:
    """Resume a paused job or explicitly re-arm a completed one-shot."""
    if bool(getattr(args, "run_at", None)) == bool(getattr(args, "run_now", False)):
        if getattr(args, "run_at", None) or getattr(args, "run_now", False):
            print(color("Use exactly one of --at or --run-now.", Colors.RED))
            return 1
        return _job_action("resume", args.job_id, "Resumed")
    from cron.jobs import AmbiguousJobReference, _hermes_now, rearm_oneshot

    run_at = _hermes_now().isoformat() if args.run_now else args.run_at
    try:
        job = rearm_oneshot(args.job_id, run_at)
    except (AmbiguousJobReference, ValueError) as exc:
        print(color(f"Failed to re-arm job: {exc}", Colors.RED))
        return 1
    if not job:
        print(color(f"Job not found: {args.job_id}", Colors.RED))
        return 1
    print(color(f"Re-armed job: {job.get('name', args.job_id)} ({args.job_id})", Colors.GREEN))
    print(f"  Next run: {job.get('next_run_at')}")
    return 0


def cron_notepad(args) -> int:
    """Handle ``hermes cron notepad <job_id> [get|set|delete|list]``.

    The per-job durable KV scratchpad (``cron/notepad.py``). This CLI is the
    write path — a running cron agent updates its own notepad by invoking
    these commands via its terminal tool; the scheduler injects non-empty
    notepads into the job prompt on each run.
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
        if action == "set":
            if key is None or value is None:
                print(color("Usage: hermes cron notepad <job_id> set <key> <value>", Colors.RED))
                return 1
            notepad.set_note(job_id, key, value)
            print(color(f"Set notepad key '{key}' for job {job_id}.", Colors.GREEN))
            return 0

        if action == "get":
            if key is None:
                print(color("Usage: hermes cron notepad <job_id> get <key>", Colors.RED))
                return 1
            stored = notepad.get_note(job_id, key)
            if stored is None:
                print(color(f"No notepad key '{key}' for job {job_id}.", Colors.YELLOW))
                return 1
            print(stored)
            return 0

        if action == "delete":
            if key is None:
                print(color("Usage: hermes cron notepad <job_id> delete <key>", Colors.RED))
                return 1
            if notepad.delete_note(job_id, key):
                print(color(f"Deleted notepad key '{key}' for job {job_id}.", Colors.GREEN))
                return 0
            print(color(f"No notepad key '{key}' for job {job_id}.", Colors.YELLOW))
            return 1

        # list (default)
        notes = notepad.list_notes(job_id)
        if not notes:
            print(color(f"Notepad for job {job_id} is empty.", Colors.DIM))
            return 0
        for note in notes:
            print(f"  {color(note['key'], Colors.YELLOW)} = {note['value']}")
            print(f"    {color('updated: ' + str(note['updated_at']), Colors.DIM)}")
        return 0
    except ValueError as exc:
        print(color(f"Notepad error: {exc}", Colors.RED))
        return 1


def cron_command(args):
    """Handle cron subcommands."""
    subcmd = getattr(args, 'cron_command', None)

    if subcmd is None or subcmd == "list":
        show_all = getattr(args, 'all', False)
        cron_list(show_all)
        return 0

    if subcmd == "status":
        cron_status()
        return 0

    if subcmd == "doctor":
        return cron_doctor()

    if subcmd == "tick":
        return cron_tick()

    if subcmd in {"runs", "history"}:
        cron_runs(getattr(args, "job_id", None), getattr(args, "limit", 20))
        return 0

    if subcmd == "incidents":
        return cron_incidents(args)

    if subcmd == "notepad":
        return cron_notepad(args)

    if subcmd in {"create", "add"}:
        return cron_create(args)

    if subcmd == "edit":
        return cron_edit(args)

    if subcmd == "pause":
        return _job_action("pause", args.job_id, "Paused")

    if subcmd == "resume":
        return cron_resume(args)

    if subcmd == "run":
        return _job_action("run", args.job_id, "Triggered")

    if subcmd in {"remove", "rm", "delete"}:
        return _job_action("remove", args.job_id, "Removed")

    print(f"Unknown cron command: {subcmd}")
    print("Usage: hermes cron [list|create|edit|pause|resume|run|remove|status|runs|doctor|tick]")
    sys.exit(1)
