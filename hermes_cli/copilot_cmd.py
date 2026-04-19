"""``hermes copilot`` CLI subcommand — manage Copilot remote sessions.

Provides launch, list, show, and takeover commands for the Copilot
remote session lifecycle managed through SessionDB.
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from hermes_state import SessionDB
from copilot_jobs.models import JobState, JobOwner


def _get_db() -> SessionDB:
    """Get a SessionDB instance using the standard Hermes home."""
    return SessionDB()


def _short_id() -> str:
    """Generate a short job ID like '20260419_153012_a1b2c3'."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"cj_{ts}_{suffix}"


def _relative_time(ts) -> str:
    """Format a timestamp as relative time."""
    if not ts:
        return "-"
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _state_badge(state: str) -> str:
    """Return a colored state indicator."""
    badges = {
        "pending": "⏳ pending",
        "running": "🟢 running",
        "idle": "💤 idle",
        "closed": "⬛ closed",
        "failed": "🔴 failed",
    }
    return badges.get(state, state)


def copilot_launch(args):
    """Launch a new Copilot remote session for a repo."""
    prompt = getattr(args, "prompt", None) or ""
    repo = getattr(args, "repo", None)
    repo_path = getattr(args, "repo_path", None)

    # If no explicit repo, try the router
    if not repo:
        if not prompt:
            print("Error: --repo or a prompt is required.", file=sys.stderr)
            sys.exit(1)
        from copilot_jobs.router import route_repo
        entry = route_repo(prompt)
        if not entry:
            print(
                "Error: Could not determine target repo from prompt. "
                "Use --repo to specify explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        repo = entry.slug
        repo_path = repo_path or entry.path
        print(f"Router selected: {repo} ({repo_path})")

    if not repo_path:
        print("Error: --repo-path is required when using --repo.", file=sys.stderr)
        sys.exit(1)

    db = _get_db()
    try:
        # Check for existing active job on this repo
        existing = db.find_active_copilot_job_for_repo(repo)
        if existing:
            print(
                f"Warning: Active job already exists for {repo}: "
                f"{existing['id']} ({existing['state']})",
                file=sys.stderr,
            )
            print(f"Use 'hermes copilot show {existing['id']}' to inspect it.")
            sys.exit(1)

        job_id = _short_id()
        signal_source = getattr(args, "signal_source", None) or "cli"
        signal_ref = getattr(args, "signal_ref", None)
        idle_ttl = getattr(args, "idle_ttl", 300)

        db.create_copilot_job(
            job_id=job_id,
            repo_slug=repo,
            repo_path=repo_path,
            prompt=prompt or None,
            signal_source=signal_source,
            signal_ref=signal_ref,
            idle_ttl_seconds=idle_ttl,
        )

        print(f"Created copilot job: {job_id}")
        print(f"  Repo:   {repo}")
        print(f"  Path:   {repo_path}")
        print(f"  State:  pending")
        print(f"  TTL:    {idle_ttl}s")

        if prompt:
            preview = prompt[:80] + ("..." if len(prompt) > 80 else "")
            print(f"  Prompt: {preview}")

        print()
        print("Next steps:")
        print(f"  1. Start Copilot: copilot --remote --cwd {repo_path}")
        print(f"  2. Update job:    hermes copilot activate {job_id} --session-id <id> --remote-name <name>")
        print(f"  3. Attach:        hermes copilot show {job_id}  (shows attach command)")

    finally:
        db.close()


def copilot_activate(args):
    """Activate a pending job by recording Copilot session details."""
    job_id = args.job_id
    session_id = getattr(args, "session_id", None)
    remote_name = getattr(args, "remote_name", None)
    pid = getattr(args, "pid", None)

    db = _get_db()
    try:
        job = db.get_copilot_job(job_id)
        if not job:
            print(f"Error: Job not found: {job_id}", file=sys.stderr)
            sys.exit(1)
        if job["state"] not in ("pending", "running"):
            print(
                f"Error: Job {job_id} is in state '{job['state']}', "
                f"cannot activate.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Build attach command
        container = os.environ.get("CONTAINER_NAME", "ryanwalden-ryanwalden")
        attach_cmd = None
        if session_id:
            attach_cmd = f"docker exec -it {container} copilot --connect={session_id}"

        db.update_copilot_job_remote(
            job_id=job_id,
            copilot_session_id=session_id,
            remote_name=remote_name,
            pid=pid,
            attach_command=attach_cmd,
        )

        if job["state"] == "pending":
            db.transition_copilot_job(job_id, "running", event_type="activated")

        print(f"Job {job_id} activated.")
        if attach_cmd:
            print(f"  Attach: {attach_cmd}")

    finally:
        db.close()


def copilot_list(args):
    """List copilot jobs."""
    state = getattr(args, "state", None)
    limit = getattr(args, "limit", 20)

    db = _get_db()
    try:
        jobs = db.list_copilot_jobs(state=state, limit=limit)
        if not jobs:
            print("No copilot jobs found.")
            return

        # Table header
        fmt = "{:<30s} {:<20s} {:<14s} {:<8s} {:<14s}"
        print(fmt.format("ID", "REPO", "STATE", "OWNER", "CREATED"))
        print("-" * 90)
        for job in jobs:
            print(fmt.format(
                job["id"][:30],
                (job["repo_slug"] or "")[:20],
                _state_badge(job["state"])[:14],
                job["owner"][:8],
                _relative_time(job["created_at"]),
            ))
    finally:
        db.close()


def copilot_show(args):
    """Show details of a copilot job."""
    job_id = args.job_id

    db = _get_db()
    try:
        job = db.get_copilot_job(job_id)
        if not job:
            print(f"Error: Job not found: {job_id}", file=sys.stderr)
            sys.exit(1)

        print(f"Job:            {job['id']}")
        print(f"State:          {_state_badge(job['state'])}")
        print(f"Owner:          {job['owner']}")
        print(f"Repo:           {job['repo_slug']}")
        print(f"Path:           {job['repo_path']}")
        print(f"Created:        {_relative_time(job['created_at'])}")
        print(f"Updated:        {_relative_time(job['updated_at'])}")

        if job.get("prompt"):
            preview = job["prompt"][:120] + ("..." if len(job["prompt"]) > 120 else "")
            print(f"Prompt:         {preview}")
        if job.get("copilot_session_id"):
            print(f"Session ID:     {job['copilot_session_id']}")
        if job.get("remote_name"):
            print(f"Remote Name:    {job['remote_name']}")
        if job.get("pid"):
            print(f"PID:            {job['pid']}")
        if job.get("attach_command"):
            print(f"Attach:         {job['attach_command']}")
        if job.get("idle_since"):
            print(f"Idle since:     {_relative_time(job['idle_since'])}")
        if job.get("idle_ttl_seconds"):
            print(f"Idle TTL:       {job['idle_ttl_seconds']}s")
        if job.get("error_text"):
            print(f"Error:          {job['error_text']}")
        if job.get("signal_source"):
            ref = f" ({job['signal_ref']})" if job.get("signal_ref") else ""
            print(f"Signal:         {job['signal_source']}{ref}")

        # Show recent events
        events = db.get_copilot_job_events(job_id, limit=10)
        if events:
            print("\nRecent events:")
            for ev in events:
                ts = _relative_time(ev["created_at"])
                transition = ""
                if ev.get("from_state") and ev.get("to_state"):
                    transition = f" ({ev['from_state']} → {ev['to_state']})"
                elif ev.get("to_state"):
                    transition = f" (→ {ev['to_state']})"
                print(f"  {ts}  {ev['event_type']}{transition}")

    finally:
        db.close()


def copilot_takeover(args):
    """Transfer ownership of a job from hermes to human."""
    job_id = args.job_id

    db = _get_db()
    try:
        job = db.take_over_copilot_job(job_id)
        print(f"Job {job_id} transferred to human.")
        print(f"  State: {_state_badge(job['state'])}")
        print(f"  Owner: {job['owner']}")
        if job.get("attach_command"):
            print(f"  Attach: {job['attach_command']}\n")
        print("TTL reaping is now suppressed for this job.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


def copilot_idle(args):
    """Mark a running job as idle (Copilot process exited)."""
    job_id = args.job_id

    db = _get_db()
    try:
        job = db.mark_copilot_job_idle(job_id)
        print(f"Job {job_id} marked idle.")
        print(f"  Idle TTL: {job.get('idle_ttl_seconds', 300)}s")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


def copilot_close(args):
    """Close an idle or failed job."""
    job_id = args.job_id

    db = _get_db()
    try:
        job = db.close_copilot_job(job_id)
        print(f"Job {job_id} closed.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


def copilot_command(args):
    """Route copilot subcommands."""
    subcmd = getattr(args, "copilot_action", None)

    if subcmd is None or subcmd == "list":
        copilot_list(args)
        return

    handlers = {
        "launch": copilot_launch,
        "activate": copilot_activate,
        "show": copilot_show,
        "takeover": copilot_takeover,
        "idle": copilot_idle,
        "close": copilot_close,
    }

    handler = handlers.get(subcmd)
    if handler:
        handler(args)
    else:
        print(f"Unknown copilot command: {subcmd}")
        print("Usage: hermes copilot [launch|activate|list|show|takeover|idle|close]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Slash command handler (interactive session: /copilot ...)
# ---------------------------------------------------------------------------

def handle_copilot_slash(raw_command: str) -> None:
    """Handle /copilot slash command from an interactive Hermes session.

    Parses the raw command text and dispatches to the appropriate handler.
    Unlike the argparse CLI path, this uses a lightweight SimpleNamespace
    so it works without sys.exit() calls in the interactive loop.
    """
    from types import SimpleNamespace

    parts = raw_command.strip().split()
    # parts[0] is "/copilot", rest are subcommand + args
    subcmd = parts[1] if len(parts) > 1 else "list"
    args_rest = parts[2:]

    try:
        if subcmd == "list":
            ns = SimpleNamespace(state=None, limit=20)
            # Parse optional --state flag
            for i, a in enumerate(args_rest):
                if a == "--state" and i + 1 < len(args_rest):
                    ns.state = args_rest[i + 1]
            copilot_list(ns)

        elif subcmd == "launch":
            ns = SimpleNamespace(
                prompt=" ".join(args_rest) if args_rest else "",
                repo=None, repo_path=None,
                signal_source="slash", signal_ref=None, idle_ttl=300,
            )
            # Parse --repo and --repo-path flags from args
            i = 0
            prompt_parts = []
            while i < len(args_rest):
                if args_rest[i] == "--repo" and i + 1 < len(args_rest):
                    ns.repo = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--repo-path" and i + 1 < len(args_rest):
                    ns.repo_path = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--idle-ttl" and i + 1 < len(args_rest):
                    ns.idle_ttl = int(args_rest[i + 1])
                    i += 2
                else:
                    prompt_parts.append(args_rest[i])
                    i += 1
            ns.prompt = " ".join(prompt_parts)
            copilot_launch(ns)

        elif subcmd == "show" and args_rest:
            ns = SimpleNamespace(job_id=args_rest[0])
            copilot_show(ns)

        elif subcmd == "activate" and args_rest:
            ns = SimpleNamespace(
                job_id=args_rest[0],
                session_id=None, remote_name=None, pid=None,
            )
            i = 1
            while i < len(args_rest):
                if args_rest[i] == "--session-id" and i + 1 < len(args_rest):
                    ns.session_id = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--remote-name" and i + 1 < len(args_rest):
                    ns.remote_name = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--pid" and i + 1 < len(args_rest):
                    ns.pid = int(args_rest[i + 1])
                    i += 2
                else:
                    i += 1
            copilot_activate(ns)

        elif subcmd == "takeover" and args_rest:
            ns = SimpleNamespace(job_id=args_rest[0])
            copilot_takeover(ns)

        elif subcmd == "idle" and args_rest:
            ns = SimpleNamespace(job_id=args_rest[0])
            copilot_idle(ns)

        elif subcmd == "close" and args_rest:
            ns = SimpleNamespace(job_id=args_rest[0])
            copilot_close(ns)

        else:
            print("Usage: /copilot [launch|list|show|activate|takeover|idle|close]")
            print()
            print("  /copilot list                        List all jobs")
            print("  /copilot launch <prompt>             Route prompt to repo and create job")
            print("  /copilot launch --repo <slug> <msg>  Create job for specific repo")
            print("  /copilot show <job_id>               Show job details")
            print("  /copilot activate <id> --session-id <sid>  Record Copilot session")
            print("  /copilot takeover <job_id>           Transfer ownership to human")
            print("  /copilot idle <job_id>               Mark job as idle")
            print("  /copilot close <job_id>              Close a job")

    except SystemExit:
        # copilot_* functions call sys.exit() on errors — catch it
        # so the interactive session stays alive.
        pass
    except ValueError as exc:
        print(f"Error: {exc}")
