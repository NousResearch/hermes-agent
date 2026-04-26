"""``hermes copilot`` CLI subcommand — launch and list Copilot sessions.

Simplified interface: launch routes a prompt to a repo, spawns copilot
with ``--remote``, captures the session ID, and logs it.  Sessions are
cloud-managed — use ``copilot --connect=<session_id>`` from any
authenticated terminal to attach, and ``copilot --resume=<session_id>``
to resume a completed session.
"""

import sys
import time
import uuid
from datetime import datetime, timezone

from agent.redact import redact_sensitive_text
from hermes_state import SessionDB


def _get_db() -> SessionDB:
    """Get a SessionDB instance using the standard Hermes home."""
    return SessionDB()


def _connect_handle(job: dict) -> str:
    """Return the best Copilot reconnect handle for connect/resume.

    Prefers the dedicated ``connect_handle`` column (populated by the
    launcher) and falls back to the job ``id``. ``signal_ref`` is
    intentionally *not* consulted here — it stores caller metadata
    (e.g. a Jira ticket ID) and would otherwise produce invalid
    ``copilot --connect=<ticket-id>`` output.
    """
    return job.get("connect_handle") or job["id"]


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


def _utc_time(ts) -> str:
    """Format a timestamp as a UTC ISO-8601 string."""
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _state_badge(state: str) -> str:
    """Return a colored state indicator."""
    badges = {
        "running": "🟢 running",
        "done": "✅ done",
        "failed": "🔴 failed",
    }
    return badges.get(state, state)


def copilot_launch(args):
    """Launch a new Copilot session for a repo.

    Routes the prompt to a repo (or uses --repo), spawns copilot with
    --remote, captures the session ID from early output, and returns
    immediately while copilot continues working in the background.
    """
    prompt = getattr(args, "prompt", None) or ""
    repo = getattr(args, "repo", None)
    repo_path = getattr(args, "repo_path", None)
    model = getattr(args, "model", None)

    # If no explicit repo, try the router
    if not repo:
        if not prompt:
            print("Error: --repo or a prompt is required.", file=sys.stderr)
            sys.exit(1)
        from copilot_remote.router import route_repo
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
    job_id = str(uuid.uuid4())

    # Create job record
    db.create_copilot_remote(
        job_id=job_id,
        repo_slug=repo,
        repo_path=repo_path,
        prompt=prompt or None,
        signal_source=getattr(args, "signal_source", None) or "cli",
        signal_ref=getattr(args, "signal_ref", None),
    )

    print(f"Launching copilot remote: {job_id}")
    print(f"  Repo: {repo}")
    if prompt:
        preview = prompt[:80] + ("..." if len(prompt) > 80 else "")
        print(f"  Prompt: {preview}")

    # Completion callback — only used for dry_run and test (_spawn) paths.
    # Real launches use complete_job.py for DB updates.
    def _on_complete(session_id, exit_code):
        try:
            state = "done" if exit_code == 0 else "failed"
            finish_db = _get_db()
            finish_db.finish_copilot_remote(
                job_id,
                state=state,
                exit_code=exit_code,
            )
            finish_db.close()
        except Exception:
            pass  # Best-effort — don't crash the daemon thread

    # Launch copilot
    from copilot_remote.launcher import launch_copilot
    from copilot_remote.models import RepoEntry as _RE

    repo_entry = _RE(slug=repo, path=repo_path)
    try:
        result = launch_copilot(
            repo_entry, prompt,
            session_id=job_id,
            model=model,
            dry_run=getattr(args, "dry_run", False),
            on_complete=_on_complete,
        )
    except Exception as exc:
        # Redact before persisting/printing — exception text from subprocess /
        # HTTP layers can include embedded tokens, headers, or other secrets
        # that would otherwise leak into state.db, downstream `show`/`list`,
        # or uncaught CLI stderr output. Also strip CR/LF to avoid console
        # log-injection via attacker-controlled exception text. The original
        # exception class is preserved so callers can still distinguish error
        # types, but its message is replaced with the redacted/sanitized text.
        from copilot_remote.router import _sanitize_for_log

        redacted = _sanitize_for_log(redact_sensitive_text(str(exc)))
        db.finish_copilot_remote(job_id, state="failed", error_text=redacted)
        db.close()
        raise exc.__class__(redacted).with_traceback(exc.__traceback__) from None

    connect_handle = result.get("connect_id") or job_id
    if connect_handle != job_id:
        db.update_copilot_remote_connect_handle(job_id, connect_handle)

    prompt_delivery_warning = result.get("prompt_delivery_warning")

    # For dry-run, the process already completed synchronously.
    if getattr(args, "dry_run", False):
        print(f"  State: {_state_badge('done')}")
    else:
        print(f"  State: 🟢 running")

    if prompt_delivery_warning:
        print(f"  Warning: {prompt_delivery_warning}", file=sys.stderr)

    print(f"\n  Connect: copilot --connect={connect_handle}")
    print(f"  Resume:  copilot --resume={connect_handle}")

    db.close()


def copilot_list(args):
    """List copilot remote jobs."""
    state = getattr(args, "state", None)
    limit = getattr(args, "limit", 20)

    db = _get_db()
    try:
        jobs = db.list_copilot_remote(state=state, limit=limit)
        if not jobs:
            print("No copilot remote jobs found.")
            return

        fmt = "{:<38s} {:<20s} {:<12s} {:<21s}"
        print(fmt.format("ID", "REPO", "STATE", "CREATED (UTC)"))
        print("-" * 95)
        for job in jobs:
            print(fmt.format(
                job["id"][:38],
                (job["repo_slug"] or "")[:20],
                _state_badge(job["state"])[:12],
                _utc_time(job["created_at"]),
            ))
    finally:
        db.close()


def copilot_show(args):
    """Show details of a copilot remote."""
    job_id = args.job_id

    db = _get_db()
    try:
        job = db.get_copilot_remote(job_id)
        if not job:
            print(f"Error: Job not found: {job_id}", file=sys.stderr)
            sys.exit(1)

        print(f"Job:      {job['id']}")
        print(f"State:    {_state_badge(job['state'])}")
        print(f"Repo:     {job['repo_slug']}")
        print(f"Path:     {job['repo_path']}")
        print(f"Created:  {_relative_time(job['created_at'])}")

        if job.get("prompt"):
            preview = job["prompt"][:120] + ("..." if len(job["prompt"]) > 120 else "")
            print(f"Prompt:   {preview}")

        sid = _connect_handle(job)
        print(f"Connect:  copilot --connect={sid}")
        print(f"Resume:   copilot --resume={sid}")

        if job.get("exit_code") is not None:
            print(f"Exit:     {job['exit_code']}")
        if job.get("error_text"):
            print(f"Error:    {job['error_text']}")
        if job.get("signal_source"):
            ref = f" ({job['signal_ref']})" if job.get("signal_ref") else ""
            print(f"Signal:   {job['signal_source']}{ref}")

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
        "show": copilot_show,
    }

    handler = handlers.get(subcmd)
    if handler:
        handler(args)
    else:
        print(f"Unknown copilot command: {subcmd}")
        print("Usage: hermes copilot [launch|list|show]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Slash command handler (interactive session: /copilot_remote ...)
# ---------------------------------------------------------------------------

def handle_copilot_remote_slash(raw_command: str) -> None:
    """Handle /copilot_remote slash command from an interactive Hermes session.

    Parses the raw command text and dispatches to the appropriate handler.
    """
    from types import SimpleNamespace

    parts = raw_command.strip().split()
    subcmd = parts[1] if len(parts) > 1 else "list"
    args_rest = parts[2:]

    try:
        if subcmd == "list":
            ns = SimpleNamespace(state=None, limit=20)
            for i, a in enumerate(args_rest):
                if a == "--state" and i + 1 < len(args_rest):
                    ns.state = args_rest[i + 1]
            copilot_list(ns)

        elif subcmd == "launch":
            ns = SimpleNamespace(
                prompt="", repo=None, repo_path=None, model=None,
                dry_run=False,
                signal_source="slash", signal_ref=None,
            )
            i = 0
            prompt_parts = []
            while i < len(args_rest):
                if args_rest[i] == "--repo" and i + 1 < len(args_rest):
                    ns.repo = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--repo-path" and i + 1 < len(args_rest):
                    ns.repo_path = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--model" and i + 1 < len(args_rest):
                    ns.model = args_rest[i + 1]
                    i += 2
                elif args_rest[i] == "--dry-run":
                    ns.dry_run = True
                    i += 1
                else:
                    prompt_parts.append(args_rest[i])
                    i += 1
            ns.prompt = " ".join(prompt_parts)
            copilot_launch(ns)

        elif subcmd == "show" and args_rest:
            ns = SimpleNamespace(job_id=args_rest[0])
            copilot_show(ns)

        else:
            print("Usage: /copilot_remote [launch|list|show]")
            print()
            print("  /copilot_remote list                        List all jobs")
            print("  /copilot_remote launch <prompt>             Route prompt → repo, launch copilot")
            print("  /copilot_remote launch --model <m> <prompt> Use specific model")
            print("  /copilot_remote launch --repo <slug> <msg>  Launch for specific repo")
            print("  /copilot_remote show <job_id>               Show job details + connect command")

    except SystemExit:
        pass
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
