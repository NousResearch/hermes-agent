"""``hermes copilot`` CLI subcommand — launch and list Copilot sessions.

Simplified interface: launch routes a prompt to a repo, spawns copilot
with ``--remote``, captures the Hermes job/session ID, and logs it.
Sessions are cloud-managed — when a dedicated Copilot reconnect handle
is available, use ``copilot --connect=<connect_handle>`` from any
authenticated terminal to attach, and
``copilot --resume=<connect_handle>`` to resume a completed session.
The Hermes job UUID is *not* a valid Copilot connect/resume handle; if
the handle could not be extracted, ``hermes copilot show <job_id>`` and
the launcher log under ``~/.hermes/logs/copilot-<job_id>.log`` are the
correct places to look.
"""

import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from agent.redact import redact_sensitive_text
from hermes_state import SessionDB


def _get_db() -> SessionDB:
    """Get a SessionDB instance using the standard Hermes home."""
    return SessionDB()


def _connect_handle(job: dict) -> Optional[str]:
    """Return the Copilot reconnect handle for connect/resume, or ``None``.

    Reads only the dedicated ``connect_handle`` column populated by the
    launcher. When that handle was not extracted (Copilot CLI changed its
    output, the verification HTTP probe failed, etc.) this returns
    ``None`` so callers can surface an explicit "connect handle
    unavailable" message rather than fabricating an invalid reconnect
    command from the Hermes job UUID. The launcher does not pass the
    Hermes ``job_id`` into Copilot via ``--resume``, so the job UUID is
    not a usable reconnect handle. ``signal_ref`` is intentionally *not*
    consulted either — it stores caller metadata (e.g. a Jira ticket
    ID) and would otherwise produce invalid
    ``copilot --connect=<ticket-id>`` output.
    """
    handle = job.get("connect_handle")
    return handle if handle else None


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

    connect_handle = result.get("connect_id")
    if connect_handle:
        db.update_copilot_remote_connect_handle(job_id, connect_handle)

    prompt_delivery_warning = result.get("prompt_delivery_warning")

    # For dry-run, the process already completed synchronously.
    if getattr(args, "dry_run", False):
        print(f"  State: {_state_badge('done')}")
    else:
        print(f"  State: 🟢 running")

    if prompt_delivery_warning:
        print(f"  Warning: {prompt_delivery_warning}", file=sys.stderr)

    if connect_handle:
        print(f"\n  Connect: copilot --connect={connect_handle}")
        print(f"  Resume:  copilot --resume={connect_handle}")
    else:
        # The launcher could not extract Copilot's remote task ID. Do not
        # fabricate a reconnect command from the Hermes job UUID — it is
        # not a valid `--connect/--resume` handle. Direct the user to the
        # launcher log and `hermes copilot show` so they can recover the
        # handle once Copilot writes it.
        print(
            "\n  Connect handle unavailable — Hermes could not extract "
            "Copilot's remote task ID.\n"
            f"  Inspect ~/.hermes/logs/copilot-{job_id}.log and re-run "
            f"`hermes copilot show {job_id}` once the handle is recorded."
        )

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
        if sid:
            print(f"Connect:  copilot --connect={sid}")
            print(f"Resume:   copilot --resume={sid}")
        else:
            print(
                "Connect:  unavailable — Hermes did not extract a Copilot "
                "reconnect handle for this job.\n"
                f"          Inspect ~/.hermes/logs/copilot-{job['id']}.log "
                "to recover it."
            )

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
    Uses ``shlex.split`` so quoted prompts (and any path argument containing
    spaces) are preserved as a single token instead of being shattered on
    whitespace.
    """
    import shlex
    from types import SimpleNamespace

    try:
        parts = shlex.split(raw_command.strip())
    except ValueError as exc:
        # Unbalanced quote / malformed escape — surface a clear error
        # instead of silently parsing a partial command.
        print(f"Error: could not parse /copilot_remote command: {exc}", file=sys.stderr)
        return
    subcmd = parts[1] if len(parts) > 1 else "list"
    args_rest = parts[2:]

    try:
        if subcmd == "list":
            ns = SimpleNamespace(state=None, limit=20)
            i = 0
            while i < len(args_rest):
                a = args_rest[i]
                if a == "--state" and i + 1 < len(args_rest):
                    ns.state = args_rest[i + 1]
                    i += 2
                elif a == "--limit" and i + 1 < len(args_rest):
                    raw = args_rest[i + 1]
                    try:
                        parsed = int(raw)
                    except ValueError:
                        print(
                            f"Error: --limit requires an integer (got {raw!r})",
                            file=sys.stderr,
                        )
                        return
                    # Clamp to a sane range so a bogus value cannot ask the
                    # DB for an unbounded slice or a non-positive count.
                    ns.limit = max(1, min(parsed, 1000))
                    i += 2
                else:
                    i += 1
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
