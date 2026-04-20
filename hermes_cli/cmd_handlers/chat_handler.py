"""`hermes chat` subcommand handler.

F-C1 step 7 — final and highest-coupling subcommand extracted out
of ``hermes_cli/main.py``. The chat handler orchestrates several
boot-time concerns before delegating to the actual CLI loop in
``cli.main``:

1. Resolve ``--continue`` / ``-c`` into a concrete ``--resume``
   session ID (either by most-recent CLI session lookup or by
   title/ID match).
2. Resolve ``--resume`` by title if it isn't already a direct
   session ID.
3. First-run guard — if no provider is configured, prompt for
   ``hermes setup`` (with a non-interactive-stdin fallback that
   prints guidance and exits).
4. Kick off a background update check and a bundled-skills sync.
5. Apply the ``--yolo`` approval bypass and ``--source`` session
   tag to the environment.
6. Build kwargs from argparse namespace and hand off to
   ``cli.main``.

Several helpers remain in ``main.py`` because they are shared with
other subcommands (``_resolve_session_by_name_or_id``,
``_resolve_last_cli_session``, ``_has_any_provider_configured``).
The extracted handler imports them lazily — F-C1 leaves shared
helpers where they are and only relocates handler bodies.
"""

from __future__ import annotations

import os
import sys


def cmd_chat(args):
    """Run interactive chat CLI."""
    from hermes_cli.main import (
        _has_any_provider_configured,
        _resolve_last_cli_session,
        _resolve_session_by_name_or_id,
    )

    # Resolve --continue into --resume with the latest CLI session or by name
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            # -c "session name" — resolve by title or ID
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            else:
                print(f"No session found matching '{continue_val}'.")
                print("Use 'hermes sessions list' to see available sessions.")
                sys.exit(1)
        else:
            # -c with no argument — continue the most recent session
            last_id = _resolve_last_cli_session()
            if last_id:
                args.resume = last_id
            else:
                print("No previous CLI session found to continue.")
                sys.exit(1)

    # Resolve --resume by title if it's not a direct session ID
    resume_val = getattr(args, "resume", None)
    if resume_val:
        resolved = _resolve_session_by_name_or_id(resume_val)
        if resolved:
            args.resume = resolved
        # If resolution fails, keep the original value — _init_agent will
        # report "Session not found" with the original input

    # First-run guard: check if any provider is configured before launching
    if not _has_any_provider_configured():
        print()
        print("It looks like Hermes isn't configured yet -- no API keys or providers found.")
        print()
        print("  Run:  hermes setup")
        print()

        from hermes_cli.setup import is_interactive_stdin, print_noninteractive_setup_guidance

        if not is_interactive_stdin():
            print_noninteractive_setup_guidance(
                "No interactive TTY detected for the first-run setup prompt."
            )
            sys.exit(1)

        try:
            reply = input("Run setup now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply in ("", "y", "yes"):
            from hermes_cli.cmd_handlers.setup_handler import cmd_setup

            cmd_setup(args)
            return
        print()
        print("You can run 'hermes setup' at any time to configure.")
        sys.exit(1)

    # Start update check in background (runs while other init happens)
    try:
        from hermes_cli.banner import prefetch_update_check
        prefetch_update_check()
    except Exception:
        pass

    # Sync bundled skills on every CLI launch (fast -- skips unchanged skills)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # --yolo: bypass all dangerous command approvals
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # --source: tag session source for filtering (e.g. 'tool' for third-party integrations)
    if getattr(args, "source", None):
        os.environ["HERMES_SESSION_SOURCE"] = args.source

    # Import and run the CLI
    from cli import main as cli_main

    # Build kwargs from args
    kwargs = {
        "model": args.model,
        "provider": getattr(args, "provider", None),
        "toolsets": args.toolsets,
        "skills": getattr(args, "skills", None),
        "verbose": args.verbose,
        "quiet": getattr(args, "quiet", False),
        "query": args.query,
        "resume": getattr(args, "resume", None),
        "worktree": getattr(args, "worktree", False),
        "checkpoints": getattr(args, "checkpoints", False),
        "pass_session_id": getattr(args, "pass_session_id", False),
        "max_turns": getattr(args, "max_turns", None),
        "council": getattr(args, "council", None),
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        cli_main(**kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
