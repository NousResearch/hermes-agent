"""``hermes chat`` resume-argument steps split out of ``hermes_cli.main``: ``--resume @claude/@codex``
imports, and recreating a removed ``--tui -w`` worktree a resumed session needs (N8)."""

import os
import sys


def import_foreign_resume(args) -> None:
    """--resume @claude / @codex: import a foreign session and resume it."""
    _resume_foreign = getattr(args, "resume", None)
    if not (isinstance(_resume_foreign, str) and _resume_foreign.strip().lower() in ("@claude", "@codex")):
        return
    from hermes_cli.foreign_sessions import import_foreign_session, pick_foreign_session

    _picked = pick_foreign_session(_resume_foreign.strip().lower().lstrip("@"))
    if _picked is None:
        sys.exit(1)
    try:
        _imported_id = import_foreign_session(_picked.source, _picked.path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"✓ Imported as {_imported_id} — resuming it now.")
    print(f"  (later: hermes --resume {_imported_id})")
    args.resume = _imported_id


def _frozen_cwd(db, session_id) -> str:
    """The cwd the owner runs *session_id*'s tools in (its local policy), or "" without one. The
    row's cwd is main's stamp, which for a classic ``-w`` session is the repo root, not the tree."""
    import sqlite3

    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    try:
        return str(local_receipt(db, session_id)["policy"].get("cwd") or "").strip()
    except (RuntimeStoreError, KeyError, AttributeError, sqlite3.Error):  # no readable policy: use the row's cwd
        return ""


def recreate_resumed_tui_worktree(args) -> None:
    """A resumed session whose frozen cwd is a ``-w`` / ``--tui -w`` tree that quit removed gets that
    tree back before attaching, or the owner refuses every turn ``workspace_unavailable`` (N8)."""
    if not getattr(args, "resume", None) or getattr(args, "worktree", False):
        return
    cwd = ""
    from hermes_cli.main import _session_db
    with _session_db() as db:  # a store error is swallowed: nothing to recreate
        cwd = (_frozen_cwd(db, args.resume)
               or ((db.get_session(args.resume) or {}).get("cwd") or "").strip())
    if not cwd or os.path.isdir(cwd):
        return
    from hermes_cli.tui_worktree_resume import recreate_tui_worktree
    tree = recreate_tui_worktree(cwd, args.resume)
    if tree:
        print(f"↪ recreated the session's worktree: {tree}")


def resolve_interactive_resume(args, *, use_tui: bool) -> None:
    """Bare --resume / -r without argument: interactive picker (like claude --resume)."""
    _resume_raw = getattr(args, "resume", None)
    if _resume_raw is not True:
        return
    if use_tui:
        os.environ["HERMES_TUI_OPEN_SESSIONS"] = "1"  # health: allow HX002 -- TUI launch flag to open session picker overlay
        args.resume = None
        return
    if not sys.stdin.isatty():
        print("Error: interactive session picker requires a TTY.", file=sys.stderr)
        sys.exit(1)
    from hermes_cli.main import _session_db
    from hermes_cli.sessions_cmd_browse import pick_interactive_session
    with _session_db() as db:
        selected_id = pick_interactive_session(db, exclude_sources=["tool"])
        if selected_id:
            args.resume = selected_id
        else:
            print("Cancelled." if db else "No previous sessions found to resume.")
            sys.exit(0 if selected_id is None and db else 1)


def resolve_session_name_or_search(args, resume_val: str, *, use_tui: bool, resolve_fn) -> None:
    """Resolve session name/ID or fall back to interactive search on TTY."""
    resolved = resolve_fn(resume_val)
    if resolved:
        args.resume = resolved
    elif sys.stdin.isatty() and not use_tui:
        from hermes_cli.main import _session_db
        from hermes_cli.sessions_cmd_browse import pick_interactive_session
        with _session_db() as db:
            selected_id = pick_interactive_session(db, exclude_sources=["tool"], initial_search=resume_val)
            if selected_id:
                args.resume = selected_id
            else:
                print("Cancelled.")
                sys.exit(0)
    else:
        args.resume = resume_val


def _create_titled_session(title: str) -> str | None:
    """Create a fresh titled session (``chat -c <title> --create-if-missing``)."""
    import logging
    try:
        from hermes_cli.gateway_session_store import create_titled_session
        return create_titled_session(title)
    except Exception:
        logging.getLogger(__name__).exception("Failed to create titled session %r", title)
        return None


def _resolve_continue_arg(args, *, use_tui: bool) -> None:
    """Resolve ``-c/--continue`` into ``args.resume``."""
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            from hermes_cli.main import _resolve_session_by_name_or_id
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            elif getattr(args, "create_if_missing", False):
                new_sid = _create_titled_session(continue_val) if use_tui else None
                if not new_sid:
                    from hermes_cli.gateway_session_store import create_titled_session
                    new_sid = create_titled_session(continue_val)
                if new_sid:
                    args.resume = new_sid
                else:
                    print(f"No session found matching '{continue_val}' and a new titled session could not be created.", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"No session found matching '{continue_val}'.", file=sys.stderr)
                print("Use 'hermes sessions list' to see available sessions, or pass --create-if-missing to start a new session with that title.", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                from hermes_cli.terminal_breadcrumbs import resolve_breadcrumb_session
                _crumb_id = resolve_breadcrumb_session()
            except Exception:
                _crumb_id = None
            if _crumb_id:
                args.resume = _crumb_id
            else:
                from hermes_cli.main import _latest_session_id
                last_id = _latest_session_id(use_tui)
                if last_id:
                    args.resume = last_id
                else:
                    kind = "TUI" if use_tui else "CLI"
                    print(f"No previous {kind} session to continue. Start a new one with `hermes`, or list sessions with `hermes sessions list`.", file=sys.stderr)
                    sys.exit(1)
