"""``hermes chat`` session-argument steps split out of ``hermes_cli.main``: ``--in DIR``, ``--resume
@claude/@codex`` imports, bare ``--resume``'s workspace picker, and recreating a removed ``--tui -w``
worktree a resumed session needs (N8)."""

import os
import sys

_BARE_RESUME_NEEDS_ID = (
    "--resume needs a session ID here (no terminal to pick in).",
    "Pass --resume latest, -c, or an ID — 'hermes sessions list' shows them.",
)


def is_bare_resume(args) -> bool:
    """``--resume``/``-r`` given with no value (``nargs="?"`` stores ``True``; ``""`` counts too)."""
    raw = getattr(args, "resume", None)
    return raw is True or (isinstance(raw, str) and not raw.strip())


def _can_prompt(args) -> bool:
    """An interactive launch is one that may block on a keyboard: not -q/-z, and a real terminal."""
    if getattr(args, "oneshot", None) or getattr(args, "oneshot_exit", False) or getattr(args, "query", None):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):  # closed/redirected std streams (embedded callers)
        return False


def resolve_bare_resume(args, *, interactive: bool = True) -> None:
    """Bare ``--resume``: pick a session from this workspace instead of demanding an ID.

    Scoped exactly like ``-c`` / ``--resume latest`` (``_resolve_workspace_key``), so the picker
    never shows another folder's history. Runs after ``--in DIR`` has chdir'd, and never in a
    non-interactive launch: a one-shot has no keyboard, and prompting would hang it.
    """
    if not is_bare_resume(args):
        return
    if not interactive or not _can_prompt(args):
        print(_BARE_RESUME_NEEDS_ID[0], file=sys.stderr)
        print(_BARE_RESUME_NEEDS_ID[1], file=sys.stderr)
        sys.exit(2)

    import argparse

    from hermes_cli.main import _resolve_workspace_key
    from hermes_cli.sessions_cmd import _listing_scope, _scope_label, _surface_reader
    from hermes_cli.sessions_cmd_browse import _session_browse_picker
    from hermes_constants import get_hermes_home
    from persistence_assembly import open_persistence, profile_scope

    key = _resolve_workspace_key()
    # A writer, as `sessions browse` opens: the picker tags status and its 'd' deletes. The rows are
    # the browse list's (human threads, by last message), narrowed to this workspace in SQL.
    with open_persistence(profile_scope(home=get_hermes_home()), eager=True) as store:
        sessions = _surface_reader(store).list_sessions_rich(
            limit=500, workspace_key=key, **_listing_scope(argparse.Namespace()))
        if not sessions:
            print(f"No sessions in {_scope_label(key)}." if key else "No sessions found.")
            print("Run 'hermes' to start one, or 'hermes sessions browse' to pick from every session.")
            sys.exit(1)
        chosen = _session_browse_picker(sessions, store=store)
    if not chosen:
        print("Cancelled.")
        sys.exit(1)
    args.resume = chosen


def resolve_chat_session_args(args, use_tui: bool) -> None:
    """Normalize --in / --resume / --continue on ``args`` before agent init.

    Order matters: ``--in DIR`` chdirs first so workspace-scoped "latest"/-c
    lookups key off DIR (and pins the session there, skipping cwd restore);
    then bare ``--resume`` → a pick from DIR's workspace, ``--resume latest`` → MRU id,
    ``--continue`` → ``--resume``, ``--resume @claude/@codex`` → imported session id,
    title → id; finally cd back into a resumed session's recorded cwd (best-effort,
    opt-out via --no-restore-cwd, skipped under --worktree).

    main's helpers are imported here, at call time, so a patch on ``hermes_cli.main``
    (``_latest_session_id``, ``_session_db``, ...) still intercepts them.
    """
    from hermes_cli.main import (
        _apply_in_dir, _latest_session_id, _resolve_continue_arg,
        _resolve_session_by_name_or_id, _session_db,
    )

    _apply_in_dir(args)

    # Bare `--resume`/`-r`: no ID given, so pick one from this workspace (after --in, so the
    # picker scopes to DIR). Non-interactive launches refuse it instead of blocking.
    resolve_bare_resume(args)

    # --resume latest: same resolution as bare `-c`. The keyword wins over a
    # session literally titled "latest" (still reachable by ID or `-c latest`).
    _resume_raw = getattr(args, "resume", None)
    if isinstance(_resume_raw, str) and _resume_raw.strip().lower() == "latest":
        _last_id = _latest_session_id(use_tui)
        if _last_id:
            args.resume = _last_id
        else:
            kind = "TUI" if use_tui else "CLI"
            print(f"No previous {kind} session found to resume.")
            print("Use 'hermes sessions list' to see available sessions.")
            sys.exit(1)

    _resolve_continue_arg(args, use_tui=use_tui)

    import_foreign_resume(args)

    resume_val = getattr(args, "resume", None)
    if resume_val:
        # On miss keep the original so _init_agent reports "Session not found" with it.
        args.resume = _resolve_session_by_name_or_id(resume_val) or resume_val

    recreate_resumed_tui_worktree(args)

    # cd back into a resumed session's recorded cwd (opt out: --no-restore-cwd;
    # --worktree owns its own dir). A missing dir warns and stays put.
    if (
        getattr(args, "resume", None)
        and not getattr(args, "no_restore_cwd", False)
        and not getattr(args, "worktree", False)
    ):
        with _session_db() as db:  # never let cwd-restore break a resume
            session = db.persistence.sessions.get(args.resume)
            _saved_cwd = ((session.cwd if session else None) or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")


def apply_in_dir(args) -> None:
    """--in DIR: chdir first so workspace-scoped lookups key off DIR; pins the session there."""
    in_dir = getattr(args, "in_dir", None)
    if not in_dir:
        return
    # Git Bash / MSYS hands us POSIX-style paths (`--in ~` → `/c/Users/x`);
    # translate drive-root spellings to native Windows form. No-op elsewhere.
    from tools.environments.local import _msys_to_windows_path

    _target_dir = os.path.abspath(os.path.expanduser(_msys_to_windows_path(in_dir)))
    if not os.path.isdir(_target_dir):
        print(f"Error: --in directory not found: {in_dir}")
        sys.exit(1)
    try:
        os.chdir(_target_dir)
    except OSError as e:
        print(f"Error: cannot enter --in directory {in_dir}: {e}")
        sys.exit(1)
    # Every cwd consumer (resolve_agent_cwd -> Codex app-server thread cwd, the
    # terminal tool, context-file discovery) prefers TERMINAL_CWD over the process
    # cwd, so a value inherited from a parent surface, the shell or .env outlives
    # this chdir and re-homes the session in the old directory (#106220). Refresh
    # it. An unset variable stays unset: the backends then derive from the new
    # process cwd (local exports it at cli import, docker mounts it, ssh and
    # container backends keep their own remote/sandbox default).
    if os.environ.get("TERMINAL_CWD", "").strip():
        os.environ["TERMINAL_CWD"] = _target_dir
    args.no_restore_cwd = True


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


def recreate_resumed_tui_worktree(args) -> None:
    """A resumed session whose frozen cwd is a ``-w`` / ``--tui -w`` tree that quit removed gets that
    tree back before attaching, or the owner refuses every turn ``workspace_unavailable`` (N8)."""
    if not getattr(args, "resume", None) or getattr(args, "worktree", False):
        return
    from hermes_cli.session_worktree import restore
    tree = restore(args.resume)
    if tree:
        print(f"↪ recreated the session's worktree: {tree}")
