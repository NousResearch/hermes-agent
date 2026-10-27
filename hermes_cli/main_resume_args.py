"""``hermes chat`` resume-argument steps split out of ``hermes_cli.main``: ``--resume @claude/@codex``
imports, bare ``--resume``'s workspace picker, and recreating a removed ``--tui -w`` worktree a
resumed session needs (N8)."""

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

    from hermes_cli.main import _resolve_workspace_key, _session_db
    from hermes_cli.sessions_cmd import _default_exclude, _scope_label, _surface_reader
    from hermes_cli.sessions_cmd_browse import _session_browse_picker

    key = _resolve_workspace_key()
    with _session_db() as db:  # kept open: the browser tags status and can delete
        sessions = _surface_reader(db).list_sessions_rich(
            source=getattr(args, "source", None), exclude_sources=_default_exclude(args),
            limit=500, workspace_key=key,
        )
        if not sessions:
            print(f"No sessions in {_scope_label(key)}." if key else "No sessions found.")
            print("Run 'hermes' to start one, or 'hermes sessions browse' to pick from every session.")
            sys.exit(1)
        chosen = _session_browse_picker(sessions, session_db=db)
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
            _saved_cwd = ((db.get_session(args.resume) or {}).get("cwd") or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")


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
