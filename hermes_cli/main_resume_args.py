"""``hermes chat`` session-argument steps split out of ``hermes_cli.main``: ``--in DIR``, ``--resume
@claude/@codex`` imports, and recreating a removed ``--tui -w`` worktree a resumed session needs (N8)."""

import os
import sys


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


def _frozen_cwd(db, session_id) -> str:
    """The cwd the owner runs *session_id*'s tools in (its local policy), or "" without one. The
    row's cwd is main's stamp, which for a classic ``-w`` session is the repo root, not the tree."""
    from persistence.core.errors import PersistenceError
    try:
        return str(db.persistence.bindings.receipt(session_id).policy.get("cwd") or "").strip()
    except (PersistenceError, AttributeError):  # no readable policy: use the row's cwd
        return ""


def recreate_resumed_tui_worktree(args) -> None:
    """A resumed session whose frozen cwd is a ``-w`` / ``--tui -w`` tree that quit removed gets that
    tree back before attaching, or the owner refuses every turn ``workspace_unavailable`` (N8)."""
    if not getattr(args, "resume", None) or getattr(args, "worktree", False):
        return
    cwd = ""
    from hermes_cli.main import _session_db
    with _session_db() as db:  # a store error is swallowed: nothing to recreate
        cwd = _frozen_cwd(db, args.resume)
        if not cwd:
            record = db.persistence.sessions.get(args.resume)
            cwd = ((record.cwd if record else None) or "").strip()
    if not cwd or os.path.isdir(cwd):
        return
    from hermes_cli.tui_worktree_resume import recreate_tui_worktree
    tree = recreate_tui_worktree(cwd, args.resume)
    if tree:
        print(f"↪ recreated the session's worktree: {tree}")
