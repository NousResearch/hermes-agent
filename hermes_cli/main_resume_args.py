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
