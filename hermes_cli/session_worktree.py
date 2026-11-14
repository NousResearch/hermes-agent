"""The ``-w`` worktree of a chat session, for any launcher: the Ink TUI's and a plugin's own chat
client. ``create`` before the client starts, ``restore`` when a session is resumed, ``finish``
after the client exits."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def create() -> dict | None:
    """Make the worktree for a chat launched from the current directory: a fresh tree under
    ``<repo>/.worktrees/`` on its own ``hermes/`` branch, locked so the stale-tree sweep keeps it.

    -> ``{path, branch, repo_root, base}``, or None when it could not be made (the reason was
    printed). Pass ``path`` as the session's cwd.
    """
    wt_info = None
    try:
        from cli import _git_repo_root, _maintain_pack_health, _prune_stale_worktrees, _setup_worktree
        repo = _git_repo_root()
        if repo:
            _prune_stale_worktrees(repo)
            # Repack on pack sprawl so `worktree add` never crawls on a
            # multi-agent box; on a thread so it can't block launch.
            import threading as _threading

            _threading.Thread(
                target=_maintain_pack_health, args=(repo,), name="pack-maintenance", daemon=True).start()
        from hermes_cli.config import get_hermes_home
        # The session's frozen cwd keeps this checkout; name the store from creation so a
        # launcher killed with its terminal still leaves the tree to its sessions.
        wt_info = _setup_worktree(retain_db=get_hermes_home() / "state.db")
    except Exception as exc:  # health: allow BLE001 -- every failure is told to the user and the launch stops
        print(f"✗ Failed to create the session worktree: {exc}", file=sys.stderr)
    return wt_info


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


def restore(session_id: str) -> str | None:
    """Bring back the worktree a resumed session needs: when the session's cwd is a ``-w`` tree
    Hermes recorded and a quit removed, recreate it at the same path, on its old branch when that
    survived. Call it before the client attaches, or every turn is refused ``workspace_unavailable``.

    -> its path, or None when the session's directory exists or is not a tree Hermes recorded.
    """
    cwd = ""
    from hermes_cli.main import _session_db
    with _session_db() as db:  # a store error is swallowed: nothing to recreate
        cwd = (_frozen_cwd(db, session_id)
               or ((db.get_session(session_id) or {}).get("cwd") or "").strip())
    if not cwd or os.path.isdir(cwd):
        return None
    from hermes_cli.tui_worktree_resume import recreate_tui_worktree
    return recreate_tui_worktree(cwd, session_id)


def _remove_idle_clean_tui_worktree(wt_info: dict) -> bool:
    """Remove the ``--tui -w`` checkout on quit when nothing still needs it (N8) -> True if gone.

    Kept when the gateway has queued or started work for a session inside it (quitting only
    detaches the viewer), still schedules work for one (an active loop, heartbeat or goal
    continuation, or a cron job bound to it), another viewer still holds one, or another process
    still works in it; and by ``_cleanup_worktree`` when it is dirty, holds unpushed commits or
    something no branch keeps.
    """
    from hermes_cli.config import get_hermes_home
    from hermes_cli.worktree_exit import _still_in_use
    from hermes_cli.worktree_ops import _retained_tree_is_live, _tree_still_served

    path = wt_info.get("path")
    if not path or not Path(path).is_dir():
        return False
    try:
        db_path = str(get_hermes_home() / "state.db")
        if _retained_tree_is_live(db_path, path, pending_only=True):
            return False
        if _still_in_use(path):
            return False
        # After the grace above: a legacy sidecar stamps its rows ``tui_shutdown`` as it exits.
        if _tree_still_served(db_path, path):
            return False
        from cli import _cleanup_worktree
        _cleanup_worktree(wt_info)
    except Exception:
        logger.debug("TUI worktree cleanup failed; keeping %s", path, exc_info=True)
        return False
    return not Path(path).exists()


def finish(info: dict, session_id: str | None, *, ask: bool) -> None:
    """What a launcher does with the worktree after its client exits and released the terminal.

    With *ask* (a clean exit), a dirty or unpushed tree gets the keep / archive / discard question
    on the terminal. A tree nothing still needs is removed with its branch. Otherwise it is kept,
    handed to *session_id* (when set) so the sweep keeps it while that session can resume, and its
    path is printed to stderr.
    """
    if ask:
        from hermes_cli.worktree_exit import ask_exit_choice
        ask_exit_choice(info)
    from hermes_cli.worktree_exit import apply_exit_choice
    if apply_exit_choice(info, print):
        return  # archived or discarded at the user's request
    if _remove_idle_clean_tui_worktree(info):
        return  # clean and idle: removed with its branch, as before the shared gateway
    # Quitting detaches a viewer; the gateway may still be executing in
    # this checkout and future resumes retain its frozen cwd.
    if session_id:
        # The launch lock names this (exiting) pid; hand it to the session so a later
        # launch's stale prune keeps the tree while the session is resumable.
        from hermes_cli.config import get_hermes_home
        from hermes_cli.worktree_ops import _retain_worktree_for_session
        _retain_worktree_for_session(info["repo_root"], info["path"], session_id,
                                     get_hermes_home() / "state.db")
    print(f"Worktree retained for this session: {info['path']}", file=sys.stderr)
