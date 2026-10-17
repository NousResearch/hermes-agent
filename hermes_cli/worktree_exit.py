"""The keep / archive / discard question when an interactive ``hermes -w`` session exits with a
dirty or unpushed worktree.

Asked once, in the foreground, after the UI released the terminal: by the classic CLI in
``_tui_shutdown`` (before ``_run_cleanup`` arms the exit watchdog) and by the TUI launcher after
the Ink child exits. The answer is applied right there, synchronously, before any exit watchdog
or teardown runs: a large archive is never cut short, and a failed one keeps the tree. Only a real interactive session
asks: ``-q``/``-z``/cron/gateway never reach these call sites, a missing TTY or a signal-driven
shutdown skips the question, and every skip means Keep (the sweep archives the tree later).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from agent.i18n import t

logger = logging.getLogger(__name__)

KEEP, ARCHIVE, DISCARD = "keep", "archive", "discard"


def _has_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def worktree_state(info: dict) -> Optional[str]:
    """``"dirty"``, ``"unpushed"`` or None (clean and pushed / gone)."""
    from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    path, repo_root = info.get("path", ""), info.get("repo_root", "")
    if not path or not os.path.isdir(path):
        return None
    if _worktree_is_dirty(path, repo_root, timeout=10):
        return "dirty"
    if _worktree_has_unpushed_commits(path, timeout=10):
        return "unpushed"
    return None


def _still_in_use(path: str, grace_s: float = 5.0) -> bool:
    """Whether a live process (other than this one) still works in *path* after a short grace:
    the TUI's own backend (``tui_gateway``) takes a moment to exit after the Ink UI closes."""
    import time

    from hermes_cli.worktree_sweep import live_process_pids
    deadline = time.monotonic() + grace_s
    while live_process_pids([path], exclude=(os.getpid(),)).get(path):
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.25)
    return False


def ask_exit_choice(info: Optional[dict]) -> str:
    """Ask what to do with a dirty/unpushed tree; records the answer in ``info["exit_choice"]``."""
    if not info or not _has_tty():
        return KEEP
    state = worktree_state(info)
    if state is None:
        return KEEP
    if _still_in_use(info["path"]):
        return KEEP  # something still runs in there; never offer to delete it
    from hermes_cli.curses_ui import curses_radiolist
    title = t(f"cli.worktree.exit_question_{state}", path=info["path"])
    items = [t("cli.worktree.exit_choice_keep"), t("cli.worktree.exit_choice_archive"),
             t("cli.worktree.exit_choice_discard")]
    try:  # Esc / Ctrl-C / a curses failure all resolve to Keep (index 0)
        idx = curses_radiolist(title, items, selected=0, cancel_returns=0)
    except Exception:
        logger.debug("worktree exit question failed; keeping the tree", exc_info=True)
        idx = None
    choice = {1: ARCHIVE, 2: DISCARD}.get(idx, KEEP) if isinstance(idx, int) else KEEP
    info["exit_choice"] = choice
    return choice


def apply_exit_choice(info: dict, echo) -> bool:
    """Carry out an Archive/Discard answer once -> True when the tree was handled (removed or a
    message printed). Keep (or no answer) -> False: the caller's default keep path runs."""
    choice = info.pop("exit_choice", None)
    if choice not in (ARCHIVE, DISCARD):
        return False
    path, repo_root, branch = info["path"], info["repo_root"], info.get("branch")
    if choice == ARCHIVE:
        import subprocess

        from hermes_cli.worktree_archive import ArchiveError
        from hermes_cli.worktree_sweep import archive_then_remove, sweep_settings
        echo(t("cli.worktree.archiving", path=path))
        try:
            ref, _kept, removed = archive_then_remove(
                repo_root, path, reason="session exit", unlock=True, exclude_pids=(os.getpid(),),
                max_untracked_mb=sweep_settings()["archive_max_file_mb"])
        except (ArchiveError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
            echo(f"\033[31m{t('cli.worktree.archive_failed', path=path, error=str(exc) or 'interrupted')}\033[0m")
            return True
        name = ref.rsplit("/", 1)[-1]
        echo(f"\033[32m{t('cli.worktree.archived', path=path, ref=ref)}\033[0m")
        echo(f"  {t('cli.worktree.restore_hint', name=name)}")
        if not removed:
            echo(f"  {t('cli.worktree.archive_not_removed', path=path)}")
        return True
    from hermes_cli._subprocess_compat import noninteractive_git_env
    from hermes_cli.worktree_ops import _git_quiet, release_lsp_clients
    release_lsp_clients(path)
    _git_quiet(["worktree", "unlock", path], repo_root, log="git worktree unlock failed (non-fatal)")
    _git_quiet(["worktree", "remove", path, "--force"], repo_root, timeout=60, log="Failed to remove worktree")
    if branch and not os.path.isdir(path):
        _git_quiet(["branch", "-D", branch], repo_root, log=f"Failed to delete branch {branch}",
                   env=noninteractive_git_env())
    echo(f"\033[33m{t('cli.worktree.discarded', path=path)}\033[0m")
    return True
