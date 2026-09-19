"""Installed-plugin checkouts: ``git pull`` with autostash, and post-pull bytecode hygiene.

Sibling of :mod:`hermes_cli.plugins_cmd` (the installer core, enable/disable state and console helpers
live there and are imported late — this module is imported BY ``plugins_cmd``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import noninteractive_git_env


def _clear_plugin_bytecode(target: Path) -> int:
    """Remove ``__pycache__`` dirs under a just-updated plugin checkout. Plugin dirs sit outside
    the repo, so the launch-time bytecode sweep never covers them and stale bytecode after a pull
    can ImportError in the next process. Never raises.

    See #60242, #6207.
    """
    removed = 0
    try:
        for cache_dir in target.rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed += 0 if cache_dir.exists() else 1
    except OSError:
        pass
    return removed


def _run_plugin_git(
    git_exe: str, target: Path, *args: str, timeout: int = 60, auth_url: str = "",
) -> subprocess.CompletedProcess:
    """Run one git command inside a plugin checkout (non-interactive). *auth_url* names the remote
    a network verb talks to; it runs anonymously first and a stored user credential for that host
    is attached only when the remote refuses anonymous access (private repos)."""
    from hermes_cli.git_credentials import run_git_with_credential_fallback
    return run_git_with_credential_fallback(
        [git_exe, *args], auth_url, env=noninteractive_git_env(), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=timeout, cwd=str(target))


def _stash_ref(git_exe: str, target: Path) -> str:
    """Current ``refs/stash`` commit, or empty string when no stash exists."""
    probe = _run_plugin_git(git_exe, target, "rev-parse", "--verify", "refs/stash")
    return probe.stdout.strip() if probe.returncode == 0 else ""


def _reapply_stash(git_exe: str, target: Path) -> bool:
    """``stash apply`` the autostash; drop it on a clean apply. False when it applied with
    errors or left unmerged paths (the stash entry is kept in that case)."""
    restore = _run_plugin_git(git_exe, target, "stash", "apply", "stash@{0}")
    unmerged = _run_plugin_git(git_exe, target, "diff", "--name-only", "--diff-filter=U")
    if restore.returncode != 0 or unmerged.stdout.strip():
        return False
    _run_plugin_git(git_exe, target, "stash", "drop", "stash@{0}")
    return True


def _autostash_dirty_tree(git_exe: str, target: Path) -> tuple[bool, str]:
    """Stash local edits before a pull. Returns ``(stash_created, error)``; a non-empty error means
    the tree is dirty but nothing was saved, so the pull must not run."""
    from hermes_cli.plugins_cmd import _safe_git_error
    status = _run_plugin_git(git_exe, target, "status", "--porcelain")
    if status.returncode != 0 or not status.stdout.strip():
        return False, ""
    pre_stash = _stash_ref(git_exe, target)
    push = _run_plugin_git(
        git_exe, target, "stash", "push", "--include-untracked", "-m", "hermes-plugin-update-autostash")
    post_stash = _stash_ref(git_exe, target)
    if not post_stash or post_stash == pre_stash:
        err = _safe_git_error(push)
        return False, (
            "Local changes in the plugin checkout could not be "
            "stashed; update aborted before touching the checkout."
            + (f"\n{err}" if err else ""))
    if push.returncode != 0:
        # Saved-but-couldn't-clean (undeletable untracked files): the stash entry is complete;
        # reset tracked mods so the pull isn't blocked by a still-dirty tree.
        _run_plugin_git(git_exe, target, "reset", "--hard", "HEAD")
    return True, ""


def _git_pull_plugin_dir(target: Path) -> tuple[bool, str]:
    """``git pull --ff-only`` a plugin checkout, autostashing local edits (users patch installed
    plugins in place, and a plain ff-only pull would then refuse forever).

    Users tweak installed plugins in place (config constants, small patches), and a plain ``pull --ff-only``
    then aborts with "Your local changes ... would be overwritten by merge" — making the plugin permanently
    un-updatable until they hand-run git. Same UX class Factory Droid fixed in v0.188 ("Updating a plugin
    marketplace now succeeds when its checkout has local changes"), and the same autostash approach ``hermes
    update`` already uses for the main checkout (PR #70161).
    """
    from hermes_cli.plugins_cmd import _resolve_git_executable, _safe_git_error
    git_exe = _resolve_git_executable()
    if not git_exe:
        return False, "git is not installed or not in PATH."
    try:
        stash_created, err = _autostash_dirty_tree(git_exe, target)
        if err:
            return False, err
        origin = _run_plugin_git(git_exe, target, "remote", "get-url", "origin", timeout=15)
        result = _run_plugin_git(git_exe, target, "pull", "--ff-only", auth_url=origin.stdout.strip())
        if result.returncode != 0:
            err = _safe_git_error(result) or "git pull failed."
            if not stash_created:
                return False, err
            # Put the user's edits back before reporting the failure.
            if _reapply_stash(git_exe, target):
                note = "Local changes were restored."
            else:
                note = "Local changes are preserved in git stash (restore with: git stash pop)."
            return False, f"{err}\n{note}"

        pulled = result.stdout.strip()
        if not stash_created:
            return True, pulled
        if _reapply_stash(git_exe, target):
            return True, pulled + "\nLocal changes were re-applied on top of the update."

        # Conflicted re-apply: leave the plugin importable on the updated
        # revision; the user's edits stay safe in the stash entry.
        _run_plugin_git(git_exe, target, "reset", "--hard", "HEAD")
        return True, pulled + (
            "\n⚠ Local changes in this plugin conflicted with the update and "
            "were NOT re-applied. They are preserved in git stash — inspect "
            "with `git stash show -p stash@{0}` and re-apply with "
            f"`git stash pop` inside {target}.")
    except FileNotFoundError:
        return False, "git is not installed or not in PATH."
    except subprocess.TimeoutExpired:
        return False, "Git operation timed out after 60 seconds."
