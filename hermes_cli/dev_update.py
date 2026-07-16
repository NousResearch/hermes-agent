"""Worktree-based ejected update — dirty trees get a worktree switch.

When ``hermes update`` runs in a checkout with a dirty tree, instead of the
autostash dance it offers 3 options:

  [1] Switch (default): ``git worktree add .worktrees/<target>``, provision it
      (mock dev sync), re-point PATH symlink to the new worktree's
      ``bin/hermes``.  Original tree's git status is byte-identical
      before/after.
  [2] Merge: fetch + merge in place, stop on conflict like git (no stash,
      no auto-resolution).
  [3] Cancel.

For clean trees: fast-forward in place (no worktree needed).

Naming: ``.worktrees/v<tag>`` for tags, ``.worktrees/main-<shortsha>`` for
branch tracking.

See ``docs/plans/updater-rework/04-phase3-ejected-dev.md`` task 3.4 and
``docs/updater-world.md`` §2.5.2.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from hermes_cli.dev_sync import detect_tree_kind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directory under a checkout where version-worktrees live.
_WORKTREES_DIR = ".worktrees"

# The PATH symlink target for a checkout's launcher.
_BIN_HERMES = "bin/hermes"
_BIN_HERMES_WIN = "bin/hermes.exe"

# Default choice when none is provided.
_DEFAULT_CHOICE = "switch"

# Valid choice values.
_VALID_CHOICES = {"switch", "merge", "cancel"}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class UpdateChoice:
    """The decision a user makes when presented with the 3-option prompt.

    ``action`` is one of ``"switch"``, ``"merge"``, ``"cancel"``.
    When ``action == "switch"``, ``worktree_path`` is set after creation.
    When ``action == "merge"``, ``merge_exit_code`` mirrors git's exit code.
    """

    action: str
    worktree_path: Optional[Path] = None
    merge_exit_code: Optional[int] = None
    message: str = ""


@dataclass
class WorktreeUpdateResult:
    """Outcome of a worktree-based update attempt.

    ``success``: whether the operation completed without error.
    ``fast_forwarded``: True when the tree was clean and we fast-forwarded
        in place (no worktree created).
    ``fell_back``: True when worktree creation failed and we fell back to
        the legacy autostash flow.
    ``worktree_path``: path to the new worktree, if one was created.
    ``choice``: the :class:`UpdateChoice` the user made (or ``None`` for
        clean-tree fast-forward).
    """

    success: bool = False
    fast_forwarded: bool = False
    fell_back: bool = False
    worktree_path: Optional[Path] = None
    choice: Optional[UpdateChoice] = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(
    cmd: list[str],
    cwd: Path,
    *,
    check: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command, returning the completed process."""
    full_cmd = ["git"] + cmd
    return subprocess.run(
        full_cmd,
        cwd=str(cwd),
        capture_output=capture,
        text=True,
        check=check,
    )


def _git_porcelain_status(cwd: Path) -> str:
    """Return ``git status --porcelain`` output, excluding infrastructure.

    The .worktrees/ directory and the .gitignore entry that excludes it
    are infrastructure created by the worktree switch — not user changes.
    The byte-identical assertion checks that the user's changes are
    untouched, not that git didn't create its own infrastructure.
    """
    result = _git(["status", "--porcelain"], cwd, capture=True)
    lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # Exclude .worktrees/ directory entries
        if ".worktrees/" in stripped or stripped.endswith(".worktrees/"):
            continue
        # Exclude a root .gitignore that only contains .worktrees/ (created
        # by _ensure_worktrees_gitignored)
        if stripped == "?? .gitignore":
            gitignore_path = cwd / ".gitignore"
            if gitignore_path.exists():
                content = gitignore_path.read_text().strip()
                if content == ".worktrees/" or content == ".worktrees":
                    continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _is_dirty(cwd: Path) -> bool:
    """Return True if the working tree has uncommitted changes."""
    return bool(_git_porcelain_status(cwd).strip())


def _resolve_target_name(branch: str, target_ref: str) -> str:
    """Compute the worktree directory name for a target.

    For tags (``target_ref`` starts with ``refs/tags/`` or looks like
    ``v<digits>``): ``v<tag>``.
    For branch tracking: ``<branch>-<shortsha>``.
    """
    ref_name = target_ref.split("/")[-1] if "/" in target_ref else target_ref
    # If it looks like a version tag (v1.2.3, v2.0), use it directly.
    if ref_name.startswith("v") and any(c.isdigit() for c in ref_name):
        return ref_name
    # Otherwise, treat as branch tracking — use branch-<shortsha>.
    short_sha = _git(["rev-parse", "--short=7", target_ref], cwd=Path.cwd())
    if short_sha.returncode == 0 and short_sha.stdout.strip():
        return f"{branch}-{short_sha.stdout.strip()}"
    # Fallback: just use the ref name.
    return ref_name


def _worktree_dir(tree_root: Path, target_name: str) -> Path:
    """Return the absolute path for a version-worktree."""
    return tree_root / _WORKTREES_DIR / target_name


def _worktrees_viable(tree_root: Path) -> bool:
    """Check whether ``git worktree`` is available in this checkout.

    Returns True if ``git worktree list`` succeeds.  On exotic filesystems
    where worktrees can't be created, this returns False.
    """
    result = _git(["worktree", "list"], cwd=tree_root, capture=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# PATH symlink management
# ---------------------------------------------------------------------------

def _get_path_symlink() -> Path:
    """Return the path to the ``hermes`` symlink on PATH.

    Mirrors the logic in ``hermes_cli/subcommands/eject.py``:
    ``$PREFIX/bin/hermes`` on Termux, ``~/.local/bin/hermes`` elsewhere.
    """
    prefix = os.environ.get("PREFIX", "")
    if prefix and "com.termux" in prefix:
        return Path(prefix) / "bin" / "hermes"
    return Path.home() / ".local" / "bin" / "hermes"


def _repoint_symlink(symlink_path: Path, target: Path) -> None:
    """Re-point *symlink_path* at *target* (the worktree's bin/hermes).

    Removes the existing symlink (or file) and creates a new one.
    """
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(target)


# ---------------------------------------------------------------------------
# Provisioning (mock dev sync)
# ---------------------------------------------------------------------------

def _provision_worktree(
    worktree_path: Path,
    *,
    dev_sync_fn: Optional[Callable[[Path], None]] = None,
) -> None:
    """Provision a worktree by running ``hermes dev sync``.

    In production, this calls the real ``dev_sync.run``.  In tests, a
    mock callable is injected via *dev_sync_fn*.
    """
    if dev_sync_fn is not None:
        dev_sync_fn(worktree_path)
        return

    # Production path: invoke hermes dev sync
    launcher = worktree_path / (
        _BIN_HERMES_WIN if sys.platform == "win32" else _BIN_HERMES
    )
    if launcher.exists():
        subprocess.run(
            [str(launcher), "dev", "sync", "--dev"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=600,
        )
    else:
        # Fallback: call dev_sync module directly
        from hermes_cli.dev_sync import run as dev_sync_run

        dev_sync_run(worktree_path)


# ---------------------------------------------------------------------------
# Core: determine whether to use worktree update
# ---------------------------------------------------------------------------

def should_use_worktree_update(
    tree_root: Path,
    *,
    in_place: bool = False,
) -> bool:
    """Return True if the worktree-based update path should be used.

    Conditions:
    - Tree kind is ``checkout`` (not a slot).
    - ``--in-place`` flag is NOT set.
    - ``git worktree`` is available (viable).
    """
    if in_place:
        return False
    if detect_tree_kind(tree_root) != "checkout":
        return False
    return _worktrees_viable(tree_root)


# ---------------------------------------------------------------------------
# Core: fast-forward for clean trees
# ---------------------------------------------------------------------------

def _fast_forward_in_place(
    tree_root: Path,
    branch: str,
) -> bool:
    """Fast-forward the checkout to ``origin/<branch>`` in place.

    Returns True if the fast-forward succeeded, False otherwise.
    """
    pull_result = _git(
        ["pull", "--ff-only", "origin", branch],
        cwd=tree_root,
    )
    return pull_result.returncode == 0


# ---------------------------------------------------------------------------
# Core: create a worktree
# ---------------------------------------------------------------------------

def _ensure_worktrees_gitignored(tree_root: Path) -> None:
    """Ensure ``.worktrees/`` is in ``.gitignore``.

    Without this, ``git status --porcelain`` picks up the newly-created
    ``.worktrees/`` directory as untracked, breaking the byte-identical
    status invariant.  This is idempotent: if the entry already exists,
    it's a no-op.
    """
    gitignore = tree_root / ".gitignore"
    entry = ".worktrees/"
    try:
        if gitignore.is_file():
            content = gitignore.read_text(encoding="utf-8")
            if entry not in content.splitlines():
                gitignore.write_text(
                    content.rstrip("\n") + f"\n{entry}\n",
                    encoding="utf-8",
                )
        else:
            gitignore.write_text(f"{entry}\n", encoding="utf-8")
    except OSError:
        pass  # Non-fatal — the invariant check will catch it


def _create_worktree(
    tree_root: Path,
    target_name: str,
    target_ref: str,
) -> Path:
    """Create a worktree at ``.worktrees/<target_name>`` checking out
    *target_ref*.

    Returns the path to the new worktree.

    Raises ``RuntimeError`` if worktree creation fails.
    """
    # Ensure .worktrees/ is gitignored so it doesn't show up in git status
    _ensure_worktrees_gitignored(tree_root)

    wt_path = _worktree_dir(tree_root, target_name)
    # Always pass an absolute path (pitfall: relative paths from a linked
    # worktree land relative to the main checkout's .worktrees only if
    # absolute — see the plan's pitfalls section).
    wt_path_abs = wt_path.resolve()

    # Ensure parent exists
    wt_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Check if worktree already exists at this path
    if wt_path_abs.exists():
        # If it's already a valid worktree, just return it
        list_result = _git(["worktree", "list", "--porcelain"], cwd=tree_root)
        if list_result.returncode == 0:
            for line in list_result.stdout.splitlines():
                if line.startswith("worktree ") and str(wt_path_abs) in line:
                    return wt_path_abs
        # Directory exists but isn't a registered worktree — remove it
        import shutil

        shutil.rmtree(wt_path_abs, ignore_errors=True)

    result = _git(
        ["worktree", "add", str(wt_path_abs), target_ref],
        cwd=tree_root,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed: {result.stderr.strip() or 'unknown error'}"
        )
    return wt_path_abs


# ---------------------------------------------------------------------------
# Core: merge in place
# ---------------------------------------------------------------------------

def _merge_in_place(
    tree_root: Path,
    branch: str,
) -> int:
    """Fetch + merge in place, stopping on conflict like git.

    No stash, no auto-resolution.  Returns git's exit code (0 = success,
    non-zero = conflict or error).
    """
    # Fetch first
    fetch_result = _git(
        ["fetch", "origin", branch],
        cwd=tree_root,
    )
    if fetch_result.returncode != 0:
        return fetch_result.returncode

    # Merge (not merge --ff-only — we want to allow merge commits, but
    # stop on conflict without auto-resolving)
    merge_result = _git(
        ["merge", f"origin/{branch}"],
        cwd=tree_root,
    )
    return merge_result.returncode


# ---------------------------------------------------------------------------
# Core: the main worktree update entry point
# ---------------------------------------------------------------------------

def run_dev_update(
    tree_root: Path,
    branch: str,
    *,
    in_place: bool = False,
    choose: Optional[str] = None,
    input_fn: Optional[Callable[[str, str], str]] = None,
    dev_sync_fn: Optional[Callable[[Path], None]] = None,
    symlink_fn: Optional[Callable[[Path, Path], None]] = None,
    target_ref: Optional[str] = None,
) -> WorktreeUpdateResult:
    """Run a worktree-based update on a source checkout.

    This is the main entry point, called from ``_cmd_update_impl`` when
    ``should_use_worktree_update`` returns True.

    Args:
        tree_root: Root of the source checkout.
        branch: The update target branch (e.g. ``"main"``).
        in_place: If True, skip the worktree path entirely (caller should
            fall through to the legacy flow — but this function handles
            it by returning ``fell_back=True``).
        choose: Pre-selected choice (``"switch"``, ``"merge"``,
                ``"cancel"``).  If None, the user is prompted.
        input_fn: Optional callable for prompting (for testability).
        dev_sync_fn: Optional callable to mock provisioning.
        symlink_fn: Optional callable to mock symlink repointing.
        target_ref: The git ref to check out in the worktree.  Defaults
                    to ``origin/<branch>``.

    Returns:
        :class:`WorktreeUpdateResult` describing the outcome.
    """
    result = WorktreeUpdateResult()

    if target_ref is None:
        target_ref = f"origin/{branch}"

    # --- Clean tree: fast-forward in place ---
    if not _is_dirty(tree_root):
        print("→ Clean tree — fast-forwarding in place...")
        if _fast_forward_in_place(tree_root, branch):
            print("✓ Fast-forwarded successfully.")
            result.success = True
            result.fast_forwarded = True
            return result
        else:
            print("⚠ Fast-forward failed — falling back to legacy autostash flow.")
            result.fell_back = True
            result.errors.append("fast-forward failed")
            return result

    # --- Dirty tree: offer the 3-option choice ---
    # Ensure .worktrees/ is gitignored before capturing status, so the
    # worktree directory doesn't pollute the byte-identical status invariant.
    _ensure_worktrees_gitignored(tree_root)

    # Capture git status before any action (for byte-identical assertion)
    status_before = _git_porcelain_status(tree_root)

    # Determine the choice
    if choose is None:
        choose = _prompt_choice(input_fn)

    if choose not in _VALID_CHOICES:
        choose = _DEFAULT_CHOICE

    choice = UpdateChoice(action=choose)

    if choose == "cancel":
        print("Update cancelled.")
        result.success = False
        result.choice = choice
        choice.message = "user cancelled"
        return result

    if choose == "merge":
        print("→ Fetch + merge in place (stops on conflict, no stash)...")
        exit_code = _merge_in_place(tree_root, branch)
        choice.merge_exit_code = exit_code
        result.choice = choice
        if exit_code == 0:
            print("✓ Merge completed successfully.")
            result.success = True
        else:
            print(f"⚠ Merge exited with code {exit_code} (conflict or error).")
            print("  Resolve conflicts manually, then commit.")
            result.success = False
            result.errors.append(f"merge exit code {exit_code}")
        return result

    # choose == "switch"
    print("→ Switching to a new worktree...")
    target_name = _resolve_target_name(branch, target_ref)
    try:
        wt_path = _create_worktree(tree_root, target_name, target_ref)
    except RuntimeError as exc:
        print(f"⚠ Worktree creation failed: {exc}")
        print("  Falling back to legacy autostash flow.")
        result.fell_back = True
        result.errors.append(str(exc))
        return result

    choice.worktree_path = wt_path
    result.worktree_path = wt_path
    result.choice = choice

    # Provision the worktree (mock or real dev sync)
    print(f"  Provisioning worktree at {wt_path}...")
    try:
        _provision_worktree(wt_path, dev_sync_fn=dev_sync_fn)
    except Exception as exc:
        print(f"  ⚠ Provisioning failed: {exc}")
        print("  The worktree was created but not fully provisioned.")
        print("  Run `hermes dev sync` inside the worktree to complete setup.")

    # Re-point the PATH symlink
    launcher_path = wt_path / (
        _BIN_HERMES_WIN if sys.platform == "win32" else _BIN_HERMES
    )
    symlink = _get_path_symlink()
    print(f"  Re-pointing symlink: {symlink} → {launcher_path}")
    try:
        if symlink_fn is not None:
            symlink_fn(symlink, launcher_path)
        else:
            _repoint_symlink(symlink, launcher_path)
    except OSError as exc:
        print(f"  ⚠ Failed to re-point symlink: {exc}")
        print(f"  Manually: ln -sf {launcher_path} {symlink}")

    # Assert original tree's git status is byte-identical
    status_after = _git_porcelain_status(tree_root)
    if status_before != status_after:
        print("  ⚠ Original tree's git status changed during worktree update!")
        print("  This should not happen — the original tree must be untouched.")
        result.errors.append("original tree status changed")
        result.success = False
    else:
        print("✓ Original tree's git status is byte-identical (untouched).")
        result.success = True

    print(f"\n✓ Switched to worktree: {wt_path}")
    print(f"  The `hermes` command now runs from this worktree.")
    print(f"  Your changes in {tree_root} are untouched.")

    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _prompt_choice(
    input_fn: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Present the 3-option choice to the user and return their selection.

    Returns ``"switch"``, ``"merge"``, or ``"cancel"``.
    """
    print()
    print("⚠ Local changes detected in this checkout.")
    print("  The worktree-based update can switch you to a fresh worktree")
    print("  without touching your changes, or merge in place.")
    print()
    print("  [1] Switch (default): create a new worktree at the target version,")
    print("      provision it, and re-point your `hermes` symlink there.")
    print("      Your current tree is left byte-identical.")
    print("  [2] Merge: fetch + merge in place. Stops on conflict (no stash,")
    print("      no auto-resolution). Resolve manually, like git merge.")
    print("  [3] Cancel")
    print()
    prompt = "Choose [1/2/3] (default 1): "
    default = "1"

    if input_fn is not None:
        response = input_fn(prompt, default).strip().lower()
    else:
        try:
            response = input(prompt).strip().lower()
        except EOFError:
            response = default

    mapping = {"1": "switch", "2": "merge", "3": "cancel",
               "switch": "switch", "merge": "merge", "cancel": "cancel"}
    return mapping.get(response, _DEFAULT_CHOICE)


# ---------------------------------------------------------------------------
# dev gc — list/remove old version-worktrees
# ---------------------------------------------------------------------------

def list_worktrees(tree_root: Path) -> list[Path]:
    """List all version-worktrees under ``.worktrees/``, sorted by mtime."""
    wt_dir = tree_root / _WORKTREES_DIR
    if not wt_dir.is_dir():
        return []
    worktrees = sorted(
        [d for d in wt_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    return worktrees


def _resolve_active_symlink_target() -> Optional[Path]:
    """Resolve the PATH symlink to find the active worktree (if any).

    Returns ``None`` if the symlink doesn't exist or doesn't point to
    a worktree.
    """
    try:
        symlink = _get_path_symlink()
        if symlink.is_symlink():
            return symlink.resolve()
    except Exception:
        pass
    return None


def gc_worktrees(
    tree_root: Path,
    keep_n: int = 2,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove old version-worktrees, keeping the most recent *keep_n*.

    Never removes the active symlink target (checked by resolving the PATH
    symlink).

    Returns a list of removed worktree paths.
    """
    worktrees = list_worktrees(tree_root)
    if not worktrees:
        return []

    active_target = _resolve_active_symlink_target()

    # Keep the most recent N, plus the active one
    to_keep = set(worktrees[-keep_n:]) if len(worktrees) > keep_n else set(worktrees)
    if active_target:
        for wt in worktrees:
            if wt.resolve() == active_target:
                to_keep.add(wt)

    to_remove = [wt for wt in worktrees if wt not in to_keep]
    if not to_remove:
        return []

    removed: list[Path] = []
    for wt in to_remove:
        if dry_run:
            removed.append(wt)
            continue
        result = _git(
            ["worktree", "remove", "--force", str(wt.resolve())],
            cwd=tree_root,
        )
        if result.returncode == 0:
            removed.append(wt)
        else:
            # Fallback: remove the directory directly
            import shutil

            shutil.rmtree(wt, ignore_errors=True)
            removed.append(wt)

    return removed
