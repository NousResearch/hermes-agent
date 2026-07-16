"""``hermes dev`` subcommand — sync / status / gc.

Provides developer-facing commands for working with source checkouts
(ejected mode).  Refuses to operate on managed slots.

- ``sync``: provisions the checkout (venv, node deps, builds)
- ``status``: shows venv health, node deps, build stamps
- ``gc``: lists/removes old version-worktrees (keep-N=2)

See ``docs/plans/updater-rework/04-phase3-ejected-dev.md`` tasks 3.1 + 3.2.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from hermes_cli.dev_sync import detect_tree_kind, run as dev_sync_run


# ---------------------------------------------------------------------------
# Tree-root resolution
# ---------------------------------------------------------------------------

def _resolve_tree_root() -> Path:
    """Resolve the tree root for the running code.

    Uses ``get_artifact_root`` from ``hermes_constants`` (same resolver
    the rest of the CLI uses for ``PROJECT_ROOT``).
    """
    from hermes_constants import get_artifact_root

    return get_artifact_root()


# ---------------------------------------------------------------------------
# Slot refusal guard
# ---------------------------------------------------------------------------

_SLOT_REFUSAL_MESSAGE = (
    "managed install — dev commands operate on source checkouts"
)


def _refuse_slot(tree_root: Path) -> bool:
    """Return True and print the refusal message if *tree_root* is a slot.

    Exits with code 2 when the tree is a slot.
    """
    if detect_tree_kind(tree_root) == "slot":
        print(f"Error: {_SLOT_REFUSAL_MESSAGE}", file=sys.stderr)
        sys.exit(2)
    return False


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_dev(args) -> None:
    """``hermes dev`` — dispatch to sync/status/gc."""
    tree_root = _resolve_tree_root()

    # Refuse slots — dev commands operate on source checkouts only
    _refuse_slot(tree_root)

    verb = getattr(args, "dev_verb", None)
    if verb == "sync":
        _cmd_dev_sync(args, tree_root)
    elif verb == "status":
        _cmd_dev_status(args, tree_root)
    elif verb == "gc":
        _cmd_dev_gc(args, tree_root)
    else:
        print(f"Unknown dev verb: {verb}", file=sys.stderr)
        sys.exit(2)


def _cmd_dev_sync(args, tree_root: Path) -> None:
    """``hermes dev sync`` — provision the checkout."""
    watch = getattr(args, "watch", False)
    only = getattr(args, "only", None)
    desktop = getattr(args, "desktop", False)

    print("→ Syncing source checkout...")
    dev_sync_run(
        tree_root,
        watch=watch,
        only=only,
        desktop=desktop,
    )
    print("✓ Sync complete")


def _cmd_dev_status(args, tree_root: Path) -> None:
    """``hermes dev status`` — show venv health, node deps, build stamps."""
    print(f"Tree root: {tree_root}")
    print(f"Tree kind: {detect_tree_kind(tree_root)}")
    print()

    # Venv health
    venv_dir = tree_root / ".venv"
    if venv_dir.is_dir():
        py_bin = venv_dir / "bin" / "python"
        if py_bin.exists():
            print(f"  ✓ venv: {venv_dir}")
        else:
            print(f"  ⚠ venv dir exists but python binary missing: {venv_dir}")
    else:
        print(f"  ✗ no venv at {venv_dir} — run: hermes dev sync")

    # Node deps
    node_modules = tree_root / "node_modules"
    if node_modules.is_dir():
        print(f"  ✓ node_modules: {node_modules}")
    else:
        print(f"  ✗ no node_modules — run: hermes dev sync")

    # Build stamps
    stamp_dir = tree_root / ".hermes-dev"
    if stamp_dir.is_dir():
        for stamp_file in sorted(stamp_dir.glob("*-build-stamp.json")):
            print(f"  ✓ stamp: {stamp_file.name}")
    else:
        print("  · no build stamps (run: hermes dev sync)")


def _cmd_dev_gc(args, tree_root: Path) -> None:
    """``hermes dev gc`` — list/remove old version-worktrees.

    keep-N=2, never the active symlink target.
    """
    keep_n = getattr(args, "keep", 2)
    dry_run = getattr(args, "dry_run", False)

    worktrees_dir = tree_root / ".worktrees"
    if not worktrees_dir.is_dir():
        print("No .worktrees directory — nothing to collect.")
        return

    # List version-worktrees (directories under .worktrees/)
    worktrees = sorted(
        [d for d in worktrees_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )

    if not worktrees:
        print("No version-worktrees to collect.")
        return

    # Determine the active symlink target (never delete it)
    active_target = None
    try:
        import os

        # Check the PATH 'hermes' symlink
        hermes_home = _get_hermes_home()
        current_symlink = hermes_home / "current"
        if current_symlink.is_symlink():
            active_target = current_symlink.resolve()
    except Exception:
        pass

    # Keep the most recent N, plus the active one
    to_keep = set(worktrees[-keep_n:]) if len(worktrees) > keep_n else set(worktrees)
    if active_target:
        for wt in worktrees:
            if wt.resolve() == active_target:
                to_keep.add(wt)

    to_remove = [wt for wt in worktrees if wt not in to_keep]

    if not to_remove:
        print(f"Nothing to remove ({len(worktrees)} worktree(s), keeping {len(to_keep)}).")
        return

    print(f"Worktrees to remove ({len(to_remove)}):")
    for wt in to_remove:
        print(f"  - {wt.name}")

    if dry_run:
        print("(dry-run — no changes made)")
        return

    for wt in to_remove:
        try:
            # Use git worktree remove for clean removal
            import subprocess

            result = subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)],
                cwd=str(tree_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"  ✓ removed {wt.name}")
            else:
                # Fallback: just remove the directory
                import shutil

                shutil.rmtree(wt, ignore_errors=True)
                print(f"  ✓ removed {wt.name} (forced)")
        except Exception as exc:
            print(f"  ✗ failed to remove {wt.name}: {exc}", file=sys.stderr)


def _get_hermes_home() -> Path:
    """Return the Hermes home directory."""
    from hermes_constants import get_hermes_home

    return get_hermes_home()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_dev_parser(subparsers, *, cmd_dev: Callable) -> None:
    """Attach the ``dev`` subcommand to ``subparsers``.

    Creates a ``dev`` parser with sub-verbs ``sync``, ``status``, ``gc``.
    """
    dev_parser = subparsers.add_parser(
        "dev",
        help="Developer commands for source checkouts (sync, status, gc)",
        description=(
            "Operate on a source checkout (ejected mode): provision it, "
            "check its state, or garbage-collect old version-worktrees. "
            "These commands do not apply to managed slot installs."
        ),
    )

    dev_subparsers = dev_parser.add_subparsers(
        dest="dev_verb",
        title="dev verbs",
        required=True,
    )

    # --- dev sync ---
    sync_parser = dev_subparsers.add_parser(
        "sync",
        help="Provision this checkout (venv, node deps, builds)",
        description=(
            "Run the full provision pipeline: create/sync the venv, "
            "install node dependencies, and build TUI/web/desktop artifacts "
            "as needed. Each build step is gated by a content-hash stamp "
            "so unchanged sources are skipped."
        ),
    )
    sync_parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Keep watching for changes and rebuild incrementally (not yet implemented).",
    )
    sync_parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STEP",
        help=(
            "Only run these steps. Valid steps: venv, launcher, node, "
            "tui, web, desktop, ledger."
        ),
    )
    sync_parser.add_argument(
        "--desktop",
        action="store_true",
        default=False,
        help="Force desktop build even if no previous desktop build exists.",
    )

    # --- dev status ---
    status_parser = dev_subparsers.add_parser(
        "status",
        help="Show venv health, node deps, and build stamps",
        description="Report the current state of this checkout's provisioned artifacts.",
    )

    # --- dev gc ---
    gc_parser = dev_subparsers.add_parser(
        "gc",
        help="List and remove old version-worktrees",
        description=(
            "Collect old version-worktrees under .worktrees/, keeping the "
            "most recent N (default 2). Never removes the active symlink target."
        ),
    )
    gc_parser.add_argument(
        "--keep",
        type=int,
        default=2,
        metavar="N",
        help="Number of recent worktrees to keep (default: 2).",
    )
    gc_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List what would be removed without actually removing anything.",
    )

    dev_parser.set_defaults(func=cmd_dev)
