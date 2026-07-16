"""``hermes eject`` subcommand — switch from a managed slot to a source checkout.

Phase 3, task 3.5 of the updater rework.

From a **slot** (managed bundle): ``hermes eject [--dir PATH]`` (default
``$HERMES_HOME/source``):

1. Clones the repo at the slot's ``git_sha`` (from the slot manifest).
2. Runs ``hermes dev sync`` (provisions the checkout).
3. Re-points the PATH symlink to the checkout's ``bin/hermes``.
4. Prints the ejected-contract caveats (§2.5).
5. Records ``.pre-eject-target`` for undo symmetry with ``adopt``.

From a **checkout**: exits 0 with "already ejected" + status.

See ``docs/plans/updater-rework/04-phase3-ejected-dev.md`` task 3.5 and
``docs/updater-world.md`` §2.5.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical repo URL — same as ``OFFICIAL_REPO_URL`` in main.py.
REPO_URL = "https://github.com/NousResearch/hermes-agent.git"

# The PATH symlink target for a checkout's launcher.
_BIN_HERMES = "bin/hermes"
# Windows launcher name.
_BIN_HERMES_WIN = "bin/hermes.exe"

# Marker file recording the pre-eject target (the managed slot path), for
# undo symmetry with ``hermes adopt``.
_PRE_EJECT_TARGET_FILE = ".pre-eject-target"


# ---------------------------------------------------------------------------
# Tree-kind detection
# ---------------------------------------------------------------------------

def _is_slot(project_root: Path) -> bool:
    """Return True if *project_root* is a managed slot (has manifest.json)."""
    return (project_root / "manifest.json").is_file()


def _is_checkout(project_root: Path) -> bool:
    """Return True if *project_root* is a source checkout (has .git)."""
    git_path = project_root / ".git"
    return git_path.is_dir() or git_path.is_file()


def _read_slot_git_sha(project_root: Path) -> Optional[str]:
    """Read the ``git_sha`` field from the slot's ``manifest.json``.

    Returns ``None`` if the manifest is missing, unreadable, or lacks
    ``git_sha``.
    """
    manifest_path = project_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sha = manifest.get("git_sha")
    if isinstance(sha, str) and sha.strip():
        return sha.strip()
    return None


# ---------------------------------------------------------------------------
# Git clone helper
# ---------------------------------------------------------------------------

def _clone_at_sha(
    dest: Path,
    git_sha: str,
    *,
    repo_url: str = REPO_URL,
) -> Path:
    """Clone *repo_url* at commit *git_sha* into *dest*.

    Uses ``git clone`` followed by ``git checkout <sha>`` so the checkout
    matches the slot's version exactly.  If *dest* already exists and is
    non-empty, it is left untouched (the clone is skipped).
    """
    if dest.exists() and any(dest.iterdir()):
        # Already cloned — attempt checkout at the requested sha.
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=dest,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "checkout", git_sha],
            cwd=dest,
            capture_output=True,
            timeout=60,
        )
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed: {result.stderr.strip() or 'unknown error'}"
        )
    # Checkout the specific sha.
    result = subprocess.run(
        ["git", "checkout", git_sha],
        cwd=dest,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git checkout {git_sha} failed: "
            f"{result.stderr.strip() or 'unknown error'}"
        )
    return dest


# ---------------------------------------------------------------------------
# dev sync helper
# ---------------------------------------------------------------------------

def _run_dev_sync(checkout_dir: Path) -> None:
    """Run ``hermes dev sync`` inside *checkout_dir*.

    This provisions the checkout (venv, node deps, builds).  We invoke the
    checkout's own launcher so the right tree's environment is set up.
    """
    launcher = checkout_dir / _BIN_HERMES
    if sys.platform == "win32":
        launcher = checkout_dir / _BIN_HERMES_WIN

    result = subprocess.run(
        [str(launcher), "dev", "sync", "--dev"],
        cwd=str(checkout_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hermes dev sync failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# Symlink management
# ---------------------------------------------------------------------------

def _get_path_symlink() -> Path:
    """Return the path to the ``hermes`` symlink on PATH.

    Mirrors the logic in ``hermes_cli/doctor.py``: ``$PREFIX/bin/hermes``
    on Termux, ``~/.local/bin/hermes`` elsewhere.
    """
    prefix = os.environ.get("PREFIX", "")
    if prefix and "com.termux" in prefix:
        return Path(prefix) / "bin" / "hermes"
    return Path.home() / ".local" / "bin" / "hermes"


def _repoint_symlink(symlink_path: Path, target: Path) -> None:
    """Re-point *symlink_path* at *target* (the checkout's bin/hermes).

    Removes the existing symlink (or file) and creates a new one.
    """
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(target)


# ---------------------------------------------------------------------------
# Caveats (§2.5)
# ---------------------------------------------------------------------------

_EJECT_CAVEATS = """\
────────────────────────────────────────────────────────────────────
  EJECTED MODE — you are now running from a source checkout.

  Caveats (§2.5 of the updater contract):

  • You build locally. You need Node.js, npm, and build tools installed.
    Run `hermes dev sync` after pulling to rebuild.
  • Syntax guard can only rollback git state, not your venv. If an
    update breaks the venv, `hermes update` can revert the code but
    you may need to re-run `hermes dev sync` to fix the environment.
  • Desktop rebuilds are on you. The Electron app is not pre-built in
    a checkout — run `hermes dev sync --desktop` to build it.
  • Update-boundary bugs can require running `hermes update` twice:
    once to pull new code, once after `dev sync` provisions it.
  • CI-untested combinations are possible. Your local toolchain
    versions may differ from CI, producing untested configurations.

  To return to managed releases: `hermes adopt`
────────────────────────────────────────────────────────────────────"""


def _print_caveats() -> None:
    """Print the ejected-contract caveats to stderr."""
    print(_EJECT_CAVEATS, file=sys.stderr)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def cmd_eject(args) -> None:
    """``hermes eject`` — switch from a managed slot to a source checkout.

    See module docstring for the full flow.
    """
    from hermes_constants import get_hermes_home

    # Resolve project root from main (avoids a circular import at module load).
    from hermes_cli.main import PROJECT_ROOT

    project_root = Path(PROJECT_ROOT)

    # --- Already ejected? ---
    if _is_checkout(project_root):
        # We're already in a checkout — nothing to do.
        print("Already ejected — running from a source checkout.")
        print(f"  Checkout: {project_root}")
        symlink = _get_path_symlink()
        if symlink.is_symlink():
            print(f"  Symlink:  {symlink} → {symlink.resolve()}")
        else:
            print(f"  Symlink:  {symlink} (not found)")
        return

    # --- Must be a slot to eject ---
    if not _is_slot(project_root):
        print(
            "Cannot eject: this does not appear to be a managed slot install.\n"
            "  Eject is for switching from a managed slot to a source checkout.\n"
            "  If you're already running from a checkout, you're already ejected.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Read git_sha from the slot manifest ---
    git_sha = _read_slot_git_sha(project_root)
    if not git_sha:
        print(
            "Cannot eject: the slot manifest does not contain a git_sha.\n"
            "  The manifest at {}/manifest.json must have a 'git_sha' field."
            .format(project_root),
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Determine the checkout destination ---
    hermes_home = get_hermes_home()
    dest_dir = Path(getattr(args, "dir", None) or (hermes_home / "source"))

    print(f"Ejecting from managed slot to source checkout...")
    print(f"  Slot:      {project_root}")
    print(f"  git_sha:   {git_sha}")
    print(f"  Checkout:  {dest_dir}")

    # --- 1. Clone the repo at the slot's git_sha ---
    try:
        _clone_at_sha(dest_dir, git_sha)
    except Exception as exc:
        print(f"\nError: failed to clone repository: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Run hermes dev sync (provisions the checkout) ---
    print(f"\nProvisioning checkout (hermes dev sync)...")
    try:
        _run_dev_sync(dest_dir)
    except Exception as exc:
        print(
            f"\nWarning: hermes dev sync failed: {exc}\n"
            f"The checkout has been cloned but not fully provisioned.\n"
            f"Run `hermes dev sync` inside {dest_dir} to complete setup.",
            file=sys.stderr,
        )

    # --- 3. Re-point the PATH symlink ---
    launcher_path = dest_dir / _BIN_HERMES
    if sys.platform == "win32":
        launcher_path = dest_dir / _BIN_HERMES_WIN

    symlink = _get_path_symlink()
    print(f"\nRe-pointing symlink: {symlink} → {launcher_path}")
    try:
        _repoint_symlink(symlink, launcher_path)
    except OSError as exc:
        print(
            f"\nError: failed to re-point symlink: {exc}\n"
            f"Manually create the symlink:\n"
            f"  ln -sf {launcher_path} {symlink}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 4. Record .pre-eject-target for undo symmetry with adopt ---
    pre_eject_path = hermes_home / _PRE_EJECT_TARGET_FILE
    try:
        pre_eject_path.write_text(str(project_root) + "\n", encoding="utf-8")
    except OSError:
        # Non-fatal — the caveats and symlink are the important parts.
        pass

    # --- 5. Print the ejected-contract caveats ---
    _print_caveats()

    print(f"\nEjected successfully. The `hermes` command now runs from:")
    print(f"  {dest_dir}")
    print(f"\nTo return to managed releases: hermes adopt")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_eject_parser(subparsers, *, cmd_eject: Callable) -> None:
    """Attach the ``eject`` subcommand to ``subparsers``."""
    eject_parser = subparsers.add_parser(
        "eject",
        help="Switch from a managed slot to a source checkout",
        description=(
            "Clone the Hermes Agent repo at the slot's version, provision it "
            "with `hermes dev sync`, and re-point your PATH symlink at the "
            "checkout's launcher.  You'll be running from source — see the "
            "ejected-contract caveats (§2.5) printed on success."
        ),
    )
    eject_parser.add_argument(
        "--dir",
        default=None,
        metavar="PATH",
        help=(
            "Destination directory for the checkout.  "
            "Default: $HERMES_HOME/source"
        ),
    )
    eject_parser.set_defaults(func=cmd_eject)
