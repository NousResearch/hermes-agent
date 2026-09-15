"""The one place that answers "which packaged desktop app should this host use?".

Four copies of that question used to exist, and they disagreed:

* ``main_desktop._desktop_packaged_executable_in`` — glob ``mac*`` + newest mtime,
* ``doctor_platform._desktop_app_bundle`` — a second glob ``mac*`` + newest mtime,
* ``scripts/install.sh`` and ``scripts/desktop-update/posix.sh`` — two hardcoded
  ``mac-arm64`` then ``mac`` candidate lists.

Only the first ever learned to reject a bundle this Mac cannot load, so the installer
and the updater could still bless an arm64 Electron packed against ``darwin-x64``
node-pty prebuilds — an app that launches and dies on "Failed to load native module:
pty.node". Divergent copies of a selection rule are how that shipped three times.

This module is importable from Python and runnable as
``python -m hermes_cli.desktop_app_path`` so the shell callers share the verdict
instead of re-deriving it.

CLI exit codes:
  0  usable app; its path is on stdout
  1  no packaged app at all
  2  an app is present but unusable on this host; the reason is on stderr

A caller that can run this module must trust its verdict: exit 2 means "do not install
or launch this", never "fall back to guessing".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from hermes_cli.main_desktop import (
    _desktop_exe_integrity_error,
    _desktop_packaged_executable_in,
)


def release_dir(project_root: Path) -> Path:
    """electron-builder's ``directories.output`` for this checkout."""
    return project_root / "apps" / "desktop" / "release"


def packaged_executable(project_root: Path) -> Optional[Path]:
    """The executable the launcher would start, validated or not (None when there is none)."""
    return _desktop_packaged_executable_in(release_dir(project_root))


def app_bundle(project_root: Path) -> Optional[Path]:
    """What callers install, sign or open: the ``.app`` on macOS, the executable elsewhere.

    Deliberately *not* filtered on loadability — this answers "which app is installed here",
    which is what ``hermes doctor`` must report on even when that app is broken. Use
    :func:`validated_app_bundle` when the answer will be acted on.
    """
    exe = packaged_executable(project_root)
    if exe is None:
        return None
    return exe.parents[2] if sys.platform == "darwin" else exe


def validated_app_bundle(project_root: Path) -> tuple[Optional[Path], Optional[str]]:
    """``(bundle, None)`` when this host can run it, else ``(None, reason)``."""
    exe = packaged_executable(project_root)
    if exe is None:
        return None, f"no packaged desktop app under {release_dir(project_root)}"
    error = _desktop_exe_integrity_error(exe)
    if error is not None:
        return None, error
    return app_bundle(project_root), None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hermes_cli.desktop_app_path", description=__doc__)
    parser.add_argument("project_root", nargs="?", default=None,
                        help="checkout root (defaults to the one this module was imported from)")
    parser.add_argument("--allow-unusable", action="store_true",
                        help="print the installed app even when this host cannot run it")
    args = parser.parse_args(argv)

    root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]

    if args.allow_unusable:
        bundle = app_bundle(root)
        if bundle is None:
            print(f"no packaged desktop app under {release_dir(root)}", file=sys.stderr)
            return 1
        print(bundle)
        return 0

    bundle, error = validated_app_bundle(root)
    if bundle is not None:
        print(bundle)
        return 0
    print(error, file=sys.stderr)
    return 1 if packaged_executable(root) is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
