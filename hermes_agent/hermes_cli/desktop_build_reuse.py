"""Reuse a packaged desktop app across commits that changed none of its build inputs.

``hermes update`` used to rebuild and repackage the Electron app (6-38 min on Windows) whenever HEAD
moved, because ``_packaged_desktop_current_for_head`` requires the app's baked commit to BE HEAD. The
freshness receipt already proves everything else the build reads is byte-identical; the baked
install stamp is the one input it cannot see, since only a build rewrites it. This module answers
the missing half: would a build at HEAD bake a stamp that differs in anything but commit provenance?

The reused app keeps naming the commit it was really built from (never rewritten: the running app
compares its boot stamp with the one on disk and would relaunch into the "swap" forever). Every
reader of that commit tolerates an older one: bundle skew counts only desktop runtime-path commits
since it, which identical inputs rule out, and bootstrap repair follows the checkout's own HEAD.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _stamp_drift(project_root: Path, baked: Path, env: dict) -> Optional[list[str]]:
    """Fields a build at HEAD would bake differently, or None when that cannot be told."""
    from pm import installed_package

    installed = installed_package("node")
    if installed is None or installed.binary is None:
        return None
    try:
        result = subprocess.run(
            [str(installed.binary), str(project_root / "scripts/build/desktop-reuse.mjs"),
             "--source", str(project_root), "--baked", str(baked)],
            cwd=project_root, env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, check=True,
        )
        drift = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return drift if isinstance(drift, list) else None


def packaged_desktop_reusable(desktop_dir: Path, project_root: Path, env: dict) -> Optional[str]:
    """The commit the packaged app was built from, when it can stand for HEAD unchanged.

    Requires a current freshness receipt for the packaged bundle (inputs and output intact, not
    torn, native binary present) whose baked stamp is the prepared stamp that receipt covers, and a
    stamp a build at HEAD would bake that differs only in commit provenance. Anything unreadable
    means build.
    """
    from hermes_cli.main_desktop import _desktop_build_needed, _packaged_resources_dir

    resources = _packaged_resources_dir(desktop_dir)
    if resources is None:
        return None
    baked_path = resources / "install-stamp.json"
    try:
        baked = json.loads(baked_path.read_text(encoding="utf-8-sig"))
        prepared = json.loads((desktop_dir / "build/install-stamp.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    commit = baked.get("commit") if isinstance(baked, dict) else None
    if not isinstance(commit, str) or not commit or prepared != baked:
        return None
    if _desktop_build_needed(desktop_dir, project_root, source_mode=False):
        return None
    if sys.platform == "win32":
        # The build resolves HEAD and branch with PM's git (build_prepared_desktop); so must this.
        import pm
        env = pm.ensure("git", base_env=env).env
    return commit if _stamp_drift(project_root, baked_path, env) == [] else None
