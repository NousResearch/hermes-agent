"""Stale Python bytecode diagnostics for ``hermes doctor``."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Iterable

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".worktrees"}


def _stale_bytecode_dirs(roots: Iterable[Path]) -> list[Path]:
    """Return cache directories containing bytecode older than its source."""
    stale: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in _SKIP_DIRS and not (current / name).is_symlink()
            ]
            if current.name != "__pycache__":
                continue
            dirnames.clear()
            if current in seen:
                continue
            seen.add(current)
            for filename in filenames:
                if not filename.endswith(".pyc"):
                    continue
                cache_file = current / filename
                try:
                    source_file = Path(importlib.util.source_from_cache(str(cache_file)))
                    if source_file.stat().st_mtime_ns > cache_file.stat().st_mtime_ns:
                        stale.append(current)
                        break
                except (OSError, ValueError):
                    continue
    return stale


@doctor_check("Python bytecode cache", "(could not inspect: {e})")
def _check_stale_bytecode(should_fix: bool, f: Finding) -> None:
    from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT

    stale = _stale_bytecode_dirs((PROJECT_ROOT, HERMES_HOME))
    if not stale:
        check_ok("Python bytecode cache", "(no stale cache detected)")
        return

    count = len(stale)
    suffix = "y" if count == 1 else "ies"
    if not should_fix:
        check_warn(f"Stale bytecode cache detected in {count} director{suffix}")
        f.issues.append(
            f"Stale bytecode cache detected in {count} director{suffix}. "
            "Run 'hermes doctor --fix' to clear."
        )
        return

    removed = 0
    for cache_dir in stale:
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            continue
        if not cache_dir.exists():
            removed += 1

    if removed:
        fixed_suffix = "y" if removed == 1 else "ies"
        check_ok(f"Cleared {removed} stale __pycache__ director{fixed_suffix}")
        check_info("Restart running gateways to load fresh modules: hermes gateway restart")
        f.fixed += removed
    remaining = count - removed
    if remaining:
        remaining_suffix = "y" if remaining == 1 else "ies"
        check_warn(f"Could not clear {remaining} stale __pycache__ director{remaining_suffix}")
        f.issues.append(
            f"Could not clear {remaining} stale __pycache__ director{remaining_suffix}; "
            "check filesystem permissions."
        )