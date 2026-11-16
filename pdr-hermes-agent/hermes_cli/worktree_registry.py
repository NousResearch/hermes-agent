"""Per-profile registry of every repository Hermes has created a worktree in.

The 6-hourly cron sweep (``cron.scheduler._worktree_maintenance_repos``) only knew the Hermes
checkout and cron job workdirs, so trees made by ``hermes -w`` / ``/worktree new`` / subagent
isolation / the Desktop git pane in any other repo were never swept. Each creation path records
its repo here; the sweep unions this list with its other sources.

The file lives under the ACTIVE profile's home (``get_hermes_home()``), so one profile's sweep
never visits another profile's repos. Writes are atomic under a cross-process lock; entries whose
repo no longer exists are dropped on every read-modify-write and filtered on read.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_NAME = "worktree_repos.json"
_holder = threading.local()


def registry_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _REGISTRY_NAME


def _repo_exists(repo: str) -> bool:
    return (Path(repo) / ".git").exists()


def _read(path: Path) -> list[str]:
    from utils import read_json_or_empty
    repos = read_json_or_empty(path).get("repos")
    return [r for r in repos if isinstance(r, str) and r] if isinstance(repos, list) else []


def registered_repos() -> list[str]:
    """Registered repo roots that still exist (read-only; never creates the file)."""
    try:
        return [r for r in _read(registry_path()) if _repo_exists(r)]
    except Exception:
        logger.debug("worktree registry read failed", exc_info=True)
        return []


def record_worktree_repo(repo_root) -> None:
    """Remember *repo_root* as a repo with Hermes-made worktrees. Never raises."""
    if not repo_root:
        return
    try:
        from hermes_cli.auth import _file_lock
        from utils import atomic_json_write

        repo = os.path.realpath(str(repo_root))
        path = registry_path()
        with _file_lock(path.with_name(path.name + ".lock"), _holder, 5.0,
                        "Timed out waiting for the worktree registry lock"):
            current = _read(path)
            kept = [r for r in dict.fromkeys(current) if _repo_exists(r)]
            if repo in kept and kept == current:
                return
            if repo not in kept:
                kept.append(repo)
            atomic_json_write(path, {"version": 1, "repos": kept})
    except Exception:
        logger.debug("worktree registry update failed for %s", repo_root, exc_info=True)
