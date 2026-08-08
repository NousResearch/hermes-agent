"""Termux bundled-skills startup sync helpers - extracted from ``hermes_cli.main``.

God-file decomposition (wave 1, shard s1, cluster c9): these functions moved
verbatim out of ``hermes_cli/main.py``. ``hermes_cli.main`` re-exports them so
the CLI surface is unchanged.

Names that still live in ``hermes_cli.main`` (``PROJECT_ROOT``, ``get_hermes_home``,
``_is_termux_startup_environment``, ``_read_git_revision_fingerprint``) are imported
lazily inside the functions that use them - the house pattern already used by
``hermes_cli/model_setup_flows.py`` and ``hermes_cli/cli_commands_mixin.py``. A
module-level ``from hermes_cli.main import ...`` would create an import cycle
(``main`` re-exports this module), and the function-level import re-reads
``hermes_cli.main``'s binding at call time, so monkeypatching the CLI module keeps
working.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes_cli import __version__, __release_date__


def _termux_bundled_skills_fingerprint() -> str:
    """Cheap invalidation key for Termux bundled-skill startup sync."""
    from hermes_cli.main import PROJECT_ROOT, _read_git_revision_fingerprint

    git_fp = _read_git_revision_fingerprint(PROJECT_ROOT)
    if git_fp:
        return git_fp
    skills_dir = PROJECT_ROOT / "skills"
    try:
        stat = skills_dir.stat()
        return f"skills:{__version__}:{__release_date__}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"skills:{__version__}:{__release_date__}:missing"


def _termux_bundled_skills_stamp_path() -> Path:
    from hermes_cli.main import get_hermes_home

    return get_hermes_home() / "skills" / ".termux_bundled_sync_stamp"


def _termux_bundled_skills_sync_needed() -> bool:
    from hermes_cli.main import _is_termux_startup_environment

    if not _is_termux_startup_environment():
        return True
    if os.environ.get("HERMES_TERMUX_FORCE_SKILLS_SYNC") == "1":
        return True
    try:
        stamp = _termux_bundled_skills_stamp_path()
        return stamp.read_text(encoding="utf-8").strip() != _termux_bundled_skills_fingerprint()
    except OSError:
        return True


def _mark_termux_bundled_skills_synced() -> None:
    from hermes_cli.main import _is_termux_startup_environment

    if not _is_termux_startup_environment():
        return
    try:
        stamp = _termux_bundled_skills_stamp_path()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_termux_bundled_skills_fingerprint() + "\n", encoding="utf-8")
    except OSError:
        pass


def _sync_bundled_skills_for_startup() -> bool:
    """Sync bundled skills, but skip unchanged Termux checkouts cheaply.

    Hashing every bundled skill is safe but expensive on older Android
    storage. The git/ref stamp keeps post-update correctness: a changed
    checkout revision forces one real sync, then later starts skip it.
    """
    from hermes_cli.main import _is_termux_startup_environment

    if _is_termux_startup_environment() and not _termux_bundled_skills_sync_needed():
        return False

    from tools.skills_sync import sync_skills

    sync_skills(quiet=True)
    _mark_termux_bundled_skills_synced()
    return True


def _termux_should_prefetch_update_check() -> bool:
    from hermes_cli.main import _is_termux_startup_environment

    if not _is_termux_startup_environment():
        return True
    return os.environ.get("HERMES_TERMUX_PREFETCH_UPDATES") == "1"
