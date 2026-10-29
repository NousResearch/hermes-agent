"""What a profile copy must not take with it.

``shutil.copytree`` cannot copy special files (a live Unix socket, a FIFO, a device node), and a
clone must not inherit another profile's non-active bookkeeping. Both rules live here so
:mod:`hermes_cli.profiles` stays a facade over the copy paths that use them: profile export, the
plugins copy, ``--clone`` (the dashboard's "New bot") and ``--clone-all``.
"""

import os
import stat
from pathlib import Path


def _non_exportable_entries(directory: str, contents: list) -> set:
    """Entries under *directory* that must never be copied out of a profile: bytecode caches,
    ``*.sock``/``*.tmp`` names, and anything that is not a regular file, directory, or symlink.
    :func:`shutil.copytree` cannot copy special files, so a single live Unix socket without a
    ``.sock`` name (or a FIFO, or a device node) would abort the whole export or clone with
    ``[Errno 6] No such device or address``. Symlinks survive — copytree recreates them."""
    ignored: set = set()
    for entry in contents:
        if entry == "__pycache__" or entry.endswith((".sock", ".tmp", ".pyc", ".pyo")):
            ignored.add(entry)
            continue
        try:
            mode = os.lstat(os.path.join(directory, entry)).st_mode
        except OSError:
            ignored.add(entry)  # vanished mid-walk — copytree would fail on it anyway
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            ignored.add(entry)
    return ignored


# Non-active skill bookkeeping a clone does not inherit from its source profile. The curator
# retires skills into ``skills/.archive/`` (flattened; ``.archive/<name>-YYYYMMDDHHMMSS`` on a
# collision), which is the SOURCE profile's history — the same reason
# ``_CLONE_ALL_HISTORY_EXCLUDE_ROOT`` drops ``sessions/``, ``cron/`` and ``backups/`` from a
# clone instead of resurrecting them in a brand-new profile. It is also the largest single
# reservoir of very deep paths in a real profile, and the clone reproduces them under a staging
# directory ~30 characters longer, so a long archived path is what turns a Windows clone into
# ``[WinError 206] The filename or extension is too long`` (#145929).
_CLONE_SKILL_ROOT_EXCLUDE = frozenset({".archive"})


def _clone_skills_ignore(skills_root: Path):
    """copytree ignore for a cloned ``skills/`` tree: :func:`_non_exportable_entries` everywhere,
    plus :data:`_CLONE_SKILL_ROOT_EXCLUDE` at the skills ROOT only.

    Root-scoped on purpose: a skill's own ``.archive`` support directory belongs to that skill
    and travels with it. ``_non_exportable_entries`` is not changed here because profile export
    and the plugins copy share it, and both deliberately keep archived skills.
    """
    try:
        root_resolved = skills_root.resolve()
    except (OSError, ValueError):
        return _non_exportable_entries  # odd layout: fall back to today's behaviour

    def _ignore(directory: str, names: list) -> set:
        ignored = _non_exportable_entries(directory, names)
        try:
            at_skills_root = Path(directory).resolve() == root_resolved
        except (OSError, ValueError):
            at_skills_root = False  # fail open — copy rather than silently drop user data
        if at_skills_root:
            ignored.update(_CLONE_SKILL_ROOT_EXCLUDE & set(names))
        return ignored

    return _ignore
