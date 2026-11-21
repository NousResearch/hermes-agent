"""Assignee enumeration: the profiles on disk and the board's known assignees.

Moved out of ``hermes_cli.kanban_db`` (the board's per-profile lane pickers read these; the
kernel does not). ``known_assignees`` keeps reading per-assignee counts through the facade
(``_counts_by_assignee`` late-bound, so a patched facade is still the seam).
"""

from __future__ import annotations

import sqlite3
from typing import Any


def list_profiles_on_disk() -> list[str]:
    """Profiles with a ``config.yaml`` plus the implicit ``default``; reads paths
    directly to avoid importing ``hermes_cli.profiles`` at startup."""
    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
        profiles_dir = default_root / "profiles"
    except Exception:
        return []

    names: set[str] = set()
    if default_root.exists():
        names.add("default")
    if profiles_dir.is_dir():
        try:
            names.update(e.name for e in profiles_dir.iterdir() if e.is_dir() and (e / "config.yaml").is_file())  # config-reader: ok — profile directory discovery marker
        except OSError:
            pass
    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """``{"name", "on_disk", "counts"}`` for every on-disk profile or task
    assignee, so a fresh profile appears in pickers before it has a task."""
    from hermes_cli import kanban_db as kb  # late-bound: a patched facade stays the seam
    on_disk = set(list_profiles_on_disk())
    counts = kb._counts_by_assignee(conn)
    return [
        {"name": name, "on_disk": name in on_disk, "counts": counts.get(name, {})}
        for name in sorted(on_disk | set(counts))
    ]
