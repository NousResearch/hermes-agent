"""Safety helpers for script-style Kanban stress tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType


_KANBAN_PATH_OVERRIDES = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
)


def configure_temp_kanban_env(hermes_home: str) -> Path:
    """Pin the process to ``hermes_home`` and discard inherited board pins."""
    root = Path(hermes_home).resolve()
    for key in _KANBAN_PATH_OVERRIDES:
        os.environ.pop(key, None)
    os.environ["HERMES_HOME"] = str(root)
    os.environ["HOME"] = str(root)
    return root


def assert_temp_kanban_db(kb: ModuleType, hermes_home: str | Path) -> Path:
    """Abort unless the DB selected by ``kb`` is below the stress tempdir."""
    root = Path(hermes_home).resolve()
    db_path = kb.kanban_db_path().resolve()
    try:
        db_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"refusing to run stress test outside tempdir: {db_path} not under {root}"
        ) from exc
    return db_path
