"""Safety helpers for script-style Kanban stress tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType


def configure_temp_kanban_env(hermes_home: str) -> Path:
    """Pin the process to ``hermes_home`` and discard inherited worker state.

    Clear the whole Kanban namespace rather than a fixed selector list so a
    newly-added DB/path pin or runtime knob cannot make a stress script touch
    live state or silently change its race parameters.
    """
    for key in tuple(os.environ):
        if key.startswith("HERMES_KANBAN_") or key == "HERMES_DELEGATED_CHILD_CONTEXT":
            os.environ.pop(key, None)
    root = Path(hermes_home).expanduser().resolve()
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
