"""Recognizing the machine's kanban board paths without opening anything.

Split out of ``hermes_cli.kanban_db`` (which is at its file-size cap): the one
board predicate the generic file boundaries ask, and the unrefused root
resolution it needs. ``kanban_home`` keeps its hosted refusal — this module
never opens a board, so it resolves the root without refusing: recognizing a
board in a process that must not open it is exactly what the predicate is for.
"""

from __future__ import annotations

import os
from pathlib import Path

def unrefused_kanban_home() -> Path:
    """``kanban_home`` without its hosted refusal: the same resolution, for the
    callers that must recognize the board root in a process that may not open
    it (see :func:`is_board_path`)."""
    from hermes_cli.kanban_db_home import scoped_home_override
    override = scoped_home_override()  # a context-scoped pin (see kanban_db_home)
    if override:
        return Path(override).expanduser()
    env_override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()

def _resolved_str(path: Path | str) -> str:
    """Symlink-resolved absolute string form of *path*; the unresolved string on
    an OSError (an unreadable path must never crash a predicate)."""
    try:
        return os.path.realpath(os.fspath(path))
    except (OSError, TypeError, ValueError):
        return str(path)

def is_board_path(path: Path | str) -> bool:
    """Whether *path* is one of this machine's kanban board databases — the
    default board at ``<root>/kanban.db``, a file inside a named board's
    directory under ``<root>/kanban/boards``, or an explicit
    ``HERMES_KANBAN_DB`` pin. This is the one predicate the generic file
    boundaries (the backup code's SQLite receiver) ask to tell a board from any
    other SQLite file, so they can refuse a hosted process before the board's
    bytes are opened or copied. Symlinks are resolved on both sides."""
    target = _resolved_str(path)
    candidates = []
    pin = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if pin:
        candidates.append(pin)
    root = unrefused_kanban_home()
    candidates.append(root / "kanban.db")
    boards = root / "kanban" / "boards"
    try:
        if boards.is_dir():
            # Any file inside a named board's directory counts (a fenced
            # descendant or a board database alike): the board owns its dir.
            boards_real = _resolved_str(boards)
            if target == boards_real or target.startswith(boards_real + os.sep):
                return True
            for entry in boards.iterdir():
                if entry.is_dir():
                    candidates.append(entry / "kanban.db")
    except OSError:
        pass
    return any(_resolved_str(candidate) == target for candidate in candidates)

def refuse_board_path_if_hosted(path: Path | str) -> None:
    """Refuse when this process serves hosted tenants and *path* is one of the
    machine's kanban board databases (see :func:`is_board_path`): the backup
    code's generic SQLite receiver must not open or copy the board's bytes
    here. An operator backs the machine up offline, from a terminal profile."""
    from hermes_kanban_port import _hosted_process
    if _hosted_process(None) and is_board_path(path):
        raise LookupError(
            "the local kanban board keeps one board root shared across every profile of a "
            "single-user machine; it refuses to open in a hosted process"
        )