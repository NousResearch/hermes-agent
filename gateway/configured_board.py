"""Canonical paths for explicitly configured gateway Kanban boards."""

from __future__ import annotations

import re
from pathlib import Path


_BOARD_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def configured_board_db_path(board: object) -> Path:
    """Resolve one explicit board without honoring the per-process DB override."""
    if not isinstance(board, str) or not _BOARD_RE.fullmatch(board):
        raise ValueError("an explicit valid Kanban board is required")

    from hermes_cli import kanban_db as kb

    if board == kb.DEFAULT_BOARD:
        return kb.kanban_home().expanduser() / "kanban.db"
    return kb.boards_root().expanduser() / board / "kanban.db"
