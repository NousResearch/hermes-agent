"""Per-board path surface (attachments, worker logs) for the Kanban DB.

Split out of ``hermes_cli.kanban_db`` so the façade does not grow: the layout rule
itself (``_board_path``) and the ``kanban.db`` / workspace roots stay in the façade,
while these thin per-board wrappers live here. Origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching ``kanban_db.<name>``
keeps working — ``task_attachments_dir`` reads ``_kb.attachments_root`` for exactly
that reason.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path


def attachments_root(board: Optional[str] = None) -> Path:
    """Per-board attachments root (``HERMES_KANBAN_ATTACHMENTS_ROOT`` wins). Workers
    read attachments by absolute path, so remote terminal backends must mount it."""
    return _kb._board_path("HERMES_KANBAN_ATTACHMENTS_ROOT", board, ("kanban", "attachments"), "attachments")


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return _kb.attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Per-board worker log dir (logs follow the board so ``hermes kanban log``
    is unambiguous when two boards share a task id)."""
    return _kb._board_path(None, board, ("kanban", "logs"), "logs")


# Late-bound origin namespace (see module docstring); imported LAST so this module
# is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
