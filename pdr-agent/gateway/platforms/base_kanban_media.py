"""Kanban attachment/board paths for the media-delivery allow and deny lists (split from
:mod:`gateway.platforms.base`)."""
import contextlib
import os
import re
from pathlib import Path


def _kanban_root() -> Path:
    """Kanban is root-shared across profiles by design (``kanban_db.kanban_home``)."""
    from gateway.platforms import base  # late: tests repoint the facade's _HERMES_ROOT
    return Path(os.environ.get("HERMES_KANBAN_HOME", "").strip() or base._HERMES_ROOT).expanduser()


def _kanban_board_dirs() -> list[Path]:
    """Every directory under ``<root>/kanban/boards`` (lax on purpose: the DENY side must catch a
    board whatever its name; the allow side filters further)."""
    with contextlib.suppress(OSError):
        return [p for p in (_kanban_root() / "kanban" / "boards").iterdir() if p.is_dir()]
    return []


def _kanban_attachment_roots() -> list[Path]:
    """Return durable Kanban attachment roots without importing kanban_db."""
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        return [Path(override).expanduser()]
    roots = [_kanban_root() / "kanban" / "attachments"]
    roots.extend(path / "attachments" for path in _kanban_board_dirs()
                 if not path.is_symlink() and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", path.name)
                 and (path / "kanban.db").is_file())
    return roots


def _kanban_board_db_paths() -> list[Path]:
    """Named-board ``kanban.db`` stores (+ sidecars): they sit beside the ATTACHMENTS dir
    ``_kanban_attachment_roots`` allowlists and hold every task, comment and run transcript."""
    from gateway.platforms import base
    return [board / name for board in _kanban_board_dirs() for name in base._sqlite_files("kanban.db")]
