"""Board-slug derivation from the resolved Kanban DB path (reporting only)."""

from __future__ import annotations

from typing import Optional


def resolved_board_slug(board: Optional[str] = None) -> str:
    """Slug of the board a call with this ``board`` argument lands on — the same
    resolution :func:`hermes_cli.kanban_db.kanban_db_path` applies, env pins and
    the worker/delegation fence (:func:`hermes_cli.kanban_db._explicit_board_intent_pinned`)
    included. ``<root>/kanban.db`` maps to ``default``; a named board maps back
    by layout (``<boards-root>/<slug>/kanban.db``); a pinned path outside that
    layout has no slug, so the path itself is returned (it still identifies the
    board the call resolves to). Purely reporting: this never changes
    resolution — callers use it to echo *where a write landed* so a fenced
    explicit ``board=`` is observable instead of silent (#142435 / #144132).
    The value is for display, not round-tripping: under a pin outside the
    boards layout it is a path, not a slug, so do not feed it back into a
    ``board=`` argument unconditionally."""
    from .kanban_db import DEFAULT_BOARD, boards_root, kanban_db_path, kanban_home

    path = kanban_db_path(board=board)
    try:
        resolved = path.expanduser().resolve()
        if resolved == kanban_home().joinpath("kanban.db").resolve():
            return DEFAULT_BOARD
        try:
            rel = resolved.parent.relative_to(boards_root().resolve())
        except ValueError:
            rel = None
        if rel is not None and len(rel.parts) == 1:
            return rel.parts[0]
    except OSError:
        pass
    return str(path)
