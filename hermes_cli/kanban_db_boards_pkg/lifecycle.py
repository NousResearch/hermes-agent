"""Board archive/delete for the Kanban DB — the destructive half of the board lifecycle.

Split out of ``hermes_cli.kanban_db_boards`` (same reason the existing split modules exist:
that file is over its line cap and may not grow). ``remove_board`` reaches the root module
late-bound through ``_kb`` exactly like its former parent, so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import time

# Late-bound origin namespace; imported LAST so the root module is fully populated before
# any call lands here.
from hermes_cli import kanban_db as _kb


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Archive (to ``boards/_archived/<slug>-<ts>/``) or delete a board;
    ``default`` cannot be removed. Returns ``{"slug", "action", "new_path"}``."""
    _kb._assert_not_delegated_child_mutation()
    normed = _kb._require_slug(slug)
    if normed == _kb.DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = _kb.board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if _kb.get_current_board() == normed:
        _kb.clear_current_board()

    # A concurrent connect() after the rename recreates an empty DB file; drop
    # the init cache first so the schema pass re-runs on it.
    _kb._INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        # Capture display metadata before the move so the tombstone below keeps
        # the user's board name instead of falling back to a title-cased slug.
        prior_meta = _kb.read_board_metadata(normed)
        archive_root = _kb.boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        suffix = 1
        while target.exists():  # rapid double-archive
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        # Leave an ``archived`` tombstone at the original slug. Stale dashboard
        # tabs / gateway pollers can keep calling connect(board=slug) after the
        # archive; without a marker the resurrect-guard cannot tell an archived
        # slug from a brand-new one and an empty board would reappear (#43243).
        _kb.write_board_metadata(
            normed,
            name=prior_meta.get("name"),
            description=prior_meta.get("description"),
            icon=prior_meta.get("icon"),
            color=prior_meta.get("color"),
            default_workdir=prior_meta.get("default_workdir"),
            project_id=prior_meta.get("project_id"),
            archived=True,
        )
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    import shutil
    shutil.rmtree(d)
    return {"slug": normed, "action": "deleted", "new_path": ""}


__all__ = ["remove_board"]
