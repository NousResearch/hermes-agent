"""Board metadata (``board.json``) and board lifecycle management for the Kanban DB:
read/write of per-board display metadata, board creation/discovery/archival, and the
``board.json``-as-identity-marker rule.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import time
from typing import Any
from typing import Optional
from pathlib import Path


def _legacy_db_holds_schema(d: Path) -> bool:
    """Whether ``d/kanban.db`` carries the kanban schema (``tasks`` sentinel).

    A pre-``board.json`` board (created before the metadata file existed)
    has an initialized DB; the #43243 stub left by a stale ``connect()`` is a
    zero-byte or schema-less file. The sentinel lookup mirrors
    ``kanban_db_connect._schema_is_present`` — read-only, one page.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{d / 'kanban.db'}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks' LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None


def _backfill_legacy_board_metadata(d: Path) -> None:
    """Write a minimal ``board.json`` for a schema-holding legacy board dir.

    Runs once per legacy board: after the first discovery the metadata file
    exists and the normal identity path takes over. Best-effort by design —
    a fenced delegated-child context (dashboard/poller read paths call
    ``list_boards``) must not crash or mutate: the board stays visible and
    the write lands on the next unfenced discovery.
    """
    try:
        _kb._assert_not_delegated_child_mutation(d)
    except PermissionError:
        return
    try:
        slug = d.name
        meta = {
            "slug": slug,
            "name": _default_board_display_name(slug),
            "description": "",
            "icon": "",
            "color": "",
            "default_workdir": None,
            "project_id": None,
            "created_at": time.time(),
            "archived": False,
        }
        (d / "board.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass


def _dir_holds_board(d: Path) -> bool:
    # ``board.json`` is the identity marker: archive/hard-delete both leave the
    # directory without it, and a stale ``connect(board=slug)`` used to leave a
    # ``kanban.db``-only stub that resurfaced in the board list as an empty
    # active board (#43243). Discovery must therefore require the metadata
    # file; a DB-only directory is a stub to ignore, never a board.
    if (d / "board.json").exists():
        return True
    # Exception: a legacy board created before ``board.json`` existed holds a
    # real, schema-initialized kanban.db. Commit 63e5656409's identity rule
    # made those invisible with no migration path (#135556), so discovery
    # backfills the marker once and treats the directory as a board again.
    # The #43243 stub has no schema and stays ignored.
    if (d / "kanban.db").is_file() and _legacy_db_holds_schema(d):
        _backfill_legacy_board_metadata(d)
        return True
    return False


def board_metadata_path(board: Optional[str] = None) -> Path:
    """``board.json`` path — display metadata only; the directory slug is the identity."""
    return _kb.board_dir(_kb._slug_or_default(board)) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """``atm10-server`` -> ``Atm10 Server``."""
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


def read_board_metadata(board: Optional[str] = None) -> dict:
    """``board.json`` merged over defaults, plus ``slug`` and ``db_path``. Never
    raises — a missing/malformed file yields the synthesized entry."""
    slug = _kb._slug_or_default(board)
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        # Project scope: new tasks inherit it (deterministic worktree + branch).
        "project_id": None,
        "created_at": None,
        "archived": False,
    }
    try:
        p = board_metadata_path(slug)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                # Never let the metadata file claim a different slug than
                # its directory — trust the filesystem.
                raw["slug"] = slug
                meta.update(raw)
    except (OSError, json.JSONDecodeError):
        pass
    meta["db_path"] = str(_kb.kanban_db_path(slug))
    return meta


def write_board_metadata(
    board: Optional[str], *, name: Optional[str] = None, description: Optional[str] = None,
    icon: Optional[str] = None, color: Optional[str] = None, archived: Optional[bool] = None,
    default_workdir: Optional[str] = None, project_id: Optional[str] = None,
) -> dict:
    """Create/update ``board.json``; unmentioned fields are preserved, ``created_at``
    set on first write. ``project_id``/``default_workdir``: ``None`` = unchanged,
    "" = clear (``project_id`` is not validated here)."""
    _kb._assert_not_delegated_child_mutation()
    slug = _kb._slug_or_default(board)
    meta = read_board_metadata(slug)
    # db_path is derived on every read; never persist it into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    for key, value in (("description", description), ("icon", icon), ("color", color)):
        if value is not None:
            meta[key] = str(value)
    if archived is not None:
        meta["archived"] = bool(archived)
    for key, value in (("default_workdir", default_workdir), ("project_id", project_id)):
        if value is not None:
            meta[key] = str(value) if value else None
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    meta["db_path"] = str(_kb.kanban_db_path(slug))
    return meta


def create_board(
    slug: str, *, name: Optional[str] = None, description: Optional[str] = None,
    icon: Optional[str] = None, color: Optional[str] = None, default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Create board dir + DB + metadata (``mkdir -p`` semantics: existing board returns its metadata)."""
    normed = _kb._require_slug(slug)
    # Explicit creation clears any archived tombstone at this slug (left by
    # remove_board(archive=True)) — otherwise _kb.init_db() below would rightly
    # refuse to recreate the archived board's DB (#43243).
    meta = write_board_metadata(
        normed, name=name, description=description, icon=icon, color=color,
        default_workdir=default_workdir, project_id=project_id, archived=False,
    )
    # Touch the DB so list_boards() sees it immediately.
    _kb.init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Metadata for every board: ``default`` first (always present), then
    ``boards/<slug>/`` dirs holding a ``kanban.db`` or ``board.json``, sorted."""
    entries = [read_board_metadata(_kb.DEFAULT_BOARD)]
    seen = {_kb.DEFAULT_BOARD}
    root = _kb.boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            try:
                normed = _kb._normalize_board_slug(child.name)  # skip junk dirs, don't raise
            except ValueError:
                continue
            if not normed or normed in seen or not _dir_holds_board(child):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


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
        prior_meta = read_board_metadata(normed)
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
        write_board_metadata(
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


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
