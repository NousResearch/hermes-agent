"""Board metadata (``board.json``) and board lifecycle management for the Kanban DB:
read/write of per-board display metadata, board creation/discovery/archival/restore,
and the ``board.json``-as-identity-marker rule.

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


def _dir_holds_board(d: Path) -> bool:
    # ``board.json`` is the identity marker: archive/hard-delete both leave the
    # directory without it, and a stale ``connect(board=slug)`` used to leave a
    # ``kanban.db``-only stub that resurfaced in the board list as an empty
    # active board (#43243). Discovery must therefore require the metadata
    # file; a DB-only directory is a stub to ignore, never a board.
    return (d / "board.json").exists()


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


def _archive_name_matches(name: str, normed: str) -> bool:
    """True for ``remove_board`` archive dirs: ``<slug>-<unix>`` or ``<slug>-<unix>-<n>``.

    Slugs may themselves contain hyphens, so a bare ``startswith(slug + "-")``
    would treat ``foo-bar-<ts>`` as an archive of ``foo`` and move another
    board's data.
    """
    prefix = f"{normed}-"
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix):]
    head, sep, tail = rest.partition("-")
    if not head.isdigit():
        return False
    return not sep or tail.isdigit()


def _looks_like_archived_board(path: Path) -> bool:
    return (path / "kanban.db").exists() or (path / "board.json").exists()


def list_archived_board_dirs(slug: str | None = None) -> list[Path]:
    """Return archived board directories under ``boards/_archived/``.

    When ``slug`` is given, only ``remove_board`` names for that slug are
    returned. Newest (lexicographically last / timestamped last) is first.
    """
    archive_root = _kb.boards_root() / "_archived"
    if not archive_root.is_dir():
        return []
    normed = None
    if slug is not None:
        normed = _kb._normalize_board_slug(slug)
        if not normed:
            return []
    entries: list[Path] = []
    for child in archive_root.iterdir():
        if not child.is_dir() or not _looks_like_archived_board(child):
            continue
        if normed is not None and not _archive_name_matches(child.name, normed):
            continue
        entries.append(child)
    entries.sort(key=lambda p: p.name, reverse=True)
    return entries


def _resolve_restore_source(normed: str, archive_path: str | Path | None) -> Path:
    """Newest matching archive, or ``archive_path`` confined under ``_archived``."""
    archive_root = (_kb.boards_root() / "_archived").resolve()
    if archive_path is None:
        candidates = list_archived_board_dirs(normed)
        if not candidates:
            raise ValueError(
                f"no archived board found for slug {normed!r} under "
                f"{_kb.boards_root() / '_archived'}"
            )
        return candidates[0]
    try:
        src = Path(archive_path).expanduser().resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"invalid archive path: {archive_path}") from exc
    try:
        src.relative_to(archive_root)
    except ValueError as exc:
        raise ValueError(
            f"archive path must be under {archive_root}; got {src}"
        ) from exc
    if not src.is_dir():
        raise ValueError(f"archive path does not exist: {src}")
    if not _archive_name_matches(src.name, normed):
        raise ValueError(
            f"archive path {src.name!r} does not match slug {normed!r} "
            f"(expected a directory named '{normed}-<timestamp>')"
        )
    if not _looks_like_archived_board(src):
        raise ValueError(
            f"archive path {src} does not look like a board "
            f"(missing kanban.db / board.json)"
        )
    return src


def _discard_live_init_cache(live: Path) -> None:
    live_db = live / "kanban.db"
    try:
        _kb._INITIALIZED_PATHS.discard(str(live_db.resolve()))
    except OSError:
        pass
    _kb._INITIALIZED_PATHS.discard(str(live_db))


def _is_archived_tombstone_dir(live: Path) -> bool:
    """True when ``live`` is only the ``archived: true`` marker ``board.json``."""
    try:
        children = list(live.iterdir())
    except OSError:
        return False
    if len(children) != 1 or children[0].name != "board.json":
        return False
    try:
        raw = json.loads(children[0].read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("archived") is True


def _park_tombstone(live: Path) -> Path:
    parked = live.with_name(f".{live.name}.restore-tombstone")
    if parked.exists():
        parked = live.with_name(f".{live.name}.restore-tombstone-{time.time_ns()}")
    live.rename(parked)
    return parked


def _live_collision(normed: str, live: Path) -> ValueError:
    return ValueError(
        f"board {normed!r} already exists at {live}. "
        f"Remove or rename it before restoring an archive."
    )


def _clear_live_for_restore(normed: str, live: Path) -> Optional[Path]:
    """Make ``live`` absent without removing user board data.

    An empty directory is removed. The post-archive tombstone is renamed
    aside and returned so the caller can delete it only after the archive
    itself is in place. A database, a non-archived ``board.json``, or any
    other contents are left untouched.
    """
    if not live.exists():
        return None
    if not live.is_dir():
        raise _live_collision(normed, live)
    if not any(live.iterdir()):
        live.rmdir()
        return None
    if _is_archived_tombstone_dir(live):
        return _park_tombstone(live)
    raise _live_collision(normed, live)


def _restore_parked_tombstone(parked: Optional[Path], live: Path) -> None:
    """Put the marker back if the rename failed and ``live`` is still absent.

    When ``live`` was recreated, drop only the marker we parked — never the
    live directory or the archive.
    """
    if parked is None or not parked.exists():
        return
    if not live.exists():
        parked.rename(live)
        return
    if not _is_archived_tombstone_dir(parked):
        return
    parked.joinpath("board.json").unlink(missing_ok=True)
    try:
        parked.rmdir()
    except OSError:
        pass


def restore_board(slug: str, *, archive_path: str | Path | None = None) -> dict:
    """Restore a board archived by :func:`remove_board`.

    Moves the newest (or explicitly named) ``boards/_archived/<slug>-<ts>``
    directory back to ``boards/<slug>/``. The archived-tombstone directory
    ``remove_board`` leaves behind is moved aside only when it contains
    nothing except that marker, and deleted only after the archive rename
    succeeds. A live board (database or any other files) is refused and
    left in place.

    Raises :class:`ValueError` when:
    - the slug is invalid / is ``default``
    - a live board already exists at ``boards/<slug>/``
    - no matching archive directory is found
    - ``archive_path`` is outside ``boards/_archived/``
    """
    _kb._assert_not_delegated_child_mutation()
    normed = _kb._require_slug(slug)
    if normed == _kb.DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be restored from _archived")

    src = _resolve_restore_source(normed, archive_path)
    live = _kb.board_dir(normed)
    _discard_live_init_cache(live)
    parked = _clear_live_for_restore(normed, live)
    live.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(live)
    except OSError as exc:
        _restore_parked_tombstone(parked, live)
        if live.exists():
            raise _live_collision(normed, live) from exc
        raise ValueError(
            f"failed to restore board {normed!r} from {src} to {live}: {exc}"
        ) from exc
    _restore_parked_tombstone(parked, live)
    return {
        "slug": normed,
        "action": "restored",
        "from_path": str(src),
        "new_path": str(live),
    }


def has_archived_board(slug: str) -> bool:
    """True if ``boards/_archived/`` holds at least one directory for ``slug``."""
    return bool(list_archived_board_dirs(slug))


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
