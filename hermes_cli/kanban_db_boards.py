"""Board metadata (``board.json``) and board lifecycle management for the Kanban DB:
read/write of per-board display metadata, board creation/discovery/archival, and the
``board.json``-as-identity-marker rule.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.

Dispatch admission (card t_17c9c847) lives here too: ``board_dispatch_enabled`` is the
one chokepoint that decides whether the dispatcher may serve a board, and
``list_dispatch_boards`` is the dispatcher's OWN enumeration — ``list_boards`` stays the
full inventory for the CLI and the dashboard, so an estate board is still VISIBLE on the
board list, it is simply never spawned from.
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


# --------------------------------------------------------------------------- #
# Dispatch admission (card t_17c9c847)
# --------------------------------------------------------------------------- #


def _board_dispatch_flag(value: Any = True) -> bool:
    """Coerce a ``board.json`` ``dispatch`` value to the admission bool.

    Absent, ``None`` or an unrecognised value means ADMITTED (``True``), so every
    board that predates the key keeps dispatching. Only an explicit false-y value
    (``false``, ``0``, ``"off"`` ...) takes a board out of the dispatch set.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    if isinstance(value, (int, float)):
        return bool(value)
    return True


def board_is_archived_stub(slug: str) -> bool:
    """A directory that is NOT a live board, but whose slug already has an ``_archived`` copy.

    The measured resurrection (card t_17c9c847): a live worker's env pins its board's
    store path, so the moment that board is archived the worker's next ``connect()`` (its
    own auto-heartbeat, a tool call) mints an EMPTY store back at the archived slug's
    directory. Two shapes are that mint, never a board somebody created:

    * a directory with no ``board.json`` at all (``create_board`` always writes one), and
    * the retired board's own tombstone — ``remove_board(archive=True)`` leaves an
      ``archived`` ``board.json`` behind — once a stale read path has minted a store
      beside it. A tombstone with no store is still the board's own record and stays
      listed; the tombstone PLUS a freshly minted ``kanban.db`` is the stub.

    An explicit ``create_board`` writes ``archived=False``, so a sanctioned re-create is
    never a stub.
    """
    try:
        meta_path = board_metadata_path(slug)
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return False
            if not (isinstance(raw, dict) and raw.get("archived")):
                return False  # a live board
            if not (_kb.board_dir(slug) / "kanban.db").exists():
                return False  # the ordinary tombstone: still this board's record
    except Exception:
        return False
    root = _kb.boards_root() / "_archived"
    if not root.is_dir():
        return False
    prefix = f"{slug}-"
    try:
        for child in root.iterdir():
            if not child.name.startswith(prefix):
                continue
            # ``<slug>-<epoch>`` (and ``<slug>-<epoch>-<n>`` on a rapid re-archive).
            if child.name[len(prefix):].split("-")[0].isdigit():
                return True
    except OSError:
        return False
    return False


def board_dispatch_enabled(board: Optional[str] = None) -> bool:
    """May the dispatcher serve ``board``? ``False`` for an estate/scratch board.

    The one chokepoint read by the dispatcher's board enumeration
    (:func:`list_dispatch_boards`) AND by its per-tick spawn guard
    (``kanban_db_dispatch.dispatch_once``), so an estate board is safe by construction —
    including from a CLI ``hermes kanban --board <estate> dispatch`` that bypasses
    enumeration entirely (card t_17c9c847).

    The ``dispatch`` key defaults to ``True`` AT READ TIME and is never materialized into
    ``board.json`` by an unrelated write, so every board that predates the key keeps
    dispatching. Fails CLOSED for a resurrected archived stub: a minted directory is not a
    board, so re-opening an archived slug's path cannot re-admit it.
    """
    slug = _kb._slug_or_default(board)
    if board_is_archived_stub(slug):
        return False
    try:
        return _board_dispatch_flag(read_board_metadata(slug).get("dispatch", True))
    except Exception:
        return True


def read_board_metadata(board: Optional[str] = None) -> dict:
    """``board.json`` merged over defaults, plus ``slug`` and ``db_path``. Never
    raises — a missing/malformed file yields the synthesized entry. The ``dispatch``
    default is applied at read time by :func:`board_dispatch_enabled`, not here: an
    unrelated write must leave a ``board.json`` that never touched the key alone."""
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
    dispatch: Optional[bool] = None,
) -> dict:
    """Create/update ``board.json``; unmentioned fields are preserved, ``created_at``
    set on first write. ``project_id``/``default_workdir``: ``None`` = unchanged,
    "" = clear (``project_id`` is not validated here). ``dispatch``: ``None`` =
    unchanged — the synthesized ``True`` default is applied at READ time by
    :func:`board_dispatch_enabled`, so a board that never touched the key keeps its
    ``board.json`` byte-identical; ``False`` marks an estate/scratch board the dispatcher
    must not serve."""
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
    if dispatch is not None:
        meta["dispatch"] = bool(dispatch)
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
    project_id: Optional[str] = None, dispatch: Optional[bool] = None,
) -> dict:
    """Create board dir + DB + metadata (``mkdir -p`` semantics: existing board returns its metadata).

    ``dispatch=False`` births an ESTATE/scratch board the dispatcher never serves (card
    t_17c9c847): the flag is written into ``board.json`` at creation, so the board is
    undispatchable by construction rather than by a teardown somebody has to remember.
    """
    normed = _kb._require_slug(slug)
    # Explicit creation clears any archived tombstone at this slug (left by
    # remove_board(archive=True)) — otherwise _kb.init_db() below would rightly
    # refuse to recreate the archived board's DB (#43243).
    meta = write_board_metadata(
        normed, name=name, description=description, icon=icon, color=color,
        default_workdir=default_workdir, project_id=project_id, archived=False,
        dispatch=dispatch,
    )
    # Touch the DB so list_boards() sees it immediately.
    _kb.init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Metadata for every board: ``default`` first (always present), then
    ``boards/<slug>/`` dirs holding a ``kanban.db`` or ``board.json``, sorted.

    A resurrected archived STUB — a minted directory whose slug already has an
    ``_archived`` copy — is NOT a board and is skipped here, so the phantom a pinned
    worker re-creates cannot re-enter any enumeration
    (:func:`board_is_archived_stub`, card t_17c9c847)."""
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
            if board_is_archived_stub(normed):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def list_dispatch_boards() -> list[dict]:
    """Live boards the DISPATCHER may serve: non-archived AND dispatch-enabled.

    The dispatcher's own board enumeration (card t_17c9c847). :func:`list_boards` stays
    the full inventory for the CLI and the dashboard, so an estate board is still VISIBLE
    on the board list — it is simply never spawned from. Both halves are load-bearing: the
    ``dispatch`` flag closes the gap the archived-only filter left (``"each board has its
    own ... dispatcher loop"``), and :func:`board_dispatch_enabled` fails closed for a
    resurrected archived stub.
    """
    return [
        meta for meta in list_boards(include_archived=False)
        if board_dispatch_enabled(meta.get("slug") or _kb.DEFAULT_BOARD)
    ]


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
