"""Completion-time prose-reference scan for the Kanban DB: a ``t_<hex>`` id named in
a completion handoff is a phantom only when NO reachable board carries it.

The scan is board-aware. A completion summary may legitimately name a card on a
SIBLING board — a cross-board handoff is normal on a multi-board host — so the
boards consulted are the completing connection's own board (read off the store
path, never the ambient current board, which a ``--board`` override or a worker's
injected pin may disagree with) plus every other registered board; an id a sibling
carries is reported as ``matched_elsewhere`` in the durable payload, never recorded
as a fabrication.

Split out of ``hermes_cli.kanban_db``, which keeps the completion path that calls
:func:`_flag_phantom_prose_refs`; origin-resident helpers are reached late-bound via
``_kb`` (import-cycle breaking) so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

# Matches ``kanban_create`` (12 hex) and ``_new_task_id`` (8 hex) ids; 8+ for forward compat.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def board_for_store_path(path: Any) -> Optional[str]:
    """The slug of the board whose canonical store IS ``path``, else ``None``.

    The single layout->board rule (``<root>/kanban.db`` = ``default``, else
    ``<root>/kanban/boards/<slug>/<leaf>``), the inverse of
    :func:`hermes_cli.kanban_db.kanban_db_path`. A completion-time prose scan uses
    it to learn which board its OWN connection belongs to — never the ambient
    current board, which a ``--board`` override or a worker's injected pin may
    disagree with.
    """
    if not path:
        return None
    try:
        resolved = Path(path).resolve()
        if resolved == (_kb.kanban_home() / "kanban.db").resolve():
            return _kb.DEFAULT_BOARD
        rel = resolved.relative_to(_kb.boards_root().resolve())
    except (OSError, ValueError):
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def board_for_connection(conn: sqlite3.Connection) -> Optional[str]:
    """The slug of the board ``conn`` is open against, or ``None`` when it cannot be told.

    The slug is read off the open file's path, never off the ambient current board:
    ``HERMES_KANBAN_DB`` can pin a database no board claims (workers are spawned
    with exactly that), and a ``--board`` override means the caller is not on
    ``get_current_board()`` either.
    """
    path = ""
    for _seq, name, file in conn.execute("PRAGMA database_list"):
        if name == "main" and file:
            path = file
    return board_for_store_path(path)


def _store_path_for_slug(slug: str, default_parts: tuple[str, ...], leaf: str) -> Path:
    """The one layout rule for a board's canonical path, by slug alone.

    Legacy ``<root>/<default_parts>`` for the ``default`` board, else
    ``board_dir(slug)/leaf``. Takes an already-normalised slug and asks no
    questions, deliberately: unlike ``kanban_db._board_path`` it consults no env pin
    and no fence, which is exactly the property a read-only sibling scan wants — it
    must open the SIBLING board's own store, never the caller's pinned one.
    """
    if slug == _kb.DEFAULT_BOARD:
        return _kb.kanban_home().joinpath(*default_parts)
    return _kb.board_dir(slug) / leaf


def _prose_ref_search_boards(conn: sqlite3.Connection) -> list[str]:
    """Board slugs a completion-time prose scan searched, in a deterministic order.

    The board ``conn`` is open against comes first (read off the file's path via
    :func:`board_for_connection` — never the ambient current board, which a
    ``--board`` override or a worker's injected pin may disagree with), then every
    other registered board by slug.
    """
    here = board_for_connection(conn)
    slugs: list[str] = [here] if here else []
    for meta in _kb.list_boards(include_archived=False):
        slug = meta.get("slug")
        if slug and slug not in slugs:
            slugs.append(slug)
    return slugs


def _missing_task_ids_everywhere(
    conn: sqlite3.Connection, ids: Iterable[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Which of ``ids`` no REACHABLE board carries, and which a SIBLING board does.

    A completion summary may legitimately name a card on another board — a
    cross-board handoff is normal — so a real card must never be recorded as a
    fabrication. Measured 2026-09-29 (card t_89626ce6): a verified ops-board child
    was written to ``suspected_hallucinated_references`` because the scan only knew
    the completing board.

    Sibling stores are opened READ-ONLY and only when the store file already
    exists, so a scan can never create a stray ``<slug>.db``.

    Returns ``(phantoms, boards_searched, matched)``: ids absent from EVERY board,
    the slugs actually consulted (for the durable payload), and ``{id: slug}`` for
    ids a sibling board carries.
    """
    ordered = list(dict.fromkeys(ids))
    missing = _kb._missing_task_ids(conn, ordered)
    here = board_for_connection(conn)
    # ``here`` is the connection already in hand; only genuine siblings are opened below.
    boards_searched = [here] if here else []
    if not missing:
        return [], boards_searched, {}

    matched: dict[str, str] = {}
    for slug in _prose_ref_search_boards(conn):
        if not missing or slug == here:
            continue
        store = _store_path_for_slug(slug, ("kanban.db",), "kanban.db")
        try:
            if not store.is_file():
                continue
            sibling = sqlite3.connect(store.resolve().as_uri() + "?mode=ro", uri=True)
        except (OSError, sqlite3.Error):
            continue
        try:
            sibling.row_factory = sqlite3.Row
            sibling.text_factory = _kb._lossy_text
            boards_searched.append(slug)
            still = _kb._missing_task_ids(sibling, missing)
            carried = set(missing) - set(still)
            for tid in carried:
                matched[tid] = slug
            missing = still
        except sqlite3.Error:
            continue
        finally:
            sibling.close()
    return missing, boards_searched, matched


def _flag_phantom_prose_refs(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int],
    summary: Optional[str], result: Optional[str], verified_cards: list[str],
) -> None:
    """Advisory post-commit scan of summary+result for ``t_<hex>`` references that
    NO reachable board carries; emits ``suspected_hallucinated_references`` in its
    own txn so the completion is already durable. Never blocks.

    The scan is board-aware: an id a SIBLING board carries is a cross-board
    reference, not a fabrication, so it is not flagged (measured 2026-09-29, card
    t_89626ce6). The payload names the boards searched, so an operator reading the
    event can tell a real cross-board card from an invented id without re-deriving
    the resolver."""
    scan_text = " ".join(filter(None, [summary, result]))
    if not scan_text:
        return
    verified = set(verified_cards)
    ids = [i for i in dict.fromkeys(_TASK_ID_PROSE_RE.findall(scan_text)) if i not in verified]
    if not ids:
        return
    phantom_refs, boards_searched, matched = _missing_task_ids_everywhere(conn, ids)
    if phantom_refs:
        payload: dict[str, Any] = {
            "phantom_refs": phantom_refs,
            "source": "completion_summary",
            "boards_searched": boards_searched,
        }
        if matched:
            payload["matched_elsewhere"] = matched
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "suspected_hallucinated_references",
                payload, run_id=run_id,
            )


# Tail import, as in the other ``kanban_db_*`` siblings: ``kanban_db`` imports this
# module after its own definitions, so binding the facade here breaks the cycle
# while ``kanban_db.<name>`` stays the seam every monkeypatch targets.
from hermes_cli import kanban_db as _kb
