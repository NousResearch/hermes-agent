"""Per-board kanban setting resolution (``kanban.boards.<slug>.<key>``).

A machine usually runs several kanban boards against one home; a single global
``kanban.orchestrator_profile`` / ``kanban.default_assignee`` cannot express
per-board routing. This module resolves those knobs board-first::

    board override  ->  caller-supplied global  ->  existing built-in fallback

It is a pure reader: callers pass the already-loaded ``kanban`` config section,
so nothing here imports ``hermes_cli.config`` (keeps it unit-testable without a
filesystem). An unset board key leaves every caller's current behaviour
byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# v1 scope: only these two knobs are board-overridable. ``boards.<slug>`` is a
# free-form mapping, so later keys can join this tuple without a schema change.
KANBAN_BOARD_SETTING_KEYS = ("orchestrator_profile", "default_assignee")

# ``HERMES_KANBAN_DB`` pin-collapse warning fires once per slug per process.
_PIN_SUPPRESSED_WARNED: set[str] = set()

# Hand-edited ``kanban.boards`` keys that normalize to nothing, warned once each.
_MALFORMED_BOARD_KEY_WARNED: set[object] = set()


def normalize_board_slug(slug: object) -> Optional[str]:
    """``kanban_db._normalize_board_slug`` (lowercase + validate); None on invalid.

    A malformed slug means "no board scope", never an exception: config is
    declarative and a hand-edited key must not crash a dispatch tick.
    """
    if slug is None:
        return None
    try:
        from hermes_cli import kanban_db

        return kanban_db._normalize_board_slug(slug)
    except (ValueError, ImportError, AttributeError):
        return None


def _clean_str(value: object) -> Optional[str]:
    """A stripped non-empty ``str`` or None; any other type is "unset"."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _ci_lookup(mapping: dict, wanted: str) -> object:
    """Case-insensitive key lookup; ``None`` when absent."""
    lowered = wanted.strip().lower()
    for key, value in mapping.items():
        if isinstance(key, str) and key.strip().lower() == lowered:
            return value
    return None


def _warn_malformed_board_keys(boards: dict) -> None:
    """Warn once per configured board key that can never name a board.

    A hand-written ``kanban.boards.bad/slug`` is silently dead weight today; the
    docs promise a warning, so surface each malformed key exactly once per
    process (mirrors ``_PIN_SUPPRESSED_WARNED``). Valid keys are untouched.
    """
    for key in boards:
        valid = isinstance(key, str) and normalize_board_slug(key) is not None
        if valid or key in _MALFORMED_BOARD_KEY_WARNED:
            continue
        _MALFORMED_BOARD_KEY_WARNED.add(key)
        logger.warning("kanban.boards: ignoring malformed board key %r", key)


def board_override(kanban_cfg: object, key: str, board: Optional[str]) -> Optional[str]:
    """``kanban.boards.<slug>.<key>`` as a stripped string, else None.

    Board keys match case-insensitively (a hand-written ``TSA-Mgmt`` still
    serves slug ``tsa-mgmt``), as does the setting key. Only ``str`` values are
    accepted; an empty string means "not set" and falls through to the global.
    """
    boards = kanban_cfg.get("boards") if isinstance(kanban_cfg, dict) else None
    if not isinstance(boards, dict) or not boards:
        return None
    _warn_malformed_board_keys(boards)
    slug = normalize_board_slug(board)
    if slug is None:
        if isinstance(board, str) and board.strip():
            logger.warning("kanban.boards: ignoring invalid board slug %r", board)
        return None
    entry = _ci_lookup(boards, slug)
    if not isinstance(entry, dict) or not entry:
        return None
    raw = _ci_lookup(entry, key)
    cleaned = _clean_str(raw)
    if cleaned is None and raw is not None and not isinstance(raw, str):
        logger.warning(
            "kanban.boards.%s.%s: expected a string, got %s; ignoring",
            slug, key, type(raw).__name__,
        )
    return cleaned


def board_pin_suppressed(board: Optional[str]) -> bool:
    """Whether ``HERMES_KANBAN_DB`` collapses ``board`` onto the pinned file.

    In that topology several enumerated slugs resolve to ONE physical database,
    so a per-slug override would route one board's config onto another board's
    cards (design R1). Conservative resolution: treat the board as unpinned
    (global only) and warn once per slug.
    """
    pin = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if not pin or not board:
        return False
    try:
        from hermes_cli import kanban_db

        target = kanban_db.kanban_db_path(board=board)
    except (ImportError, AttributeError, OSError, ValueError):
        return False
    try:
        if Path(target).expanduser().resolve() != Path(pin).expanduser().resolve():
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    if board not in _PIN_SUPPRESSED_WARNED:
        _PIN_SUPPRESSED_WARNED.add(board)
        logger.warning(
            "kanban.boards.%s: ignored because HERMES_KANBAN_DB pins every board "
            "to one database; using the global setting", board,
        )
    return True


def effective_setting(
    kanban_cfg: object,
    key: str,
    board: Optional[str],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Board override for ``key``, else ``fallback`` (the caller's global value).

    ``fallback`` is the global the CALLER already resolved (e.g. the value it
    passes to ``dispatch_once``). This function never reaches into
    ``kanban_cfg[key]`` itself, so a caller that deliberately passes nothing
    (the standalone daemon) keeps its "no fallback" semantics.
    """
    if board is not None and not board_pin_suppressed(board):
        override = board_override(kanban_cfg, key, board)
        if override is not None:
            return override
    return _clean_str(fallback)
