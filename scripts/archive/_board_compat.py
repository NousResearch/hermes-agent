#!/usr/bin/env python3
"""Board identity compatibility layer for operational cron scripts.

The KENSEI board registry was reorganised: three legacy board slugs were
retired in favour of canonical successors:

    legacy ``default``     → canonical ``core``        (DB moved from
                             ``<root>/kanban.db`` to
                             ``<root>/kanban/boards/core/kanban.db``)
    legacy ``ops``         → canonical ``security-ops``
    legacy ``content-lead``→ canonical ``content``

The seven operational scripts covered by W1 Batch 1 hardcode the legacy
slugs (and their DB paths).  This module lets those scripts resolve a
legacy slug to the *current* DB path **reversibly**: if the legacy board's
DB still exists on disk the legacy path is returned (migration has not
happened yet on that host); otherwise the canonical successor's path is
resolved via :func:`hermes_cli.kanban_db.kanban_db_path`.

Constraints honoured (from the controller-verified scope):

* Reuses the existing canonical board resolution — no parallel registry.
* Does **not** replace semantic domain labels/keywords.  Routing-key maps
  and human text keep their original slugs; only the DB *path identity*
  is resolved through this layer.
* Reversible: works whether or not the migration has happened on the host.
"""

from __future__ import annotations

from pathlib import Path

# Legacy slug → canonical successor.  Boards not listed here (apps, research,
# kensei-rebuild …) are unchanged and pass through verbatim.
LEGACY_BOARD_MAP: dict[str, str] = {
    "default": "core",
    "ops": "security-ops",
    "content-lead": "content",
}


def _kanban_db():
    """Import kanban_db lazily so the module is import-safe without the repo
    on ``sys.path`` (unit tests stub it; scripts that ``sys.path.insert`` the
    repo root get the real one)."""
    from hermes_cli import kanban_db as kb
    return kb


def canonical_board_slug(slug: str) -> str:
    """Return the canonical successor for a (possibly legacy) board slug.

    Unchanged boards pass through unchanged.
    """
    return LEGACY_BOARD_MAP.get(slug, slug)


def resolve_board_db(slug: str) -> Path:
    """Resolve a board slug to its ``kanban.db`` path with reversible fallback.

    Resolution order:

    1. If *slug* is a retired legacy slug **and** its legacy DB still exists
       on disk → return the legacy path (migration not yet done on this host).
    2. Otherwise → resolve the canonical successor via
       :func:`hermes_cli.kanban_db.kanban_db_path`.

    Unchanged slugs (apps, research, …) always go straight to the canonical
    resolver.
    """
    kb = _kanban_db()
    canonical = canonical_board_slug(slug)
    if canonical == slug:
        # Not a retired slug — resolve directly.
        return kb.kanban_db_path(slug)
    # Retired slug: prefer legacy DB if it still exists (reversible).
    legacy_path = kb.kanban_db_path(slug)
    if legacy_path.exists():
        return legacy_path
    # Legacy board gone — fall back to canonical successor.
    return kb.kanban_db_path(canonical)


def resolve_board_db_str(slug: str) -> str:
    """Same as :func:`resolve_board_db` but returns ``str`` for scripts that
    build paths with ``os.path.join``."""
    return str(resolve_board_db(slug))


def canonical_board_list(legacy_boards: list[str]) -> list[str]:
    """Map a list of (possibly legacy) board slugs to canonical slugs,
    deduplicated, preserving first-seen order.

    Used by scripts that enumerate boards via the CLI (``hermes kanban
    --board <slug>``) where the slug must be an existing board.
    """
    seen: set[str] = set()
    out: list[str] = []
    for slug in legacy_boards:
        c = canonical_board_slug(slug)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_board_db_map(legacy_slugs: list[str]) -> dict[str, Path]:
    """Build a ``{legacy_slug: db_path}`` mapping for scripts that iterate a
    board dict and open SQLite connections directly.

    The *keys* are the original (possibly legacy) slugs — semantic labels are
    preserved.  The *values* are resolved through :func:`resolve_board_db`.
    """
    return {slug: resolve_board_db(slug) for slug in legacy_slugs}
