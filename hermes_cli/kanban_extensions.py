"""Deployment-owned extension columns for the kanban board (issue #109800).

A deployment can extend the ``tasks`` table with its own columns (e.g.
``tasks."Token消耗"``, created idempotently by a local migration tool outside
Hermes) and enforce board rules about them ("every active card must declare
whether it consumes cloud tokens"). The native ``kanban_create`` tool had no
way to see or supply such columns, so cards created by in-task calls silently
omitted them. This module owns the optional *overlay file* that declares those
columns to Hermes::

    <kanban_home>/kanban/extension-columns.json

    {
      "columns": [
        {
          "name": "Token消耗",             # tasks-table column, created by your migration
          "type": "string",                 # string|integer|number|boolean (default string)
          "required": true,                 # kanban_create rejects calls that omit it
          "description": "…",               # model-facing description in the tool schema
          "enum": ["是", "否", "未知"]       # optional closed value set
        }
      ]
    }

A bare JSON list is accepted as shorthand for ``{"columns": [...]}``.
``kanban_create`` exposes declared columns in its tool schema (required ones
in ``required``) and passes supplied values straight through to the INSERT;
``kanban_db.create_task(extra_fields=...)`` additionally validates every name
against the live table. The overlay is deployment-scoped — ``kanban_home`` is
shared across profiles by design, like the board itself.

Never raises for the caller: a missing or malformed overlay behaves exactly
like an undeclared one, with a warning — a hand-edited file must not take
card creation down (same contract as ``board.json``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXTENSION_COLUMNS_FILENAME = "extension-columns.json"

#: JSON Schema primitive advertised per declared column. SQLite itself is
#: dynamically typed; the type only tells the model what shape to send.
VALID_COLUMN_TYPES = ("string", "integer", "number", "boolean")


def extension_columns_path() -> Path:
    """``<kanban_home>/kanban/extension-columns.json``; absence = no extensions."""
    from hermes_cli import kanban_db as kb

    return kb.kanban_home() / "kanban" / EXTENSION_COLUMNS_FILENAME


def _normalize_column(raw: Any) -> Optional[dict[str, Any]]:
    """One overlay entry -> ``{name, type, required, description, enum}``; None when unusable."""
    if not isinstance(raw, dict):
        logger.warning("kanban extension columns: ignoring non-object entry %r", raw)
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        logger.warning(
            "kanban extension columns: entry needs an exact, non-blank 'name' (got %r)", name)
        return None
    column_type = raw.get("type", "string")
    if column_type not in VALID_COLUMN_TYPES:
        logger.warning(
            "kanban extension columns: %r declares unknown type %r; using 'string'",
            name, column_type)
        column_type = "string"
    enum = raw.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        logger.warning(
            "kanban extension columns: %r declares an invalid 'enum' %r; ignoring it",
            name, enum)
        enum = None
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        description = (
            f"Deployment extension column {name!r} on the tasks table (declared in "
            f"{EXTENSION_COLUMNS_FILENAME}); the value is written to the created "
            f"card's column.")
    return {
        "name": name,
        "type": column_type,
        "required": bool(raw.get("required", False)),
        "description": description.strip(),
        "enum": list(enum) if enum else None,
    }


def load_extension_columns() -> list[dict[str, Any]]:
    """Declared extension columns in overlay order; ``[]`` when undeclared.

    Malformed input never raises — see the module docstring. Entries without a
    usable ``name`` are skipped individually, so one bad row cannot hide the
    rest of a hand-edited overlay.
    """
    path = extension_columns_path()
    try:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "kanban extension columns: cannot read %s (%s); treating as undeclared", path, exc)
        return []
    if isinstance(raw, dict):
        raw = raw.get("columns")
    if not isinstance(raw, list):
        logger.warning(
            "kanban extension columns: %s must hold {'columns': [...]} (or a list); "
            "treating as undeclared", path)
        return []
    columns: dict[str, dict[str, Any]] = {}
    for entry in raw:
        column = _normalize_column(entry)
        if column is None:
            continue
        if column["name"] in columns:
            logger.warning(
                "kanban extension columns: duplicate name %r; keeping the first entry",
                column["name"])
            continue
        columns[column["name"]] = column
    return list(columns.values())
