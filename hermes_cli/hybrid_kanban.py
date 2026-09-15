"""Canonical domain commands for the human + agent Hybrid Kanban.

Hybrid workspaces deliberately live alongside agentic tasks in the existing
per-board Kanban SQLite database.  A Hybrid column is an organizing list, not
an execution status: no command in this module mutates ``tasks`` or invokes
the agentic dispatcher.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db


class HybridKanbanError(ValueError):
    """A rejected Hybrid Kanban domain command."""


class HybridKanbanConflict(HybridKanbanError):
    """The caller acted on an obsolete card/column revision."""


def _now() -> int:
    return int(time.time())


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(10)}"


def _json(value: Optional[dict[str, Any]]) -> Optional[str]:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value else None


def _decode(value: Optional[str]) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    if "metadata" in value:
        value["metadata"] = _decode(value["metadata"])
    return value


def _activity(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    kind: str,
    actor_type: str,
    actor_id: Optional[str],
    session_id: Optional[str],
    source: Optional[str],
    card_id: Optional[str] = None,
    column_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    conn.execute(
        "INSERT INTO hybrid_activity (board_id, card_id, column_id, kind, actor_type, actor_id, session_id, source, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (board_id, card_id, column_id, kind, actor_type, actor_id, session_id, source, _json(payload), _now()),
    )


def _require_actor(actor_type: str) -> str:
    value = (actor_type or "").strip().lower()
    if value not in {"human", "agent", "system"}:
        raise HybridKanbanError("actor_type must be human, agent, or system")
    return value


def _require_board(conn: sqlite3.Connection, board_id: str, *, include_archived: bool = False) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM hybrid_boards WHERE id = ?", (board_id,)).fetchone()
    if row is None or (not include_archived and row["archived"]):
        raise HybridKanbanError("Hybrid board not found")
    return row


def _require_column(conn: sqlite3.Connection, column_id: str, *, include_archived: bool = False) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM hybrid_columns WHERE id = ?", (column_id,)).fetchone()
    if row is None or (not include_archived and row["archived"]):
        raise HybridKanbanError("Hybrid column not found")
    return row


def _require_card(conn: sqlite3.Connection, card_id: str, *, include_archived: bool = False) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM hybrid_cards WHERE id = ?", (card_id,)).fetchone()
    if row is None or (not include_archived and row["archived"]):
        raise HybridKanbanError("Hybrid card not found")
    return row


def _reindex(conn: sqlite3.Connection, table: str, scope_column: str, scope_id: str, ordered_ids: Iterable[str]) -> None:
    # Dense positions make reconstruction deterministic and permit a simple,
    # atomic repair whenever two old clients submit stale order intent.
    current_ids = _ordered_ids(conn, table, scope_column, scope_id)
    # SQLite enforces the unique rank constraints eagerly.  Move every live
    # sibling through a private negative namespace first, then publish the
    # dense canonical sequence.  This avoids transient collisions on swaps.
    for temporary, entity_id in enumerate(current_ids, start=1):
        conn.execute(
            f"UPDATE {table} SET position = ? WHERE id = ? AND {scope_column} = ?",
            (-1_000_000_000 - temporary, entity_id, scope_id),
        )
    for position, entity_id in enumerate(ordered_ids):
        conn.execute(f"UPDATE {table} SET position = ? WHERE id = ? AND {scope_column} = ?", (position, entity_id, scope_id))


def _ordered_ids(conn: sqlite3.Connection, table: str, scope_column: str, scope_id: str) -> list[str]:
    return [r["id"] for r in conn.execute(
        f"SELECT id FROM {table} WHERE {scope_column} = ? AND archived = 0 ORDER BY position, created_at, id", (scope_id,)
    )]


def _place(ids: list[str], entity_id: str, before_id: Optional[str], after_id: Optional[str]) -> list[str]:
    if before_id and after_id:
        raise HybridKanbanError("Specify before_id or after_id, not both")
    ids = [item for item in ids if item != entity_id]
    if before_id:
        if before_id not in ids:
            raise HybridKanbanError("before_id is not in the destination list")
        ids.insert(ids.index(before_id), entity_id)
    elif after_id:
        if after_id not in ids:
            raise HybridKanbanError("after_id is not in the destination list")
        ids.insert(ids.index(after_id) + 1, entity_id)
    else:
        ids.append(entity_id)
    return ids


def create_board(conn: sqlite3.Connection, *, name: str, description: str = "", actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise HybridKanbanError("Board name is required")
    actor_type = _require_actor(actor_type)
    board_id, now = _id("hb"), _now()
    with kanban_db.write_txn(conn):
        conn.execute("INSERT INTO hybrid_boards (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (board_id, name, description, now, now))
        _activity(conn, board_id=board_id, kind="board_created", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": name})
    return get_board(conn, board_id)


def list_boards(conn: sqlite3.Connection, *, include_archived: bool = False) -> list[dict[str, Any]]:
    query = "SELECT * FROM hybrid_boards" + ("" if include_archived else " WHERE archived = 0") + " ORDER BY updated_at DESC, id"
    return [_row(row) for row in conn.execute(query)]


def get_board(conn: sqlite3.Connection, board_id: str, *, include_archived: bool = False) -> dict[str, Any]:
    board = _row(_require_board(conn, board_id, include_archived=include_archived))
    where = "board_id = ?" + ("" if include_archived else " AND archived = 0")
    columns = [_row(r) for r in conn.execute(f"SELECT * FROM hybrid_columns WHERE {where} ORDER BY position, created_at, id", (board_id,))]
    for column in columns:
        column["cards"] = [_row(r) for r in conn.execute(
            "SELECT * FROM hybrid_cards WHERE column_id = ?" + ("" if include_archived else " AND archived = 0") + " ORDER BY position, created_at, id", (column["id"],)
        )]
    board["columns"] = columns
    return board


def create_column(conn: sqlite3.Connection, *, board_id: str, name: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise HybridKanbanError("Column name is required")
    actor_type = _require_actor(actor_type)
    column_id, now = _id("hc"), _now()
    with kanban_db.write_txn(conn):
        _require_board(conn, board_id)
        position = len(_ordered_ids(conn, "hybrid_columns", "board_id", board_id))
        conn.execute("INSERT INTO hybrid_columns (id, board_id, name, position, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", (column_id, board_id, name, position, now, now))
        _activity(conn, board_id=board_id, column_id=column_id, kind="column_created", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": name})
    return _row(_require_column(conn, column_id))


def move_column(conn: sqlite3.Connection, *, column_id: str, before_id: Optional[str] = None, after_id: Optional[str] = None, expected_revision: Optional[int] = None, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        column = _require_column(conn, column_id)
        if expected_revision is not None and column["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid column changed; refetch before moving")
        ids = _place(_ordered_ids(conn, "hybrid_columns", "board_id", column["board_id"]), column_id, before_id, after_id)
        _reindex(conn, "hybrid_columns", "board_id", column["board_id"], ids)
        now = _now()
        conn.execute("UPDATE hybrid_columns SET revision = revision + 1, updated_at = ? WHERE id = ?", (now, column_id))
        _activity(conn, board_id=column["board_id"], column_id=column_id, kind="column_moved", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"before_id": before_id, "after_id": after_id})
    return _row(_require_column(conn, column_id))


def create_card(conn: sqlite3.Connection, *, board_id: str, column_id: str, title: str, description: str = "", metadata: Optional[dict[str, Any]] = None, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise HybridKanbanError("Card title is required")
    actor_type = _require_actor(actor_type)
    card_id, now = _id("hcard"), _now()
    with kanban_db.write_txn(conn):
        _require_board(conn, board_id)
        column = _require_column(conn, column_id)
        if column["board_id"] != board_id:
            raise HybridKanbanError("Column belongs to another board")
        position = len(_ordered_ids(conn, "hybrid_cards", "column_id", column_id))
        conn.execute("INSERT INTO hybrid_cards (id, board_id, column_id, title, description, position, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (card_id, board_id, column_id, title, description, position, _json(metadata), now, now))
        _activity(conn, board_id=board_id, card_id=card_id, column_id=column_id, kind="card_created", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"title": title})
    return get_card(conn, card_id)


def get_card(conn: sqlite3.Connection, card_id: str, *, include_archived: bool = False) -> dict[str, Any]:
    card = _row(_require_card(conn, card_id, include_archived=include_archived))
    card["activity"] = [_row(r) for r in conn.execute("SELECT * FROM hybrid_activity WHERE board_id = ? AND (card_id = ? OR card_id IS NULL) ORDER BY id DESC", (card["board_id"], card_id))]
    for item in card["activity"]:
        item["payload"] = _decode(item.get("payload"))
    return card


def update_card(conn: sqlite3.Connection, *, card_id: str, title: Optional[str] = None, description: Optional[str] = None, expected_revision: Optional[int] = None, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        if expected_revision is not None and card["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid card changed; refetch before editing")
        fields: list[str] = []
        values: list[Any] = []
        if title is not None:
            if not title.strip():
                raise HybridKanbanError("Card title is required")
            fields.append("title = ?")
            values.append(title.strip())
        if description is not None:
            fields.append("description = ?")
            values.append(description)
        if fields:
            now = _now()
            fields.extend(["revision = revision + 1", "updated_at = ?"])
            values.extend([now, card_id])
            conn.execute(f"UPDATE hybrid_cards SET {', '.join(fields)} WHERE id = ?", values)
            _activity(conn, board_id=card["board_id"], card_id=card_id, column_id=card["column_id"], kind="card_updated", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"fields": ["title" if title is not None else None, "description" if description is not None else None]})
    return get_card(conn, card_id)


def move_card(conn: sqlite3.Connection, *, card_id: str, target_column_id: str, before_id: Optional[str] = None, after_id: Optional[str] = None, expected_revision: Optional[int] = None, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        target = _require_column(conn, target_column_id)
        if target["board_id"] != card["board_id"]:
            raise HybridKanbanError("Target column belongs to another board")
        if expected_revision is not None and card["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid card changed; refetch before moving")
        old_column_id = card["column_id"]
        destination_ids = _ordered_ids(conn, "hybrid_cards", "column_id", target_column_id)
        destination_ids = _place(destination_ids, card_id, before_id, after_id)
        if old_column_id != target_column_id:
            # Reserve a rank that cannot collide with the target column's
            # temporary namespace while the card changes membership.
            conn.execute("UPDATE hybrid_cards SET position = ? WHERE id = ?", (-2_000_000_000, card_id))
            conn.execute("UPDATE hybrid_cards SET column_id = ? WHERE id = ?", (target_column_id, card_id))
            _reindex(conn, "hybrid_cards", "column_id", old_column_id, _ordered_ids(conn, "hybrid_cards", "column_id", old_column_id))
        now = _now()
        conn.execute("UPDATE hybrid_cards SET revision = revision + 1, updated_at = ? WHERE id = ?", (now, card_id))
        _reindex(conn, "hybrid_cards", "column_id", target_column_id, destination_ids)
        _activity(conn, board_id=card["board_id"], card_id=card_id, column_id=target_column_id, kind="card_moved", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"from_column_id": old_column_id, "to_column_id": target_column_id, "before_id": before_id, "after_id": after_id})
    return get_card(conn, card_id)


def delete_card(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    actor_type: str = "human",
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = None,
) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id, include_archived=True)
        column_id = card["column_id"]
        board_id = card["board_id"]
        conn.execute("DELETE FROM hybrid_cards WHERE id = ?", (card_id,))
        remaining = _ordered_ids(conn, "hybrid_cards", "column_id", column_id)
        _reindex(conn, "hybrid_cards", "column_id", column_id, remaining)
        _activity(
            conn,
            board_id=board_id,
            card_id=card_id,
            column_id=column_id,
            kind="card_deleted",
            actor_type=actor_type,
            actor_id=actor_id,
            session_id=session_id,
            source=source,
            payload={"title": card["title"]},
        )
    return True


def delete_column(
    conn: sqlite3.Connection,
    *,
    column_id: str,
    actor_type: str = "human",
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = None,
) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        column = _require_column(conn, column_id, include_archived=True)
        board_id = column["board_id"]
        conn.execute("DELETE FROM hybrid_cards WHERE column_id = ?", (column_id,))
        conn.execute("DELETE FROM hybrid_columns WHERE id = ?", (column_id,))
        remaining = _ordered_ids(conn, "hybrid_columns", "board_id", board_id)
        _reindex(conn, "hybrid_columns", "board_id", board_id, remaining)
        _activity(
            conn,
            board_id=board_id,
            column_id=column_id,
            kind="column_deleted",
            actor_type=actor_type,
            actor_id=actor_id,
            session_id=session_id,
            source=source,
            payload={"name": column["name"]},
        )
    return True


def delete_board(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    actor_type: str = "human",
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = None,
) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        _require_board(conn, board_id, include_archived=True)
        conn.execute("DELETE FROM hybrid_cards WHERE board_id = ?", (board_id,))
        conn.execute("DELETE FROM hybrid_columns WHERE board_id = ?", (board_id,))
        conn.execute("DELETE FROM hybrid_activity WHERE board_id = ?", (board_id,))
        conn.execute("DELETE FROM hybrid_boards WHERE id = ?", (board_id,))
    return True


def get_board_activity(
    conn: sqlite3.Connection,
    board_id: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    _require_board(conn, board_id, include_archived=True)
    rows = conn.execute(
        "SELECT * FROM hybrid_activity WHERE board_id = ? ORDER BY id DESC LIMIT ?",
        (board_id, max(1, min(limit, 500))),
    ).fetchall()
    items = [_row(r) for r in rows]
    for item in items:
        item["payload"] = _decode(item.get("payload"))
    return items

