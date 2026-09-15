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


DELEGATION_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


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


def _decode_list(value: Optional[str]) -> list[str]:
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if item]
    return []


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


def _require_checklist(conn: sqlite3.Connection, checklist_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM hybrid_checklists WHERE id = ?", (checklist_id,)).fetchone()
    if row is None:
        raise HybridKanbanError("Hybrid checklist not found")
    _require_card(conn, row["card_id"])
    return row


def _require_checklist_item(conn: sqlite3.Connection, item_id: str) -> tuple[sqlite3.Row, sqlite3.Row]:
    item = conn.execute("SELECT * FROM hybrid_checklist_items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise HybridKanbanError("Hybrid checklist item not found")
    return item, _require_checklist(conn, item["checklist_id"])


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


def _delegation_state(task_status: Optional[str]) -> str:
    if task_status == "done":
        return "completed"
    if task_status == "archived":
        return "cancelled"
    if task_status in {"blocked", "review"}:
        return "waiting"
    if task_status in {"ready", "todo", "triage", "scheduled"}:
        return "queued"
    if task_status == "running":
        return "running"
    return "failed"


def _latest_task_projection(conn: sqlite3.Connection, task_id: str) -> tuple[Optional[str], Optional[str], list[str]]:
    """Read only compact result metadata from the canonical task/run owner."""
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        return "failed", "The linked Agent Task no longer exists.", []
    run = conn.execute(
        "SELECT summary, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    metadata: dict[str, Any] = {}
    run_summary: Optional[str] = None
    if run is not None:
        run_summary = run["summary"]
        try:
            decoded = json.loads(run["metadata"] or "{}")
            if isinstance(decoded, dict):
                metadata = decoded
        except (TypeError, ValueError):
            metadata = {}
    result_ref = metadata.get("result_ref")
    if not isinstance(result_ref, str) or not result_ref.strip():
        result_ref = None
    evidence = metadata.get("evidence_refs")
    if evidence is None:
        evidence = metadata.get("evidence")
    evidence_refs = [str(item) for item in evidence if item] if isinstance(evidence, list) else []
    summary = run_summary or (task.result if task.status == "done" else None)
    if summary is not None:
        summary = str(summary)[:2_000]
    return _delegation_state(task.status), summary, evidence_refs


def _sync_card_delegations_in_txn(conn: sqlite3.Connection, card_id: str) -> None:
    """Persist terminal/progress projections without copying task payloads."""
    rows = conn.execute(
        "SELECT * FROM hybrid_card_delegations WHERE human_card_id = ? ORDER BY attempt",
        (card_id,),
    ).fetchall()
    for row in rows:
        state, summary, evidence_refs = _latest_task_projection(conn, row["agent_task_id"])
        # An explicit bridge failure/cancellation is authoritative even though
        # its canonical task is archived as the safe terminal operation.
        if row["state"] in {"failed", "cancelled"}:
            state = row["state"]
        task = kanban_db.get_task(conn, row["agent_task_id"])
        result_ref = None
        if task is not None:
            run = conn.execute(
                "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (row["agent_task_id"],),
            ).fetchone()
            if run is not None:
                try:
                    metadata = json.loads(run["metadata"] or "{}")
                    if isinstance(metadata, dict) and isinstance(metadata.get("result_ref"), str):
                        result_ref = metadata["result_ref"]
                except (TypeError, ValueError):
                    pass
        changed = (
            state != row["state"]
            or summary != row["result_summary"]
            or result_ref != row["result_ref"]
            or json.dumps(evidence_refs, ensure_ascii=False) != (row["evidence_refs"] or "[]")
        )
        if not changed:
            continue
        now = _now()
        terminal_at = now if state in DELEGATION_TERMINAL_STATES and not row["completed_at"] else row["completed_at"]
        conn.execute(
            "UPDATE hybrid_card_delegations SET state = ?, result_ref = ?, evidence_refs = ?, "
            "result_summary = ?, updated_at = ?, completed_at = ? WHERE human_card_id = ? AND attempt = ?",
            (
                state,
                result_ref,
                json.dumps(evidence_refs, ensure_ascii=False),
                summary,
                now,
                terminal_at,
                card_id,
                row["attempt"],
            ),
        )
        if state in DELEGATION_TERMINAL_STATES and row["state"] not in DELEGATION_TERMINAL_STATES:
            _activity(
                conn,
                board_id=_require_card(conn, card_id, include_archived=True)["board_id"],
                card_id=card_id,
                kind=f"delegation_{state}",
                actor_type="system",
                actor_id="kanban",
                session_id=None,
                source="delegation-projection",
                payload={
                    "agent_task_id": row["agent_task_id"],
                    "attempt": row["attempt"],
                    "result_ref": result_ref,
                    "evidence_refs": evidence_refs,
                },
            )
        elif state != row["state"]:
            # Progress is a projection of the canonical Agent Task, not a
            # second lifecycle. Persisting this narrow activity fact lets the
            # existing Kanban event stream invalidate the affected Human Card
            # immediately instead of waiting for its fallback poll interval.
            _activity(
                conn,
                board_id=_require_card(conn, card_id, include_archived=True)["board_id"],
                card_id=card_id,
                kind="delegation_progressed",
                actor_type="system",
                actor_id="kanban",
                session_id=None,
                source="delegation-projection",
                payload={
                    "agent_task_id": row["agent_task_id"],
                    "attempt": row["attempt"],
                    "previous_state": row["state"],
                    "state": state,
                },
            )


def _delegation_payloads(conn: sqlite3.Connection, card_id: str) -> list[dict[str, Any]]:
    with kanban_db.write_txn(conn, allow_nested=True):
        _sync_card_delegations_in_txn(conn, card_id)
        rows = conn.execute(
            "SELECT * FROM hybrid_card_delegations WHERE human_card_id = ? ORDER BY attempt",
            (card_id,),
        ).fetchall()
    return [
        {
            "human_card_id": row["human_card_id"],
            "agent_task_id": row["agent_task_id"],
            "attempt": row["attempt"],
            "state": row["state"],
            "result_ref": row["result_ref"],
            "evidence_refs": _decode_list(row["evidence_refs"]),
            "summary": row["result_summary"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]


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
        for card in column["cards"]:
            delegations = _delegation_payloads(conn, card["id"])
            card["delegation"] = delegations[-1] if delegations else None
            card["delegations"] = delegations
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


def delegate_card(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    assignee: Optional[str] = None,
    new_attempt: bool = False,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Explicitly create (or return) the canonical Agent Task for a card.

    The outer transaction serializes duplicate UI clicks and includes the
    link row, so an interrupted delegation cannot leave an unlinked task. A
    normal repeat returns the existing active/terminal attempt; callers must
    pass ``new_attempt=True`` to deliberately create a new Agent Task after a
    terminal attempt. Retrying a blocked attempt is a separate operation and
    keeps its original task identity.
    """
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        _sync_card_delegations_in_txn(conn, card_id)
        latest = conn.execute(
            "SELECT * FROM hybrid_card_delegations WHERE human_card_id = ? ORDER BY attempt DESC LIMIT 1",
            (card_id,),
        ).fetchone()
        if latest is not None and (not new_attempt or latest["state"] not in DELEGATION_TERMINAL_STATES):
            return get_card(conn, card_id)

        attempt = int(latest["attempt"] if latest is not None else 0) + 1
        idempotency_key = f"hybrid-card:{card_id}:attempt:{attempt}"
        context = (
            f"Human card: {card_id}\n"
            f"Human board: {card['board_id']}\n"
            f"Human column: {card['column_id']}\n"
        )
        metadata = _decode(card["metadata"])
        if metadata:
            context += "Human card metadata:\n" + json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n"
        body = f"{card['description']}\n\n{context}".strip()
        task_id = kanban_db.create_task(
            conn,
            title=card["title"],
            body=body,
            assignee=assignee,
            created_by=actor_id or "human-card",
            session_id=session_id,
            board=None,
            initial_status="running",
            idempotency_key=idempotency_key,
        )
        now = _now()
        conn.execute(
            "INSERT INTO hybrid_card_delegations "
            "(human_card_id, agent_task_id, attempt, state, evidence_refs, created_at, updated_at) "
            "VALUES (?, ?, ?, 'delegated', '[]', ?, ?)",
            (card_id, task_id, attempt, now, now),
        )
        _activity(
            conn,
            board_id=card["board_id"],
            card_id=card_id,
            column_id=card["column_id"],
            kind="delegation_created",
            actor_type="human",
            actor_id=actor_id,
            session_id=session_id,
            source=source or "hybrid-delegation",
            payload={"agent_task_id": task_id, "attempt": attempt},
        )
        _sync_card_delegations_in_txn(conn, card_id)
    return get_card(conn, card_id)


def retry_card_delegation(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Retry the same Agent Task identity when it is waiting in ``blocked``.

    A failed/cancelled terminal attempt cannot be silently resurrected. Use
    :func:`delegate_card` with ``new_attempt=True`` for a new attempt and new
    task identity.
    """
    _require_card(conn, card_id)
    _delegations = _delegation_payloads(conn, card_id)
    if not _delegations:
        raise HybridKanbanError("Hybrid card has not been delegated")
    latest = _delegations[-1]
    task = kanban_db.get_task(conn, latest["agent_task_id"])
    if latest["state"] != "waiting" or task is None or task.status != "blocked":
        raise HybridKanbanError("Only a blocked delegation can retry the same Agent Task")
    if not kanban_db.unblock_task(conn, task.id):
        raise HybridKanbanConflict("Agent Task changed; refetch before retrying")
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        _activity(
            conn,
            board_id=card["board_id"],
            card_id=card_id,
            column_id=card["column_id"],
            kind="delegation_retried",
            actor_type="human",
            actor_id=actor_id,
            session_id=session_id,
            source=source or "hybrid-delegation",
            payload={"agent_task_id": task.id, "attempt": latest["attempt"], "retry_kind": "same_task"},
        )
        _sync_card_delegations_in_txn(conn, card_id)
    return get_card(conn, card_id)


def _finish_card_delegation(
    conn: sqlite3.Connection,
    *,
    card_id: str,
    state: str,
    summary: str,
    actor_id: Optional[str],
    session_id: Optional[str],
    source: Optional[str],
) -> dict[str, Any]:
    if state not in {"failed", "cancelled"}:
        raise ValueError("invalid terminal delegation state")
    _require_card(conn, card_id)
    delegations = _delegation_payloads(conn, card_id)
    if not delegations:
        raise HybridKanbanError("Hybrid card has not been delegated")
    latest = delegations[-1]
    if latest["state"] in DELEGATION_TERMINAL_STATES:
        return get_card(conn, card_id)
    task = kanban_db.get_task(conn, latest["agent_task_id"])
    if task is not None and task.status not in {"done", "archived"}:
        # The canonical store has an explicit, safe terminal archive path. The
        # bridge retains the more specific failed/cancelled projection here.
        kanban_db.archive_task(conn, task.id)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        now = _now()
        conn.execute(
            "UPDATE hybrid_card_delegations SET state = ?, result_summary = ?, updated_at = ?, completed_at = ? "
            "WHERE human_card_id = ? AND attempt = ?",
            (state, summary[:2_000], now, now, card_id, latest["attempt"]),
        )
        _activity(
            conn,
            board_id=card["board_id"],
            card_id=card_id,
            column_id=card["column_id"],
            kind=f"delegation_{state}",
            actor_type="human",
            actor_id=actor_id,
            session_id=session_id,
            source=source or "hybrid-delegation",
            payload={"agent_task_id": latest["agent_task_id"], "attempt": latest["attempt"], "summary": summary[:500]},
        )
    return get_card(conn, card_id)


def fail_card_delegation(conn: sqlite3.Connection, *, card_id: str, summary: str = "Agent Task failed.", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    return _finish_card_delegation(conn, card_id=card_id, state="failed", summary=summary, actor_id=actor_id, session_id=session_id, source=source)


def cancel_card_delegation(conn: sqlite3.Connection, *, card_id: str, summary: str = "Delegation cancelled by a human.", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    return _finish_card_delegation(conn, card_id=card_id, state="cancelled", summary=summary, actor_id=actor_id, session_id=session_id, source=source)


def sync_card_delegations(conn: sqlite3.Connection, *, card_id: str) -> dict[str, Any]:
    """Refresh the durable progress/result projection from canonical tasks."""
    _require_card(conn, card_id, include_archived=True)
    with kanban_db.write_txn(conn):
        _sync_card_delegations_in_txn(conn, card_id)
    return get_card(conn, card_id, include_archived=True)


def sync_delegations_for_agent_task(conn: sqlite3.Connection, *, agent_task_id: str) -> bool:
    """Project a canonical task transition to every linked Human Card.

    The caller is normally the existing append-only Kanban event projector.
    Returning a boolean keeps that boundary independent of Hybrid card
    payloads while avoiding a parallel realtime mechanism.
    """
    with kanban_db.write_txn(conn):
        rows = conn.execute(
            "SELECT DISTINCT human_card_id FROM hybrid_card_delegations WHERE agent_task_id = ?",
            (agent_task_id,),
        ).fetchall()
        for row in rows:
            _sync_card_delegations_in_txn(conn, row["human_card_id"])
    return bool(rows)


def _archived_position(conn: sqlite3.Connection, table: str, scope_column: str, scope_id: str) -> int:
    row = conn.execute(
        f"SELECT MIN(position) AS minimum FROM {table} WHERE {scope_column} = ?",
        (scope_id,),
    ).fetchone()
    minimum = row["minimum"] if row and row["minimum"] is not None else 0
    return min(-1, int(minimum) - 1)


def archive_card(conn: sqlite3.Connection, *, card_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id, include_archived=True)
        if card["archived"]:
            return False
        conn.execute(
            "UPDATE hybrid_cards SET archived = 1, position = ?, revision = revision + 1, updated_at = ? WHERE id = ?",
            (_archived_position(conn, "hybrid_cards", "column_id", card["column_id"]), _now(), card_id),
        )
        _activity(conn, board_id=card["board_id"], card_id=card_id, column_id=card["column_id"], kind="card_archived", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"title": card["title"]})
    return True


def restore_card(conn: sqlite3.Connection, *, card_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id, include_archived=True)
        if not card["archived"]:
            return False
        column = _require_column(conn, card["column_id"])
        ordered = _ordered_ids(conn, "hybrid_cards", "column_id", column["id"])
        ordered.append(card_id)
        conn.execute("UPDATE hybrid_cards SET archived = 0 WHERE id = ?", (card_id,))
        _reindex(conn, "hybrid_cards", "column_id", column["id"], ordered)
        conn.execute("UPDATE hybrid_cards SET revision = revision + 1, updated_at = ? WHERE id = ?", (_now(), card_id))
        _activity(conn, board_id=card["board_id"], card_id=card_id, column_id=card["column_id"], kind="card_restored", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"title": card["title"]})
    return True


def archive_column(conn: sqlite3.Connection, *, column_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        column = _require_column(conn, column_id, include_archived=True)
        if column["archived"]:
            return False
        conn.execute("UPDATE hybrid_columns SET archived = 1, position = ?, revision = revision + 1, updated_at = ? WHERE id = ?", (_archived_position(conn, "hybrid_columns", "board_id", column["board_id"]), _now(), column_id))
        _activity(conn, board_id=column["board_id"], column_id=column_id, kind="column_archived", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": column["name"]})
    return True


def restore_column(conn: sqlite3.Connection, *, column_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        column = _require_column(conn, column_id, include_archived=True)
        if not column["archived"]:
            return False
        _require_board(conn, column["board_id"])
        ordered = _ordered_ids(conn, "hybrid_columns", "board_id", column["board_id"])
        ordered.append(column_id)
        conn.execute("UPDATE hybrid_columns SET archived = 0 WHERE id = ?", (column_id,))
        _reindex(conn, "hybrid_columns", "board_id", column["board_id"], ordered)
        conn.execute("UPDATE hybrid_columns SET revision = revision + 1, updated_at = ? WHERE id = ?", (_now(), column_id))
        _activity(conn, board_id=column["board_id"], column_id=column_id, kind="column_restored", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": column["name"]})
    return True


def archive_board(conn: sqlite3.Connection, *, board_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        board = _require_board(conn, board_id, include_archived=True)
        if board["archived"]:
            return False
        conn.execute("UPDATE hybrid_boards SET archived = 1, revision = revision + 1, updated_at = ? WHERE id = ?", (_now(), board_id))
        _activity(conn, board_id=board_id, kind="board_archived", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": board["name"]})
    return True


def restore_board(conn: sqlite3.Connection, *, board_id: str, actor_type: str = "human", actor_id: Optional[str] = None, session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        board = _require_board(conn, board_id, include_archived=True)
        if not board["archived"]:
            return False
        conn.execute("UPDATE hybrid_boards SET archived = 0, revision = revision + 1, updated_at = ? WHERE id = ?", (_now(), board_id))
        _activity(conn, board_id=board_id, kind="board_restored", actor_type=actor_type, actor_id=actor_id, session_id=session_id, source=source, payload={"name": board["name"]})
    return True


def get_card(conn: sqlite3.Connection, card_id: str, *, include_archived: bool = False) -> dict[str, Any]:
    card = _row(_require_card(conn, card_id, include_archived=include_archived))
    delegations = _delegation_payloads(conn, card_id)
    card["delegation"] = delegations[-1] if delegations else None
    card["delegations"] = delegations
    card["checklists"] = list_checklists(conn, card_id=card_id, include_archived_card=include_archived)
    card["activity"] = [_row(r) for r in conn.execute("SELECT * FROM hybrid_activity WHERE board_id = ? AND (card_id = ? OR card_id IS NULL) ORDER BY id DESC", (card["board_id"], card_id))]
    for item in card["activity"]:
        item["payload"] = _decode(item.get("payload"))
    return card


def list_checklists(conn: sqlite3.Connection, *, card_id: str, include_archived_card: bool = False) -> list[dict[str, Any]]:
    _require_card(conn, card_id, include_archived=include_archived_card)
    checklists = [_row(row) for row in conn.execute(
        "SELECT * FROM hybrid_checklists WHERE card_id = ? ORDER BY position, created_at, id",
        (card_id,),
    )]
    for checklist in checklists:
        checklist["items"] = [_row(row) for row in conn.execute(
            "SELECT * FROM hybrid_checklist_items WHERE checklist_id = ? ORDER BY position, created_at, id",
            (checklist["id"],),
        )]
    return checklists


def create_checklist(conn: sqlite3.Connection, *, card_id: str, title: str,
                     actor_type: str = "human", actor_id: Optional[str] = None,
                     session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    if not title or not title.strip():
        raise HybridKanbanError("Checklist title is required")
    with kanban_db.write_txn(conn):
        card = _require_card(conn, card_id)
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM hybrid_checklists WHERE card_id = ?", (card_id,)
        ).fetchone()[0]
        checklist_id, now = _id("hcl"), _now()
        conn.execute(
            "INSERT INTO hybrid_checklists (id, card_id, title, position, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (checklist_id, card_id, title.strip(), position, now, now),
        )
        _activity(conn, board_id=card["board_id"], card_id=card_id, column_id=card["column_id"],
                  kind="checklist_created", actor_type=actor_type, actor_id=actor_id,
                  session_id=session_id, source=source, payload={"checklist_id": checklist_id, "title": title.strip()})
    return next(item for item in list_checklists(conn, card_id=card_id) if item["id"] == checklist_id)


def add_checklist_item(conn: sqlite3.Connection, *, checklist_id: str, body: str,
                       actor_type: str = "human", actor_id: Optional[str] = None,
                       session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    if not body or not body.strip():
        raise HybridKanbanError("Checklist item body is required")
    with kanban_db.write_txn(conn):
        checklist = _require_checklist(conn, checklist_id)
        card = _require_card(conn, checklist["card_id"])
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM hybrid_checklist_items WHERE checklist_id = ?", (checklist_id,)
        ).fetchone()[0]
        item_id, now = _id("hci"), _now()
        conn.execute(
            "INSERT INTO hybrid_checklist_items (id, checklist_id, body, position, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, checklist_id, body.strip(), position, now, now),
        )
        _activity(conn, board_id=card["board_id"], card_id=card["id"], column_id=card["column_id"],
                  kind="checklist_item_created", actor_type=actor_type, actor_id=actor_id,
                  session_id=session_id, source=source, payload={"checklist_id": checklist_id, "item_id": item_id})
    return _row(conn.execute("SELECT * FROM hybrid_checklist_items WHERE id = ?", (item_id,)).fetchone())


def update_checklist_item(conn: sqlite3.Connection, *, item_id: str, body: Optional[str] = None,
                          completed: Optional[bool] = None, expected_revision: Optional[int] = None,
                          actor_type: str = "human", actor_id: Optional[str] = None,
                          session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    if body is not None and not body.strip():
        raise HybridKanbanError("Checklist item body is required")
    with kanban_db.write_txn(conn):
        item, checklist = _require_checklist_item(conn, item_id)
        if expected_revision is not None and item["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid checklist item changed; refetch before editing")
        fields, values = [], []
        if body is not None:
            fields.append("body = ?")
            values.append(body.strip())
        if completed is not None:
            fields.append("completed = ?")
            values.append(int(completed))
        if fields:
            fields.extend(["revision = revision + 1", "updated_at = ?"])
            values.extend([_now(), item_id])
            conn.execute(f"UPDATE hybrid_checklist_items SET {', '.join(fields)} WHERE id = ?", values)
            card = _require_card(conn, checklist["card_id"])
            _activity(conn, board_id=card["board_id"], card_id=card["id"], column_id=card["column_id"],
                      kind="checklist_item_updated", actor_type=actor_type, actor_id=actor_id,
                      session_id=session_id, source=source,
                      payload={"checklist_id": checklist["id"], "item_id": item_id, "completed": completed})
    return _row(conn.execute("SELECT * FROM hybrid_checklist_items WHERE id = ?", (item_id,)).fetchone())


def move_checklist_item(conn: sqlite3.Connection, *, item_id: str, before_id: Optional[str] = None,
                        after_id: Optional[str] = None, expected_revision: Optional[int] = None,
                        actor_type: str = "human", actor_id: Optional[str] = None,
                        session_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        item, checklist = _require_checklist_item(conn, item_id)
        if expected_revision is not None and item["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid checklist item changed; refetch before moving")
        ordered = [row["id"] for row in conn.execute(
            "SELECT id FROM hybrid_checklist_items WHERE checklist_id = ? ORDER BY position, created_at, id", (checklist["id"],)
        )]
        ordered = _place(ordered, item_id, before_id, after_id)
        for temporary, current_id in enumerate(ordered, start=1):
            conn.execute("UPDATE hybrid_checklist_items SET position = ? WHERE id = ?", (-1_000_000_000 - temporary, current_id))
        for position, current_id in enumerate(ordered):
            conn.execute("UPDATE hybrid_checklist_items SET position = ? WHERE id = ?", (position, current_id))
        conn.execute("UPDATE hybrid_checklist_items SET revision = revision + 1, updated_at = ? WHERE id = ?", (_now(), item_id))
        card = _require_card(conn, checklist["card_id"])
        _activity(conn, board_id=card["board_id"], card_id=card["id"], column_id=card["column_id"],
                  kind="checklist_item_moved", actor_type=actor_type, actor_id=actor_id,
                  session_id=session_id, source=source, payload={"checklist_id": checklist["id"], "item_id": item_id})
    return _row(conn.execute("SELECT * FROM hybrid_checklist_items WHERE id = ?", (item_id,)).fetchone())


def delete_checklist_item(conn: sqlite3.Connection, *, item_id: str, expected_revision: Optional[int] = None,
                          actor_type: str = "human", actor_id: Optional[str] = None,
                          session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        item, checklist = _require_checklist_item(conn, item_id)
        if expected_revision is not None and item["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid checklist item changed; refetch before deleting")
        card = _require_card(conn, checklist["card_id"])
        conn.execute("DELETE FROM hybrid_checklist_items WHERE id = ?", (item_id,))
        remaining = [row["id"] for row in conn.execute(
            "SELECT id FROM hybrid_checklist_items WHERE checklist_id = ? ORDER BY position, created_at, id", (checklist["id"],)
        )]
        for position, current_id in enumerate(remaining):
            conn.execute("UPDATE hybrid_checklist_items SET position = ? WHERE id = ?", (position, current_id))
        _activity(conn, board_id=card["board_id"], card_id=card["id"], column_id=card["column_id"],
                  kind="checklist_item_deleted", actor_type=actor_type, actor_id=actor_id,
                  session_id=session_id, source=source, payload={"checklist_id": checklist["id"], "item_id": item_id})
    return True


def delete_checklist(conn: sqlite3.Connection, *, checklist_id: str, expected_revision: Optional[int] = None,
                     actor_type: str = "human", actor_id: Optional[str] = None,
                     session_id: Optional[str] = None, source: Optional[str] = None) -> bool:
    actor_type = _require_actor(actor_type)
    with kanban_db.write_txn(conn):
        checklist = _require_checklist(conn, checklist_id)
        if expected_revision is not None and checklist["revision"] != expected_revision:
            raise HybridKanbanConflict("Hybrid checklist changed; refetch before deleting")
        card = _require_card(conn, checklist["card_id"])
        conn.execute("DELETE FROM hybrid_checklist_items WHERE checklist_id = ?", (checklist_id,))
        conn.execute("DELETE FROM hybrid_checklists WHERE id = ?", (checklist_id,))
        remaining = [row["id"] for row in conn.execute(
            "SELECT id FROM hybrid_checklists WHERE card_id = ? ORDER BY position, created_at, id", (card["id"],)
        )]
        for position, current_id in enumerate(remaining):
            conn.execute("UPDATE hybrid_checklists SET position = ? WHERE id = ?", (position, current_id))
        _activity(conn, board_id=card["board_id"], card_id=card["id"], column_id=card["column_id"],
                  kind="checklist_deleted", actor_type=actor_type, actor_id=actor_id,
                  session_id=session_id, source=source, payload={"checklist_id": checklist_id})
    return True


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
        checklist_ids = [row["id"] for row in conn.execute(
            "SELECT id FROM hybrid_checklists WHERE card_id = ?", (card_id,)
        )]
        for checklist_id in checklist_ids:
            conn.execute("DELETE FROM hybrid_checklist_items WHERE checklist_id = ?", (checklist_id,))
        conn.execute("DELETE FROM hybrid_checklists WHERE card_id = ?", (card_id,))
        conn.execute("DELETE FROM hybrid_card_delegations WHERE human_card_id = ?", (card_id,))
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
        conn.execute(
            "DELETE FROM hybrid_card_delegations WHERE human_card_id IN "
            "(SELECT id FROM hybrid_cards WHERE column_id = ?)",
            (column_id,),
        )
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
        conn.execute(
            "DELETE FROM hybrid_card_delegations WHERE human_card_id IN "
            "(SELECT id FROM hybrid_cards WHERE board_id = ?)",
            (board_id,),
        )
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
