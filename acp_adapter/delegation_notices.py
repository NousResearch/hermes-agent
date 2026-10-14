"""Settle the children of a background delegation in clients that draw them as rows.

``delegate_task`` detaches a background batch and enqueues ONE completion event per delegation on
the shared ``process_registry.completion_queue``; the CLI and the gateway drain that queue between
turns, so the result arrives as the next turn. An ACP client is not that drain: a chat UI that
renders the children as rows (Orca's native chat) kept them "working" long after the children
finished, with nothing to settle them. This module claims the events that belong to a live ACP
session and renders each one as a ``session/update`` — a line the person can read, plus the roster
states under ``_meta`` for the client's subagent rows.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

CONSUMER = "acp"
# The record fields that carry the dispatching session, most specific first.
_SESSION_FIELDS = ("session_key", "parent_session_id", "origin_session_id", "origin_ui_session_id")
# Hermes' own outcome vocabulary, as a client's five subagent states.
_STATES = {
    "completed": "completed",
    "failed": "failed",
    "error": "failed",
    "timeout": "failed",
    "cancelled": "stopped",
    "canceled": "stopped",
    "stopped": "stopped",
}
# The key the client reads the roster from, beside the human-readable line.
META_KEY = "hermes"


def owns_event(event: dict[str, Any], session_id: str) -> bool:
    """Whether ``event`` was dispatched by ``session_id``."""
    return bool(session_id) and any(event.get(field) == session_id for field in _SESSION_FIELDS)


def child_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per child: a batch reports ``results``, a single delegation is its own row.

    Ids match the rows the dispatch produced, so a client settles the roster it already drew
    (``<delegation_id>:<task index>`` for a batch, the delegation id for a single child).
    """
    delegation_id = str(event.get("delegation_id") or "")
    results = event.get("results")
    if not isinstance(results, list) or not results:
        results = [{"task_index": None, "status": event.get("status"), "summary": event.get("summary")}]
    rows: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        index = entry.get("task_index")
        rows.append({
            "id": f"{delegation_id}:{index}" if index is not None else delegation_id,
            "state": _STATES.get(str(entry.get("status") or "").lower(), "unverifiable"),
            "summary": str(entry.get("summary") or ""),
        })
    return rows


def summary_line(event: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """The line the person reads, naming what each child answered."""
    if len(rows) == 1:
        row = rows[0]
        text = row["summary"] or str(event.get("goal") or "")
        return f"Subagent {row['state']}: {text}".strip()
    finished = sum(1 for row in rows if row["state"] == "completed")
    answers = "; ".join(row["summary"] or row["id"] for row in rows)
    return f"{finished} of {len(rows)} subagents finished: {answers}"


def completion_update(event: dict[str, Any]) -> dict[str, Any]:
    """The ``session/update`` for one finished delegation."""
    rows = child_rows(event)
    return {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": summary_line(event, rows)},
        "_meta": {META_KEY: {"subagents": rows}},
    }


def claim_completions(
    session_ids: Iterable[str], consumer: str = CONSUMER
) -> list[tuple[str, dict[str, Any], Optional[str]]]:
    """Take the completions belonging to ``session_ids``, leaving every other event queued.

    Returns ``(session_id, event, claim_id)``. The claim is the ledger's promise that this drain
    owns the delivery: acknowledge it with :func:`mark_delivered` once the client has the update,
    or hand it back with :func:`release` when the emission failed.
    """
    from tools import async_delegation
    from tools.process_registry import process_registry

    queue = process_registry.completion_queue
    ids = [session_id for session_id in session_ids if session_id]
    claimed: list[tuple[str, dict[str, Any], Optional[str]]] = []
    requeue: list[Any] = []
    while True:
        try:
            event = queue.get_nowait()
        except Exception:
            break
        session_id = next(
            (i for i in ids if isinstance(event, dict) and owns_event(event, i)), None
        )
        if session_id is None or not isinstance(event, dict) or event.get("type") != "async_delegation":
            requeue.append(event)
            continue
        try:
            claim_id = async_delegation.claim_event_delivery(event, consumer)
        except Exception:
            logger.debug("Delegation completion claim failed", exc_info=True)
            requeue.append(event)
            continue
        if claim_id is None:
            requeue.append(event)  # another drain owns this one
            continue
        claimed.append((session_id, event, claim_id or None))
    for event in requeue:
        try:
            queue.put(event)
        except Exception:
            logger.debug("Could not requeue a completion event", exc_info=True)
    return claimed


def mark_delivered(event: dict[str, Any], claim_id: Optional[str]) -> None:
    """Acknowledge a claimed completion so a restart does not deliver it again."""
    if claim_id is None:
        return
    try:
        from tools import async_delegation

        async_delegation.mark_completion_delivered(str(event.get("delegation_id") or ""))
    except Exception:
        logger.debug("Could not mark a completion delivered", exc_info=True)


def release(event: dict[str, Any], claim_id: Optional[str]) -> None:
    """Hand a claimed completion back after a failed emission, so another drain can take it."""
    if claim_id is None:
        return
    try:
        from tools import async_delegation

        async_delegation.release_completion_delivery(str(event.get("delegation_id") or ""), claim_id)
    except Exception:
        logger.debug("Could not release a completion claim", exc_info=True)
