"""Cancel kanban work (``kanban.cancel`` / ``kanban.cancel_all``, B3).

The gateway verbs over :func:`hermes_cli.kanban_db.cancel_task` /
:func:`hermes_cli.kanban_db.cancel_tasks_for_session`: one card (plus its
non-terminal descendants) or every non-terminal card one session created.
A live worker is signalled (SIGTERM, short grace, SIGKILL, fingerprint-checked)
before the card is marked; a worker that cannot be signalled keeps running
until it exits, and its result is still discarded — completion refuses a
``cancelled`` card. Cancellation never touches the circuit breaker and never
rewrites anything already persisted.

Authorization is the profile boundary: ``session:control`` plus the actor's
profile matching this authority's. The origin session is a filter key, not an
identity to prove — background work outlives its session by design, so a
stop-all must also work for a finished session; within one profile every
session is the same user's. A foreign profile answers ``profile_mismatch``
with no data. The board is shared across profiles by design
(``kanban_home``), so the board resolves the way the gateway's own kanban
watchers resolve it; ``board`` only selects which one.

The kill can block for seconds (``_poll_worker_exit``), so all database work
runs off-loop; the handler itself only validates and awaits.
"""

from __future__ import annotations

import asyncio

from persistence.core.errors import AuthorityRefusal as RuntimeStoreError


def _require_control(connection):
    authority, actor = connection.authority, connection.actor
    if "session:control" not in actor.capabilities:
        raise RuntimeStoreError("permission_denied")
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError("profile_mismatch")
    return authority, actor


def _text_param(params, name, *, required):
    value = params.get(name)
    if value is None:
        if required:
            raise RuntimeStoreError("invalid_params", field=name)
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeStoreError("invalid_params", field=name)
    return value


def _cancel_one(board, task_id, reason, cancelled_by):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing(board=board) as conn:
        outcome = kb.cancel_task(conn, task_id, reason=reason, cancelled_by=cancelled_by)
    return {
        "task_id": outcome.task_id,
        "cancelled": outcome.cancelled,
        "status": outcome.status,
        "cancelled_ids": outcome.cancelled_ids,
        "workers_terminated": outcome.workers_terminated,
        "workers_survived": outcome.workers_survived,
    }


def _cancel_all(board, origin_session_id, reason, cancelled_by):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing(board=board) as conn:
        cancelled = kb.cancel_tasks_for_session(
            conn, origin_session_id, reason=reason, cancelled_by=cancelled_by,
        )
    return {"origin_session_id": origin_session_id, "cancelled_ids": cancelled}


async def cancel_task(connection, ref, params):
    """``kanban.cancel`` handler: one card + descendants, ``session:control``."""
    _require_control(connection)
    task_id = _text_param(params, "task_id", required=True)
    board = _text_param(params, "board", required=False)
    reason = _text_param(params, "reason", required=False)
    cancelled_by = _text_param(params, "cancelled_by", required=False)
    return await asyncio.to_thread(_cancel_one, board, task_id, reason, cancelled_by)


async def cancel_all(connection, ref, params):
    """``kanban.cancel_all`` handler: stop-all for one origin session."""
    _require_control(connection)
    origin_session_id = _text_param(params, "origin_session_id", required=True)
    board = _text_param(params, "board", required=False)
    reason = _text_param(params, "reason", required=False)
    cancelled_by = _text_param(params, "cancelled_by", required=False)
    return await asyncio.to_thread(
        _cancel_all, board, origin_session_id, reason, cancelled_by,
    )
