"""Deliver an accepted Kanban Stop through the existing authenticated owner protocol."""
import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def _bindings(conn, task_id, event_id):
    return [{**json.loads(row[1]), 'run_id': row[0]} for row in conn.execute(
        "SELECT run_id,payload FROM task_events WHERE task_id=? AND kind='owner_admitted' AND id<? ORDER BY id",
        (task_id, event_id))]


def _checked_session(home, task_id, board_db, binding):
    from hermes_state_local import POLICY_PREFIX
    path = home / 'state.db'
    if Path(binding['db']).resolve() != path.resolve():
        raise RuntimeError('previous task owner belongs to another profile')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + binding['session_id'],)).fetchone()
    saved = json.loads(row[0]) if row else {}
    policy = saved.get('policy') or {}
    context = json.loads(policy.get('kanban_json') or '{}')
    if (Path(saved.get('profile_id', '')).resolve() != home.resolve() or policy.get('source') != 'kanban'
            or context.get('task_id') != task_id or context.get('run_id') != binding['run_id']
            or Path(context.get('db', '')).resolve() != board_db.resolve()):
        raise RuntimeError('previous task execution identity is unverified')
    return binding['session_id']


async def _deliver(home, task_id, board_db, bindings):
    from hermes_cli.gateway_client import connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    found = await asyncio.to_thread(discover_gateway_endpoint, home)
    if found.state != 'ready' or found.endpoint is None:
        raise RuntimeError('task blocked; process Stop not confirmed: native owner unavailable')
    async with connect_gateway(endpoint=found.endpoint) as client:
        for binding in bindings:
            sid = await asyncio.to_thread(_checked_session, home, task_id, board_db, binding)
            snapshot = await client.rpc('session.resume', session_id=sid)
            # Each native attempt has its own session. Only attempts recorded before
            # this accepted Stop are targets; a successor's session is never included.
            await client.rpc('session.interrupt', session_id=sid,
                             execution_generation=snapshot['execution_generation'])


def request_native_stop(conn, task_id, event_id):
    bindings = _bindings(conn, task_id, event_id)
    if not bindings:
        return
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import gateway_profile_home
    home = Path(gateway_profile_home(get_hermes_home()))
    board_db = Path(next(row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'))
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_deliver(home, task_id, board_db, bindings))
        return
    # Never block the owner loop on a call back into itself. The committed Stop
    # event and predecessor admission check preserve uncertainty if delivery fails.
    task = loop.create_task(_deliver(home, task_id, board_db, bindings))
    def completed(done):
        if not done.cancelled() and (error := done.exception()) is not None:
            logger.warning('Task %s is blocked; native process Stop is unconfirmed (%s)', task_id, type(error).__name__)
    task.add_done_callback(completed)


def block_native_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    kind: Optional[str] = None, expected_run_id: Optional[int] = None,
) -> bool:
    """Commit the native block before requesting scoped Stop for prior owner sessions."""
    stops = []
    changed = _commit_block(conn, task_id, reason=reason, kind=kind, expected_run_id=expected_run_id, stops=stops)
    if changed and stops:
        request_native_stop(conn, task_id, stops[-1])
    return changed


def _commit_block(
    conn: sqlite3.Connection, task_id: str, *, reason, kind, expected_run_id, stops,
) -> bool:
    """``running``/``ready`` -> ``blocked`` (or ``todo`` / ``triage``, see
    :func:`kb._route_block`). ``kind='dependency'`` with no incomplete parent is
    re-kinded to ``needs_input`` (sticky) so ``recompute_ready`` cannot
    promote it into a context-free respawn. ``transient`` still counts
    toward the loop breaker so a forever-flaky task escalates. True on any
    transition.

    An already-``blocked`` card that the failure breaker parked UNTYPED
    (``block_kind IS NULL``, no live run) is classified in place when *kind*
    is supplied: ``block_kind``/``block_recurrences`` are set and a ``blocked``
    audit event is appended, while status, failure evidence and the terminal
    runs stay exactly as the breaker left them. A typed block, a card with a
    live run, or a kind-less call on a blocked card are still refused.
    """
    from hermes_cli import kanban_db as kb
    if kind is not None and kind not in kb.VALID_BLOCK_KINDS:
        raise ValueError(f"block kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} or None")
    with kb.write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        # The breaker (``_record_task_failure``) parks cards ``blocked`` with no
        # ``block_kind`` and no ``blocked`` event -- the policy is the
        # supervisor's, not the kernel's -- but the transition guard below only
        # matches running/ready, so that policy could never be attached later
        # (#117363). Classify in place; never re-type or flap status. A caller
        # asserting run ownership (``expected_run_id``) cannot own a parked
        # card -- its run is over -- so it is refused like any stale worker.
        if cur_row["status"] == "blocked":
            if kind is None or expected_run_id is not None or kb._row_get(cur_row, "block_kind") is not None:
                return False
            classified = conn.execute(
                "UPDATE tasks SET block_kind = ?, block_recurrences = 1 "
                "WHERE id = ? AND status = 'blocked' AND block_kind IS NULL "
                "AND current_run_id IS NULL",
                (kind, task_id),
            ).rowcount
            if classified != 1:
                return False
            kb._append_event(conn, task_id, "blocked", {
                "kind": kind, "reason": reason, "classified_in_place": True,
            })
            stops.append(conn.execute('SELECT last_insert_rowid()').fetchone()[0])
            return True
        source_status = kb._retry_status_for_run(conn, task_id) if cur_row["status"] == "running" else "ready"
        requested_kind = kind
        rekind_reason = None
        # ``dependency`` only waits on incomplete parents. A worker filing that
        # kind with none open would park in ``todo`` and ``recompute_ready``
        # would promote+respawn it context-free on the next tick. Re-kind to
        # ``needs_input`` so it is sticky until a human unblocks.
        if kind == "dependency" and kb._parents_satisfied(conn, task_id):
            kind = "needs_input"
            rekind_reason = "no_open_parent"
        new_status, event_kind, set_sql, params, payload = kb._route_block(
            kind, reason, source_status, prev_kind=kb._row_get(cur_row, "block_kind"),
            prev_recurrences=int(kb._row_get(cur_row, "block_recurrences") or 0),
        )
        if rekind_reason:
            payload["requested_kind"] = requested_kind
            payload["rekind_reason"] = rekind_reason
        sql = f"""
                UPDATE tasks
                   SET status        = '{new_status}',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       {set_sql}
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """
        params = (*params, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = kb._end_or_synthesize_run(
            conn, task_id, outcome="blocked", status="blocked", summary=reason, synthesize=bool(reason),
        )
        kb._append_event(conn, task_id, event_kind, payload, run_id=run_id)
        if event_kind in {'blocked', 'block_loop_detected'}:
            stops.append(conn.execute('SELECT last_insert_rowid()').fetchone()[0])
        blocked_task = kb.get_task(conn, task_id)
        if kind == "dependency":
            # Historical ordering: the dependency lane fires inside the txn.
            kb._fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
            return True
    kb._fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
    return True
