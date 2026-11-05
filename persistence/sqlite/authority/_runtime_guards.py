"""Transcript mutation guards evaluated on the writer connection of the transaction that publishes or rewrites a
target, the delete target walk and its footprint, the row probes and continuation clauses they use, and the
gateway's history-mount check."""

from persistence.core.errors import AuthorityRefusal

from .._schema_ddl import _sql_json_extract


_ENDED_ROW_SQL = "SELECT ended_at, end_reason FROM sessions WHERE id = ?"


def _ended_by_compression(row) -> bool:
    return row is not None and row["ended_at"] is not None and row["end_reason"] == "compression"


def _non_continuation_child_sql(child: str = "", parent: str = "?") -> str:
    """``  AND ...`` clauses rejecting children that are NOT compression continuations of *parent*
    (branch/delegate/reset forks, tool sessions).  Markers are bound to the parent id: continuations
    inherit ``model_config`` verbatim, so a marker naming another row is inherited, not a fork.
    ``child`` is the column prefix (``""``, ``"c."``)."""
    return "".join(
        f"  AND COALESCE({_sql_json_extract(f'{child}model_config', f'$.{marker}')}, '') != {parent}\n"
        for marker in ("_branched_from", "_delegate_from", "_reset_from")
    ) + f"  AND COALESCE({child}source, '') != 'tool'\n"


def require_not_executing(conn, session_ids):
    """Refuse while a turn is running (or its outcome is unknown) on any of ``session_ids``.
    Queued admissions are allowed: a follower waits on the logical owner and simply runs
    against whatever physical target the mutation publishes. Workers have no queued state
    (registered/running/unknown are all live), so any non-terminal worker is executing."""
    for sid in session_ids:
        admissions = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('started','unknown')", (sid,)).fetchall()
        workers = conn.execute("SELECT status FROM worker_executions WHERE session_id=? AND status IN ('registered','running','unknown')", (sid,)).fetchall()
        states = {row[0] for row in [*admissions, *workers]}
        if 'unknown' in states:
            raise AuthorityRefusal('unknown_execution')
        if states:
            raise AuthorityRefusal('session_busy')


def _require_transcript_unleased(holder, conn, sid):
    # A logical owner closed by compression is an ancestor, not a transcript
    # target; its live successor (also in session_ids) carries the lease/lock.
    if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (sid,)).fetchone()):
        return
    from ..conversation._store.messages import _check_transcript_write_guards
    _check_transcript_write_guards(holder, conn, sid, None,
        reject_active_turn_lease=True, reject_active_compression_lock=True)


def require_target_advanceable(holder, conn, session_ids):
    """Fence for writes that publish a new physical target (reset, compress, model):
    refuse executing/unknown work and live transcript leases, never a queued follower,
    which waits on the logical owner and runs against whatever target is published."""
    require_not_executing(conn, session_ids)
    for sid in session_ids:
        _require_transcript_unleased(holder, conn, sid)


def require_idle(holder, conn, session_ids):
    for sid in session_ids:
        admissions = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')", (sid,)).fetchall()
        workers = conn.execute("SELECT status FROM worker_executions WHERE session_id=? AND status IN ('registered','running','unknown')", (sid,)).fetchall()
        states = {row[0] for row in [*admissions, *workers]}
        if 'unknown' in states:
            raise AuthorityRefusal('unknown_execution')
        if states:
            raise AuthorityRefusal('session_busy')
        _require_transcript_unleased(holder, conn, sid)


def _continuation_edges(conn, sid):
    """One both-direction compression step from *sid*: its continuation children and its
    compression parent. Fork markers are matched against the edge's parent id (the
    ``_non_continuation_child_sql`` rule), never by presence: a continuation copies its parent's
    ``model_config``, so an inherited ``_branched_from``/``_delegate_from`` naming an older row
    does not make it a fork, and a delegate's continuation published by a canonical ``compress``
    carries no marker at all."""
    edge = _non_continuation_child_sql('child.', 'parent.id')
    rows = conn.execute(
        'SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id=parent.id '
        "WHERE parent.id=? AND parent.end_reason='compression'\n" + edge
        + ' UNION SELECT parent.id FROM sessions child JOIN sessions parent ON parent.id=child.parent_session_id '
        "WHERE child.id=? AND parent.end_reason='compression'\n" + edge, (sid, sid)).fetchall()
    return [row[0] for row in rows]


def delete_targets(conn, session_id, rule=None):
    from ..conversation._store.sessions import _collect_delegate_child_ids
    import json
    from ..conversation._store.compression import _CHAIN_CAP
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import validate_local_lineage
    from ..conversation._store.local import local_lineage_owner
    targets = {session_id}
    # A reset/compression segment of a local conversation is that conversation: the listing
    # shows one row for it, so its delete takes the creation id (policy, FIFO) and every segment.
    owner = local_lineage_owner(conn, session_id)
    targets.add(owner)
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + owner,)).fetchone()
    if saved is not None:
        receipt = json.loads(saved[0])
        validate_local_lineage(conn, receipt)
        targets.update(receipt.get('lineage', [session_id]))
    # Canonical admissions bind to the compression root for every producer, not only
    # local receipts: every physical continuation of a target goes with it, or the next
    # message on the route re-admits the "deleted" conversation through the surviving child.
    # The walk also runs BACKWARD to the root (#57543): a sidebar row carries the chain tip's
    # id, and a surviving root re-projects as the "deleted" conversation on the next reload.
    # Every walked row must belong to its anchor's principal domain: only the requested row was
    # authorized, so a parent link into another principal's chain (an imported or forged edge)
    # stops the walk instead of deleting their conversation. A delegate subtree is the parent's
    # by construction and anchors its own chain (a continuation copies the delegate's route).
    # Delegates and continuations close over each other to a fixed point: a compressed
    # delegate's continuation (and its queued work) goes with it, or require_idle never sees it
    # and the delete orphans it.
    from ._binding_history import same_history_owner
    anchor = dict.fromkeys(targets, session_id)
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        found = {}
        for sid in frontier:
            for nxt in _continuation_edges(conn, sid):
                if nxt not in targets and nxt not in found and same_history_owner(conn, anchor[sid], nxt, rule):
                    found[nxt] = anchor[sid]
        for sid in _collect_delegate_child_ids(conn, frontier):
            if sid not in targets:
                found.setdefault(sid, sid)
        if not found:
            break
        anchor.update(found)
        targets.update(found)
        frontier = list(found)
    return [session_id, *sorted(targets - {session_id})]


def delete_footprint(holder, session_id, owner=None):
    """``(targets, children)`` in one read: the delete walk of *session_id* and every row whose parent is a target."""
    with holder._read_ctx() as conn:
        targets = delete_targets(conn, session_id, owner)
        children = conn.execute(f"SELECT id FROM sessions WHERE parent_session_id IN ({','.join('?' * len(targets))})",
                                targets).fetchall()
    return targets, [row[0] for row in children]
