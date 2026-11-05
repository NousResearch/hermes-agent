"""Local creation receipts on the store: the reservation, the receipt reads, the hand-off of receipts a messaging lane
took over, the physical-target resolution, the untracked-compression adoption, the local ``/new`` reset and the idle
end.

These functions raise ``AuthorityRefusal`` with main's reason codes. The conversation family's lineage bodies
(``validate_local_lineage``, ``advance_local_target``, ``local_lineage_owner``), its row helpers and the ``state_meta``
prefixes are imported from it where each function uses them; the hand-off bodies (``moved_receipts``,
``hand_off_conversations``) are the gateway family's.
"""

import json
import time

from persistence.authority.bindings import SessionBinding
from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES
from persistence.core.errors import AuthorityRefusal
from persistence.core.identity import SessionId
from session_rules.legacy_local import not_client_labels
from session_rules.local_create import same_create_retry

from ..gateway._local_lineage import hand_off_conversations, moved_receipts
from ._runtime import _epoch, _json


def commit_local_session(holder, *, epoch, receipt, client_request, created_source, owner=None):
    """Reserve the row, route and immutable policy together, before publication. Returns ``(receipt, created)``:
    the stored receipt of an exact retry with False, else the receipt this write inserted with True."""
    request = client_request  # the staged request's own client_env, never stored
    if request is None:
        if receipt['policy'].get('client_env_json') is not None:
            # Lost (a copy): never judge a retry by this attempt's own filtered map instead.
            raise AuthorityRefusal('invalid_params')
        request = {}
    encoded = _json(receipt)
    receipt = json.loads(encoded)
    sid, route = receipt['session_id'], receipt['route']
    from ..conversation._store.local import POLICY_PREFIX
    key = POLICY_PREFIX + sid
    def write(conn):
        _epoch(conn, epoch)
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None:
            try:
                saved = json.loads(old[0])
                same = all(saved[k] == receipt[k] for k in (
                    'profile_id', 'principal_id', 'request_id', 'session_id', 'route'))
                same = same and same_create_retry(saved['policy'], receipt['policy']['request_json'], request)
            except (ValueError, TypeError, KeyError) as exc:
                raise AuthorityRefusal('storage_unavailable') from exc
            if not same:
                raise AuthorityRefusal('invalid_params')
            return saved, False
        from ..conversation._store.retirement import RETIRED_PREFIX
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone():
            # Deleted: its receipt was retired with the transcript; a create/cron retry must not
            # resurrect the row (and re-fire its prompt) under the same deterministic id.
            raise AuthorityRefusal('not_found')
        if ('legacy_session_id' not in receipt and
                conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone()):
            raise AuthorityRefusal('storage_unavailable')
        if conn.execute("SELECT 1 FROM gateway_routing WHERE scope='' AND session_key=?", (route,)).fetchone():
            raise AuthorityRefusal('admission_conflict')
        if 'legacy_session_id' in receipt:
            from ._local_migration import bind_legacy_target
            bind_legacy_target(holder, conn, receipt, owner)
        policy, entry = receipt['policy'], receipt['entry']
        from datetime import datetime
        started = datetime.fromisoformat(entry['created_at']).timestamp()
        # Auxiliary usage inserts created_source='unknown' when it creates the row
        # (record_auxiliary_usage carries no surface). The creator upsert repairs
        # 'unknown', but this DO NOTHING insert does not, so the surface is stamped here.
        # The agent's own accounting passes its surface (turn_usage._agent_session_source)
        # and is not that writer. gui's in-process agent is desktop; a managed worker
        # creates with the policy label gui.
        created = created_source or None
        from ..conversation._store.sessions import _own_profile_name
        conn.execute('''INSERT INTO sessions(id,source,created_source,user_id,session_key,chat_id,chat_type,
            model,cwd,profile_name,origin_json,started_at) VALUES(?,?,?,?,?,?,'dm',?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING''',
            (sid, policy['source'], created, receipt['principal_id'], route,
             entry['origin']['chat_id'], policy['model'], _row_cwd(policy), _own_profile_name(holder),
             _json(entry['origin']), started))
        conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                     (route, _json(entry), started))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, encoded))
        return receipt, True
    return holder._execute_write(write)


def _row_cwd(policy):
    """The sessions-row ``cwd``: the create request's ``row_cwd`` (``""`` = none) when it sent one,
    else the execution cwd. Row metadata only (workspace grouping, ``-c``, cwd restore); the
    frozen policy's ``cwd`` / ``TERMINAL_CWD`` stay where the session's tools run."""
    try:
        request = json.loads(policy.get('request_json') or '{}')
    except (TypeError, ValueError):
        request = {}
    value = request.get('row_cwd') if isinstance(request, dict) else None
    return policy['cwd'] if not isinstance(value, str) else (value or None)


def release_handed_off_receipts(holder, *, epoch, lanes):
    """Boot heal for #143286: an older build's ``/handoff`` (or a platform ``/resume``) moved a local
    session onto a messaging lane but kept its local receipt, so every open, pin or rename refused it
    (``stored identity mismatch``) and its title never resolved. Retire each such receipt, so the row
    restores from its route like any other platform session, and move the rest of its conversation
    there with it (``hand_off_conversations``). Only receipts whose CURRENT target a messaging lane
    now holds (a source in *lanes*): any other refused receipt stays as it is, logged when it is touched."""
    def write(conn):
        _epoch(conn, epoch)
        moved = {owner: (target, source) for owner, (target, source) in moved_receipts(conn).items()
                 if source in lanes}
        return hand_off_conversations(conn, moved)
    return holder._execute_write(write)


def local_receipt(holder, session_id):
    from ..conversation._store.local import POLICY_PREFIX
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('storage_unavailable')
    try:
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError('invalid receipt')
        return value
    except (TypeError, ValueError) as exc:
        raise AuthorityRefusal('storage_unavailable') from exc


def binding_record(receipt):
    """The stored receipt as a detached ``SessionBinding``; a receipt missing an identity field is
    ``storage_unavailable``."""
    try:
        lineage = receipt.get('lineage', [receipt['session_id']])
        return SessionBinding(
            profile_id=receipt['profile_id'], principal_id=receipt['principal_id'], request_id=receipt['request_id'],
            session_id=SessionId(receipt['session_id']), route=receipt['route'], entry=receipt['entry'],
            policy=receipt['policy'], lineage=tuple(SessionId(sid) for sid in lineage),
            legacy_session_id=receipt.get('legacy_session_id'))
    except (AttributeError, TypeError, KeyError) as exc:
        raise AuthorityRefusal('storage_unavailable') from exc


def receipt_of(binding):
    """The receipt dict a ``SessionBinding`` was read from (``lineage`` present unless it is the default)."""
    receipt = {'profile_id': binding.profile_id, 'principal_id': binding.principal_id,
               'request_id': binding.request_id, 'session_id': binding.session_id, 'route': binding.route,
               'entry': dict(binding.entry), 'policy': dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt['legacy_session_id'] = binding.legacy_session_id
    if binding.lineage != (binding.session_id,):
        receipt['lineage'] = list(binding.lineage)
    return json.loads(json.dumps(receipt))


def legacy_adoption_scan(holder, owner):
    """``(ids, marked)`` in one read: the candidate session ids ordered by start time then id (a legacy local
    source, or a routeless row whose label :func:`not_client_labels` does not refuse under *owner*), and the ids
    carrying a local creation receipt or a legacy binding (key prefix stripped)."""
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_migration import LEGACY_PREFIX
    local = tuple(sorted(LEGACY_LOCAL_SOURCES))
    refused = tuple(sorted(not_client_labels(owner)))
    with holder._read_ctx() as conn:
        # The routeless rows of every other label (a phone app's own) are candidates too; the refused labels
        # are filtered here so the bulk of cron/subagent rows never reaches the caller's per-row check.
        ids = [row[0] for row in conn.execute(
            f"SELECT id FROM sessions WHERE source IN ({','.join('?' * len(local))})"
            f" OR (source NOT IN ({','.join('?' * len(refused))}) AND ifnull(session_key,'')=''"
            " AND ifnull(chat_id,'')='' AND ifnull(origin_json,'')='') ORDER BY started_at, id",
            local + refused)]
        bound = {row[0] for row in conn.execute(
            'SELECT key FROM state_meta WHERE key LIKE ? OR key LIKE ?', (POLICY_PREFIX + '%', LEGACY_PREFIX + '%'))}
    return ids, {key[len(prefix):] for key in bound for prefix in (POLICY_PREFIX, LEGACY_PREFIX) if key.startswith(prefix)}


def handoff_head(holder, session_id):
    """``(local owner or None, row)`` a ``/handoff`` of *session_id* moves: the conversation the user
    is looking at. A local conversation continues from its receipt's current target (its latest
    ``/new`` segment, then that segment's compression tip), never from an earlier segment; any other
    row from its compression tip."""
    from ..conversation._store.compression import get_compression_tip
    from ..conversation._store.local import POLICY_PREFIX, _policy_target, local_lineage_owner
    with holder._read_ctx() as conn:
        owner = local_lineage_owner(conn, session_id)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + owner,)).fetchone()
    target = _policy_target(saved[0]) if saved else None
    start = target or session_id
    return (owner if target else None), get_compression_tip(holder, start) or start


# The receipts boot restores: an open row (or one compressed away, for its catch-up), unsettled input
# or a stored close. An ended session keeps its receipt and stays resumable; ``authorize`` restores it
# on first touch, so the live registry no longer grows with every session ever opened (#143547 LC-36).
_BOOT_RECEIPTS_SQL = (
    "SELECT m.key FROM state_meta m LEFT JOIN sessions s "
    "ON s.id = CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.entry.session_id') END "
    "WHERE m.key LIKE ? AND (s.ended_at IS NULL OR s.end_reason = 'compression' "
    "OR EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id = substr(m.key, ?) "
    "AND a.status IN ('queued','started','unknown')) "
    "OR EXISTS (SELECT 1 FROM state_meta c WHERE c.key = ? || substr(m.key, ?)))")


def local_receipt_ids(holder):
    """The ids of the receipts boot restores, in one read."""
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX
    start = len(POLICY_PREFIX) + 1
    with holder._read_ctx() as conn:
        return [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            _BOOT_RECEIPTS_SQL, (POLICY_PREFIX + '%', start, CLOSE_PREFIX, start))]


def end_idle_local_session(holder, *, epoch, session_id, target_id, reason):
    """Stamp ``ended_at`` on *target_id* only while the logical *session_id*'s FIFO is idle.

    The idle check and the stamp share one owner write txn, so an admission committed first keeps
    the row open; one committed after is reopened by the drain (``reopen_local_session``)."""
    from ..conversation._store.sessions import _end_and_bump
    def write(conn):
        _epoch(conn, epoch)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')",
                        (session_id,)).fetchone():
            return 0
        if conn.execute("SELECT 1 FROM worker_executions WHERE session_id IN (?,?) AND status IN ('registered','running','unknown')",
                        (session_id, target_id)).fetchone():
            return 0
        return _end_and_bump(holder, conn, 'UPDATE sessions SET ended_at=?, end_reason=? WHERE id=? AND ended_at IS NULL',
                             (time.time(), reason, target_id), target_id, reason)
    return holder._execute_write(write)


def local_physical_target(conn, session_id):
    """Resolve a local owner's current transcript on the caller's transaction."""
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import validate_local_lineage
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return session_id
    receipt = json.loads(saved[0])
    if receipt['session_id'] != session_id:
        raise AuthorityRefusal('storage_unavailable')
    return validate_local_lineage(conn, receipt)


def _untracked_steps(conn, tip, lineage):
    """Every compression step past *tip*, in order; None when the chain leads back into itself.

    One ``_walk_compression_chain`` stops after ``_CHAIN_CAP`` steps, which is not the end of the
    chain: walk on from the last step until a walk finds nothing more. Each batch only adds ids never
    seen, so the walk ends; a step back into the lineage or an earlier batch is a cycle.
    """
    from ..conversation._store.compression import _CHAIN_STEP_SQL, _walk_compression_chain
    steps, seen, current = [], {*lineage, tip}, tip
    while True:
        batch = _walk_compression_chain(current, lambda sid: conn.execute(_CHAIN_STEP_SQL, (sid,)).fetchone())[1:]
        if not batch:
            return steps
        if seen.intersection(batch):
            return None
        seen.update(batch)
        steps.extend(batch)
        current = batch[-1]


def adoptable_compression(conn, receipt):
    """``(steps, routed)``: the untracked compression steps *receipt* would adopt and its route's
    entry, else None. Read-only; ``adopt_untracked_compression`` repeats it inside its write."""
    from ..conversation._store.local_lineage import _routed_entry
    tip = receipt['entry']['session_id']
    children = _untracked_steps(conn, tip, receipt.get('lineage', [receipt['session_id']]))
    if not children:
        return None
    identity = (receipt['route'], receipt['session_id'], receipt['principal_id'])
    for child_id in children:
        child = conn.execute('SELECT session_key,chat_id,user_id FROM sessions WHERE id=?', (child_id,)).fetchone()
        if tuple(child) != identity:
            return None
    routed = _routed_entry(conn, receipt['route'], tip)
    return (children, routed) if routed is not None else None


def compression_adoptable(holder, receipt):
    """Whether ``adopt_untracked_compression`` would adopt steps for *receipt*, in one read."""
    with holder._read_ctx() as conn:
        return adoptable_compression(conn, receipt) is not None


def adopt_untracked_compression(holder, *, epoch, session_id, expected):
    """Record, once, compression steps a receipt-unaware build published past the receipt's tip.

    0.22.2 (the classic line) resumes a 0.22.1 local session by id. Its compaction ends the tip with
    ``compression`` and copies the parent's route columns onto the child, but never advances the
    receipt, so ``validate_local_lineage`` refused the session forever (#140060). On this owner a
    compaction advances the receipt in its own transaction, so such a step comes only from an older
    writer. A child carrying the receipt's own route, owner and principal is that session's
    continuation: append it to the lineage and move the receipt's and the routing entry's target to
    it, nothing else. The stored receipt must still be *expected* (a ``SessionBinding``), the copy the
    caller checked against its owner. Anything else is left for validation to refuse, unchanged.
    """
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import _commit_local_target, validate_local_lineage
    def write(conn):
        _epoch(conn, epoch)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
        receipt = json.loads(saved[0])
        try:
            if binding_record(receipt) != expected:
                return
        except AuthorityRefusal:  # a receipt missing an identity field is not the expected one
            return
        adoptable = adoptable_compression(conn, receipt)
        if adoptable is None:
            return
        children, routed = adoptable
        receipt['entry'] = dict(receipt['entry'], session_id=children[-1])
        receipt['lineage'] = [*receipt.get('lineage', [session_id]), *children]
        validate_local_lineage(conn, receipt)
        _commit_local_target(conn, receipt, dict(routed, session_id=children[-1]))
    holder._execute_write(write)


def reset_local_target(holder, *, epoch, parent_session_id, entry):
    from ..conversation._store.compression import _publish_child_session_row
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import advance_local_target, validate_local_lineage
    from ..conversation._store.messages import _bump_conversation_generation
    def write(conn):
        _epoch(conn, epoch)
        parent = conn.execute('SELECT * FROM sessions WHERE id=?', (parent_session_id,)).fetchone()
        if parent is None or parent['end_reason'] == 'compression':
            raise AuthorityRefusal('admission_conflict')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (POLICY_PREFIX + parent['chat_id'],)).fetchone()
        if saved is None:
            raise AuthorityRefusal('storage_unavailable')
        receipt = json.loads(saved[0])
        logical_id = receipt['session_id']
        if validate_local_lineage(conn, receipt) != parent_session_id:
            raise AuthorityRefusal('admission_conflict')
        from ._runtime_guards import require_not_executing
        require_not_executing(conn, list({logical_id, parent_session_id}))
        policy = receipt['policy']
        _publish_child_session_row(holder, conn, parent, parent_session_id=parent_session_id,
            child_session_id=entry['session_id'], source=policy['source'], model=policy['model'],
            model_config={'_reset_from': parent_session_id}, system_prompt=None,
            cwd=policy['cwd'], profile_name=parent['profile_name'])
        conn.execute("UPDATE sessions SET ended_at=?,end_reason='session_reset' WHERE id=?",
                     (time.time(), parent_session_id))
        _bump_conversation_generation(holder, conn, parent_session_id, 'session_reset')
        advance_local_target(conn, parent_session_id, entry['session_id'], entry=entry)
        conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (logical_id,))
    holder._execute_write(write)


def unused_session(conn, session_id):
    """*session_id* holds nothing: no transcript, ledger rows or children (what the SessionStore's
    stale-route heal, #54878, mints as a route's replacement before any turn runs on it)."""
    return conn.execute(
        'SELECT 1 FROM messages WHERE session_id=? UNION ALL SELECT 1 FROM session_admissions '
        'WHERE target_session_id=? UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? '
        'UNION ALL SELECT 1 FROM sessions WHERE parent_session_id=? LIMIT 1',
        (session_id, session_id, session_id, session_id)).fetchone() is None


def target_has_history(holder, session_id):
    """Whether *session_id* holds a transcript, ledger rows or children, in one read."""
    with holder._read_ctx() as conn:
        return not unused_session(conn, session_id)


def local_session_ids(holder):
    """Every id holding a creation receipt, in store order."""
    from ..conversation._store.local import POLICY_PREFIX
    with holder._read_ctx() as conn:
        return [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            'SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',))]


def local_lineages(holder):
    """``(owner, lineage)`` of every receipt, as the catalog reads them: an undecodable receipt, a non-string owner
    or a non-list lineage is skipped, and non-string members are dropped."""
    from ..conversation._store.local import POLICY_PREFIX
    found = []
    with holder._read_ctx() as conn:
        rows = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
    for (raw,) in rows:
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        owner = receipt.get('session_id') if isinstance(receipt, dict) else None
        lineage = receipt.get('lineage') if isinstance(owner, str) else None
        if isinstance(lineage, list):
            found.append((owner, [physical for physical in lineage if isinstance(physical, str)]))
    return found


def binding_marks(holder, session_id, lineage_of=None):
    """``(policy, legacy, lineage owner)``: whether a creation receipt and a legacy binding exist for *session_id*,
    and with *lineage_of* and either present, ``local_lineage_owner(lineage_of)``, in one read."""
    from ..conversation._store.local import POLICY_PREFIX, local_lineage_owner
    from ..conversation._store.local_migration import LEGACY_PREFIX
    with holder._read_ctx() as conn:
        policy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
        legacy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + session_id,)).fetchone()
        owner = local_lineage_owner(conn, lineage_of) if lineage_of and (policy or legacy) else None
    return policy is not None, legacy is not None, owner


def lineage_owner(holder, session_id):
    """``local_lineage_owner`` in one read."""
    from ..conversation._store.local import local_lineage_owner
    with holder._read_ctx() as conn:
        return local_lineage_owner(conn, session_id)


def conversation_ids(holder, session_id):
    """``local_conversation_ids`` in one read."""
    from ..conversation._store.local import local_conversation_ids
    with holder._read_ctx() as conn:
        return local_conversation_ids(conn, session_id)


def validate_lineage(holder, receipt):
    """``validate_local_lineage`` of the passed receipt in one read."""
    from ..conversation._store.local_lineage import validate_local_lineage
    with holder._read_ctx() as conn:
        return validate_local_lineage(conn, receipt)
