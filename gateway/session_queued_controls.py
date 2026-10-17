"""Steer and Edit of a still-queued admission: ``prompt.steer_queued`` / ``prompt.replace_queued``.

Both consume the queued row inside one owner write transaction, so it can never also run as its own
queued turn, and a retry replays the recorded outcome. There is no reorder verb: one FIFO per session.

A steer is a TRANSFER to the running claim, not an outcome: the row goes queued -> terminal
``steering`` bound to the claim's generation, and settles with that claim (``steered`` once the claim
completes with the text delivered, ``interrupted`` otherwise; queued again in its own FIFO place when
the text never reached the claim's agent). The fence is ``authority.steer_transfers``, changed only
under the session's ``event_stream.lock`` (the lock the drain clears ``executing`` under): no storage
wait ever happens under that lock, which loop-side receipts, attaches and publication take.
"""
import asyncio
from dataclasses import asdict
from functools import partial

from hermes_state_runtime import (
    STEERING, RuntimeStoreError, check_replaced_input, find_session_admission, get_session_admission,
    replacement_binding, retire_replaced_input, revert_steer_input, steer_session_input,
)

# A steer carries plain text and its presentation only. Attachments, finite runs, classic-CLI turns,
# native envelopes and automation rows are whole turns: neither verb may drop what they carry.
_TEXT_ONLY = frozenset({'text', 'surface_v1', 'display_v1', 'local_operator_v1'})
# Sessions launched from an interactive local surface (classic CLI, TUI, Desktop, ACP editor): their
# plain-text rows are prompt.submit input, whose observers settle a ``steered``/``cancelled`` row.
# cron, kanban, a2a and hosted-room (bot_room) producers read a saved result for every outcome they
# do not name, so a transfer or replacement would strand them: never offered, refused by name.
_INTERACTIVE_SOURCES = frozenset({'cli', 'tui', 'gui', 'acp'})


def handlers(connection):
    return {'prompt.steer_queued': partial(steer_queued, connection),
            'prompt.replace_queued': partial(replace_queued, connection)}


def _interactive(authority, session_id):
    """The session's frozen launch source (a local session's row records it) names a local surface.
    Callers hold the stream lock: a plain read (``get_session`` flushes token counts, a write)."""
    sources = authority.__dict__.setdefault('_launch_sources', {})
    if session_id not in sources:
        with authority.db._read_ctx() as conn:
            found = conn.execute('SELECT source FROM sessions WHERE id=?', (session_id,)).fetchone()
        if found is None:
            return False
        sources[session_id] = found[0]
    return sources[session_id] in _INTERACTIVE_SOURCES


def queued_controls(authority, row):
    """The queued-row verbs this owner serves for ``row``: a client shows only these (an older owner
    lists none, so its clients show none)."""
    payload = row.get('payload') or {}
    text = payload.get('text')
    if (row.get('status') != 'queued' or set(payload) - _TEXT_ONLY or not isinstance(text, str) or not text.strip()
            or (payload.get('display_v1') or {}).get('kind') == 'hidden'
            or not _interactive(authority, row['target_session_id'])):
        return []
    return ['steer', 'replace']


async def _queued(connection, ref, admission_id):
    receipt = await connection.authority.receipt(connection.actor, ref, admission_id)
    return receipt, get_session_admission(connection.authority.db, admission_id=admission_id)


def steer_transfers(authority, session_id, generation):
    """``{admission_id: delivered}`` handed to this claim; the caller holds the stream lock. Read by
    the claim's settlement after its execution returned, when no transfer can still be delivered."""
    entry = authority.steer_transfers.get(session_id)
    return dict(entry[1]) if entry is not None and entry[0] == generation else None


def _register(authority, session_id, generation, admission_id):
    entry = authority.steer_transfers.get(session_id)
    if entry is None or entry[0] != generation:
        entry = authority.steer_transfers[session_id] = (generation, {})
    entry[1].setdefault(admission_id, False)  # a concurrent duplicate never resets a delivery


def _unregister(authority, session_id, generation, admission_id):
    entry = authority.steer_transfers.get(session_id)
    if entry is not None and entry[0] == generation and entry[1].get(admission_id) is False:
        del entry[1][admission_id]


def _receiving_agent(authority, session_id, generation):
    """The agent the running claim adopted, while that claim still executes. None before adoption (a
    latched text would be dropped if construction then failed) and for a turn with no steer seam."""
    adopted_generation, agent = authority.adopted.get(session_id, (None, None))
    if (adopted_generation != generation or authority.executing.get(session_id) != generation
            or not callable(getattr(agent, 'steer', None))):
        return None
    return agent


def _deliver(authority, session_id, generation, agent, admission_id, text):
    """Hand the committed text to the claim's agent; the caller holds the stream lock. False once the
    claim stopped executing (its settlement then decides the row) or its agent refused the text."""
    entry = authority.steer_transfers.get(session_id)
    if (entry is None or entry[0] != generation or admission_id not in entry[1]
            or _receiving_agent(authority, session_id, generation) is not agent):
        return False
    if not entry[1][admission_id]:
        entry[1][admission_id] = bool(agent.steer(text))
    return entry[1][admission_id]


async def steer_queued(connection, ref, params):
    authority, admission_id = connection.authority, params['admission_id']
    authority.authorize(connection.actor, ref, 'session:control')
    receipt, row = await _queued(connection, ref, admission_id)
    if row['status'] == 'terminal' and row['outcome'] in (STEERING, 'steered'):
        return asdict(receipt)  # an exact retry replays the committed transfer
    if receipt.status != 'queued':
        raise RuntimeStoreError('admission_started')
    if 'steer' not in queued_controls(authority, row):
        raise RuntimeStoreError('unsupported_control')
    handle = authority._handle(ref)
    if handle.execution_state != 'running':
        raise RuntimeStoreError('session_idle')
    generation, text, live = handle.execution_generation, row['payload']['text'], authority.sessions[ref.session_id]
    session_id = ref.session_id
    with live.event_stream.lock:
        agent = _receiving_agent(authority, session_id, generation)
        if agent is None:
            # Refused before the row is touched: it stays queued and runs in FIFO order.
            raise RuntimeStoreError('steer_unavailable', field='admission_id',
                                    detail='the running turn has no agent to take a steer yet')
        _register(authority, session_id, generation, admission_id)
    from gateway.session_runtime_workers import tracked_write
    try:
        # The storage wait runs with no lock held; the registration above is the claim's fence.
        committed, fresh = await tracked_write(authority, partial(
            steer_session_input, authority.db, epoch=authority.epoch, admission_id=admission_id, generation=generation))
    except BaseException:
        with live.event_stream.lock:
            _unregister(authority, session_id, generation, admission_id)
        raise
    if fresh:
        with live.event_stream.lock:
            delivered = _deliver(authority, session_id, generation, agent, admission_id, text)
        if not delivered:
            # The claim ended first: the text reached nobody, so the admission is queued again.
            await tracked_write(authority, partial(revert_steer_input, authority.db, epoch=authority.epoch,
                                                   admission_id=admission_id, generation=generation))
            with live.event_stream.lock:
                _unregister(authority, session_id, generation, admission_id)
            authority._publish_pending(ref)
            authority._schedule(ref)
            raise RuntimeStoreError('steer_unavailable', field='admission_id',
                                    detail='the running turn ended before the steer reached it')
        authority._publish_pending(ref)  # the card leaves the queue: its text is in the running turn
    return asdict(authority._receipt(committed))


def publish_transfers(authority, ref, generation):
    """After the claim's settlement: each admission it carried publishes the outcome it settled to
    (``steered``, or ``cancelled`` for an interrupted transfer, as a turn's completion reports it)."""
    live = authority.sessions.get(ref.session_id)
    entry = authority.steer_transfers.get(ref.session_id)
    if live is None or entry is None or entry[0] != generation:
        return
    with live.event_stream.lock:
        authority.steer_transfers.pop(ref.session_id, None)
    for admission_id in entry[1]:
        row = get_session_admission(authority.db, admission_id=admission_id)
        if row is not None and row['status'] == 'terminal' and row['outcome'] in ('steered', 'interrupted'):
            authority._settle_cancelled(ref, admission_id,
                                        outcome='steered' if row['outcome'] == 'steered' else 'cancelled')
    authority._publish_pending(ref)  # an undelivered transfer is queued again


async def replace_queued(connection, ref, params):
    """Cancel the queued admission and admit the edit at the FIFO tail under a new identity, in one
    write. The replacement is bound to the admission it retired: an exact retry (same old admission,
    same new ``submission_id``) answers the same new admission; any other old admission under that
    ``submission_id`` is ``admission_conflict``, decided inside the admission's own transaction."""
    authority, actor, old_id = connection.authority, connection.actor, params['admission_id']
    request = connection.submission(ref, {key: value for key, value in params.items() if key != 'admission_id'})
    receipt, row = await _queued(connection, ref, old_id)
    replacement = await asyncio.to_thread(find_session_admission, authority.db, principal_id=actor.subject,
                                          session_id=ref.session_id, request_id=request.request_id)
    if replacement is None:
        if receipt.status != 'queued':
            raise RuntimeStoreError('admission_started')
        if 'replace' not in queued_controls(authority, row):
            raise RuntimeStoreError('unsupported_control')
        authority.cancel_obligations.add(old_id)  # owed from before the write, like prompt.cancel
    elif (receipt.status, receipt.outcome) != ('terminal', 'cancelled'):
        raise RuntimeStoreError('admission_conflict')  # that request id names other work
    binding = replacement_binding(actor.subject, ref.session_id, request.request_id)
    try:
        admitted = await authority.submit(actor, request, queue_while_restarting=connection._operator is True,
                                          _authorize_write=partial(retire_replaced_input, admission_id=old_id, binding=binding),
                                          _replay_guard=partial(check_replaced_input, admission_id=old_id, binding=binding))
    finally:
        old = get_session_admission(authority.db, admission_id=old_id)
        if old_id in authority.cancel_obligations and (old['status'], old['outcome']) == ('terminal', 'cancelled'):
            authority._settle_cancelled(ref, old_id)
        elif old['status'] == 'queued':
            authority.cancel_obligations.discard(old_id)  # refused: nothing committed
    from gateway.session_ingress_media import release_admission_media
    await asyncio.to_thread(release_admission_media, authority.db, old_id)
    return asdict(admitted)
