"""Cancellation-safe hand-over of an admitted input batch into the running turn.

The authority owns the whole lifecycle: registration, the durable
``steer_session_input`` commit, delivery to the adopted agent, and requeue when
delivery is refused or the turn ends first. The submitting viewer disconnecting
never aborts a committed hand-over, and a later submit cannot overtake it.

Only batches the provider admits are offered; with no provider this module
returns the receipt unchanged, exactly like upstream.
"""

from __future__ import annotations

import asyncio
from functools import partial

from session_input_checkpoint import redirect_part
from session_input_primitives import (
    AdmissionSnapshot, agent_phase, batch_record, has_provider, log, original_payload,
    transfer_mode, wants_transfer,
)


def _interactive(authority, ref) -> bool:
    """A session a person is chatting in, judged by the surface that created it.

    ``sessions.source`` is the live platform label, which a local Bot Chat row stores as
    ``local``; ``created_source`` keeps the originating surface (``desktop``/``cli``/...).
    Either naming an interactive surface counts.
    """
    from gateway.session_queued_controls import _INTERACTIVE_SOURCES, _interactive as upstream
    sid = ref.session_id
    if upstream(authority, sid):
        return True
    with authority.db._read_ctx() as conn:
        row = conn.execute("SELECT created_source FROM sessions WHERE id=?", (sid,)).fetchone()
    return bool(row is not None and row[0] in _INTERACTIVE_SOURCES)


def _candidate(authority, ref, row) -> bool:
    from hermes_state_runtime import list_session_admissions
    if row is None or row['status'] != 'queued':
        log.info('input transfer skip reason=not_queued')
        return False
    if not _interactive(authority, ref):
        log.info('input transfer skip reason=not_interactive')
        return False
    _, record = batch_record(row['payload'])
    if not record:
        log.info('input transfer skip reason=not_a_batch')
        return False
    if str(original_payload(row['payload']).get('text', '')).lstrip().startswith('/'):
        log.info('input transfer skip reason=command')
        return False
    # A command, another sender, or any whole-turn work ahead is a FIFO barrier.
    queued = [item for item in list_session_admissions(authority.db, session_id=ref.session_id)
              if item['status'] == 'queued']
    if not queued or queued[0]['admission_id'] != row['admission_id']:
        log.info('input transfer skip reason=fifo_barrier')
        return False
    return True


async def transfer_after_submit(authority, ref, receipt):
    """The authority owns the entire handoff, even after the submitting viewer disconnects."""
    if not has_provider() or receipt is None:
        return receipt
    from gateway.session_runtime_workers import track_mutation
    task = track_mutation(authority, _serialized_transfer(authority, ref, receipt))
    return await asyncio.shield(task)


async def _serialized_transfer(authority, ref, receipt):
    from gateway.session_authority import keyed_turn
    lanes = vars(authority).setdefault('_input_transfer_lanes', {})
    async with keyed_turn(lanes, ref.session_id):
        return await _transfer(authority, ref, receipt)


def _snapshot(authority, ref, row, admission_id) -> AdmissionSnapshot:
    _, record = batch_record(row['payload'])
    return AdmissionSnapshot(
        profile_id=authority.profile_id,
        session_id=ref.session_id,
        admission_id=admission_id,
        generation=row.get('generation'),
        principal_id=row['principal_id'],
        parts=tuple(record['parts']),
        text=row['payload'].get('text', ''),
    )


async def _transfer(authority, ref, receipt):
    """Commit first, then offer exactly this admission to the running generation.

    Registration and delivery share the authority's settlement fence. Storage never
    runs under the stream lock. A late/refused delivery requeues through the existing
    transfer settlement; retries of terminal/steering rows never deliver twice.
    """
    from gateway.session_queued_controls import _receiving_agent, _register, _unregister
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import get_session_admission, steer_session_input, revert_steer_input, RuntimeStoreError
    if not has_provider():
        return receipt
    sid, admission_id = ref.session_id, receipt.admission_id
    live = authority.sessions.get(sid)
    if live is None:
        log.info('input transfer skip reason=no_live_session')
        return receipt
    row = await asyncio.to_thread(get_session_admission, authority.db, admission_id=admission_id)
    if not _candidate(authority, ref, row):
        return receipt
    with live.event_stream.lock:
        generation = authority.executing.get(sid)
        agent = _receiving_agent(authority, sid, generation) if generation is not None else None
        if agent is None:
            log.info('input transfer skip reason=no_adopted_agent')
            return receipt
        if getattr(agent, '_session_input_policy', None) is None:
            log.info('input transfer skip reason=policy_unbound')
            return receipt
        if (getattr(agent, 'api_mode', None) == 'codex_app_server'
                or getattr(agent, '_hard_interrupt_requested', None) is not None
                and agent._hard_interrupt_requested.is_set()):
            log.info('input transfer skip reason=turn_not_steerable')
            return receipt
        _register(authority, sid, generation, admission_id)
    try:
        committed, fresh = await tracked_write(authority, partial(steer_session_input, authority.db,
            epoch=authority.epoch, admission_id=admission_id, generation=generation))
    except RuntimeStoreError as exc:
        with live.event_stream.lock:
            _unregister(authority, sid, generation, admission_id)
        if exc.reason not in {'session_idle', 'admission_started', 'stale_generation'}:
            raise
        return receipt
    if fresh:
        with live.event_stream.lock:
            current = _receiving_agent(authority, sid, generation)
            delivered = current is agent and _deliver(authority, sid, generation, agent, committed)
        if not delivered:
            log.info('input transfer requeue reason=not_delivered')
            await tracked_write(authority, partial(revert_steer_input, authority.db,
                epoch=authority.epoch, admission_id=admission_id, generation=generation))
            with live.event_stream.lock:
                _unregister(authority, sid, generation, admission_id)
            authority._schedule(ref)
        authority._publish_pending(ref)
    else:
        log.info('input transfer skip reason=not_fresh admission=%s', admission_id)
    current = await asyncio.to_thread(get_session_admission, authority.db, admission_id=admission_id)
    return authority._receipt(current)


def _deliver(authority, sid, generation, agent, row) -> bool:
    entry = authority.steer_transfers.get(sid)
    if (entry is None or entry[0] != generation
            or getattr(agent, '_hard_interrupt_requested', None) is not None
            and agent._hard_interrupt_requested.is_set()):
        return False
    admission_id = row['admission_id']
    if entry[1].get(admission_id):
        return True
    _, record = batch_record(row['payload'])
    parts = record['parts']
    # A hand-over is attempted before the next submit, so normally one part.
    # An already folded backlog stays intact as one ticket with every timestamp.
    text = row['payload']['text'] if len(parts) != 1 else parts[0]['text']
    arrived = parts[0]['at']
    mode = transfer_mode(agent_phase(agent))
    if mode == 'queue':
        return False
    accepted = (agent.redirect(text, ticket=admission_id, arrived=arrived) if mode == 'redirect'
                else agent.steer(text, ticket=admission_id, arrived=arrived))
    entry[1][admission_id] = bool(accepted)
    log.info('input path=fifo_%s accepted=%s generation=%s', mode, bool(accepted), generation)
    return bool(accepted)
