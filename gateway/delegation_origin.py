"""Resolve a delegation's connector proof from its own admitted request.

Only the gateway-owned ledger supplies the sender. No room origin, current speaker,
client metadata, or completion payload may select another user's authorization.
"""
import asyncio
import json
from datetime import datetime

from hermes_state_runtime import RuntimeStoreError, find_session_admission


TRANSPORT_FIELDS = ('platform', 'chat_id', 'chat_type', 'thread_id', 'scope_id', 'profile')


def _matches_dispatch(envelope, entry, task, dispatched_at):
    if (not envelope or 'provenance' not in envelope or 'automation' in envelope
            or envelope.get('route') != entry.session_key):
        return False
    source = envelope['source']
    return (source.get('user_id') == task['user_id']
            and (not task.get('scope_id') or source.get('scope_id') == task['scope_id'])
            and (not task.get('message_id') or envelope.get('event', {}).get('message_id') == task['message_id'])
            and datetime.fromisoformat(envelope['timestamp']).timestamp() <= dispatched_at)


def _dispatch_envelope(db, entry, owner, task, dispatched_at):
    # Same durable principal as native admission. The origin supplies only the
    # transport, never the sender; the latter comes exclusively from task_json.
    source = entry.origin
    principal = 'messaging:' + json.dumps([source.profile, source.platform.value, source.chat_id,
                                          source.thread_id, task['user_id']], separators=(',', ':'))
    if task.get('message_id'):
        admitted = find_session_admission(db, principal_id=principal, session_id=owner,
                                          request_id=task['message_id'])
        if admitted is not None and admitted.get('outcome') != 'cancelled':
            envelope = admitted['payload'].get('native_text_v1')
            if _matches_dispatch(envelope, entry, task, dispatched_at):
                return envelope
    # Legacy/chained dispatches lack a message pin; edits have suffixed request
    # IDs. Read only this sender's admissions via the existing principal index,
    # one payload at a time, off the event loop. No history-sized Python list or
    # arbitrary age/count cutoff that would strand a long-running worker.
    with db._read_ctx() as conn:
        rows = conn.execute('SELECT payload_json FROM session_admissions '
                            'WHERE principal_id=? AND target_session_id=? '
                            "AND outcome IS NOT 'cancelled' ORDER BY seq DESC", (principal, owner))
        for admitted in rows:
            envelope = json.loads(admitted['payload_json']).get('native_text_v1')
            if _matches_dispatch(envelope, entry, task, dispatched_at):
                return envelope
    raise RuntimeStoreError('admission_conflict')


async def delegation_route_envelope(authority, entry, identity, *, authorized=None):
    """Return verified dispatch proof, or None for non-delegation/senderless legacy records.

    Older records may lack scope_id and message_id: require a matching admitted
    sender before dispatch, never a later speaker or normalized automation timestamp.
    Existing owner/reset and live connector checks still run at admission and execution.
    Missing ledger/parent proof or an unsettled final result fails closed; only
    an existing senderless legacy record retains the prior single-sender policy.
    """
    try:
        producer = json.loads(identity)
    except (TypeError, ValueError):
        return None
    if not isinstance(producer, list) or len(producer) != 3 or producer[0] != 'async_delegation':
        return None
    interim = isinstance(producer[2], str) and producer[2].startswith('task_failure:')
    if producer[2] and not interim:
        return None
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT origin_session, parent_session_id, state, dispatched_at, task_json '
                           'FROM async_delegations WHERE delegation_id=?', (producer[1],)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    task = json.loads(row['task_json'] or '{}')
    # No invented sender or broader policy for producers without retained proof.
    if not task.get('user_id'):
        return None
    verdict, owner, _ = await authority.runner._lookup_completion_owner(
        row['parent_session_id'], entry.session_id)
    # Match the ledger's replay predicate; interim failures belong to a running batch.
    if (row['origin_session'] != entry.session_key or (row['state'] == 'running' and not interim)
            or verdict != 'deliver'
            or authority.logical_owner(owner) != authority.logical_owner(entry.session_id)):
        raise RuntimeStoreError('admission_conflict')
    envelope = await asyncio.to_thread(_dispatch_envelope, authority.db, entry,
        authority.logical_owner(entry.session_id), task, row['dispatched_at'])
    # Share only within this batch admission, never across turns or preclaim.
    # Different connector proofs or role requirements still get their own check.
    source = {key: value for key, value in envelope['source'].items() if key != 'message_id'}
    grant = json.dumps([source, envelope['provenance'], envelope.get('reauthorize')], sort_keys=True)
    if authorized is None or grant not in authorized:
        await reauthorize_sender(authority.runner, envelope)
        if authorized is not None:
            authorized.add(grant)
    return envelope


async def reauthorize_sender(runner, envelope):
    """Reuse native live authorization, including fresh Discord role membership."""
    from gateway.session_envelope import restore_native_source, _validate_native
    from gateway.session_ingress_context import reauthorize_roles
    from gateway.platforms.event import MessageEvent
    source = restore_native_source({'native_text_v1': envelope}, runner)
    provenance = envelope['provenance']
    fresh_roles = await reauthorize_roles(runner, source, provenance)
    _validate_native(runner, MessageEvent(text='', source=source), provenance, fresh_roles)
