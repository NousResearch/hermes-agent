"""Externally conducted conversation, recorded by the canonical session owner."""
import asyncio
from dataclasses import asdict

from gateway.session_busy_controls import authorize
from gateway.session_persistence import authority_persistence
from persistence.authority.external_messages import validate_messages
from persistence.core.errors import AuthorityRefusal


async def append_message(connection, ref, params):
    allowed = {'session_id', 'profile', 'role', 'content', 'client_id', 'observed'}
    if set(params) - allowed:
        raise AuthorityRefusal('invalid_params')
    messages = validate_messages([{key: value for key, value in params.items() if key not in {'session_id', 'profile'}}])
    authorize(connection, ref, params, 'session:submit')
    result = await append_messages(connection.authority, connection.actor, ref, messages)
    return {'ok': True, **result['messages'][0]}


async def append_messages(authority, actor, ref, messages):
    from gateway.session_runtime_workers import track_mutation
    authority._require_admission_open()
    authority.authorize(actor, ref, 'session:submit')
    task = track_mutation(authority, _append(authority, actor, ref, messages))
    return await asyncio.shield(task)


async def _append(authority, actor, ref, messages):
    from gateway.session_external_review import memory_interval, schedule_review
    live = authority.sessions[ref.session_id]
    async with live.mutation_lock:
        authority.authorize(actor, ref, 'session:submit')
        interval = await asyncio.to_thread(memory_interval, authority, live)
        result = await asyncio.to_thread(
            authority_persistence(authority).runtime.append_external_messages, epoch=authority.epoch,
            session_id=ref.session_id, messages=messages, memory_interval=interval)
        if result.inserted_ids:
            # Replay/resume and other viewers learn that the transcript changed; never
            # publish message.complete, which would pretend the active task finished.
            live.event_stream.publish(ref.session_id, {'session_id': result.session_id,
                'message_ids': list(result.inserted_ids)}, event_type='session.messages.appended')
        if result.review_due:
            schedule_review(authority, ref, live)
        return {'session_id': result.session_id, 'messages': [asdict(receipt) for receipt in result.messages]}
