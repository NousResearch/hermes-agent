"""Externally conducted conversation, recorded by the canonical session owner."""
import asyncio

from gateway.session_busy_controls import authorize
from hermes_state_external_messages import append_external_messages, validate_messages
from hermes_state_runtime import RuntimeStoreError


async def append_message(connection, ref, params):
    allowed = {'session_id', 'profile', 'role', 'content', 'client_id', 'observed'}
    if set(params) - allowed:
        raise RuntimeStoreError('invalid_params')
    messages = validate_messages([{key: value for key, value in params.items() if key not in {'session_id', 'profile'}}])
    authorize(connection, ref, params, 'session:submit')
    result = await append_messages(connection.authority, connection.actor, ref, messages)
    return {'ok': True, **result['messages'][0]}


async def append_messages(authority, actor, ref, messages):
    from gateway.session_runtime_workers import track_mutation
    messages = validate_messages(messages)
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
        result, inserted, review = await asyncio.to_thread(
            append_external_messages, authority.db, epoch=authority.epoch,
            session_id=ref.session_id, messages=messages, memory_interval=interval)
        if inserted:
            # Replay/resume and other viewers learn that the transcript changed; never
            # publish message.complete, which would pretend the active task finished.
            live.event_stream.publish(ref.session_id, {'session_id': result['session_id'],
                'message_ids': [row['_row_id'] for row in inserted]}, event_type='session.messages.appended')
        if review:
            schedule_review(authority, ref, live)
        return result
