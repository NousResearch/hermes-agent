"""Fence an unsettled claim in a live owner without inventing a terminal result."""
import asyncio
import logging
import sqlite3

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch

logger = logging.getLogger(__name__)


def _mark_unknown(authority, row):
    def write(conn):
        _epoch(conn, authority.epoch)
        current = _admission(conn, row['admission_id'])
        if current['owner_epoch'] != authority.epoch or current['generation'] != row['generation']:
            raise RuntimeStoreError('stale_generation')
        if current['status'] in {'terminal', 'unknown'}:
            return current['status']
        if current['status'] != 'started':
            raise RuntimeStoreError('stale_generation')
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return 'unknown'
    return authority.db._execute_write(write)


async def recover_failed_settlement(authority, row):
    # A full disk or held SQLite writer may also block the recovery stamp. Retain this owner
    # and its captured result until the storage fence can commit; never manufacture a reply.
    while True:
        try:
            from gateway.session_runtime_workers import track_mutation
            writer = track_mutation(authority, asyncio.to_thread(_mark_unknown, authority, row))
            return await asyncio.shield(writer)
        except (OSError, sqlite3.Error):
            await asyncio.sleep(1)


def completion_payload(row, settled, response, captured):
    # ``status`` is the message.complete contract's TurnStatus: the Desktop
    # extends a Stopped bubble to the persisted partial only on 'interrupted'.
    complete = {
        'text': response, 'content': response, 'admission_id': row['admission_id'],
        'outcome': 'cancelled' if settled['outcome'] == 'interrupted' else settled['outcome'],
        'status': {'completed': 'complete', 'interrupted': 'interrupted'}.get(
            settled['outcome'], 'error')}
    # Only the agent's reuse site sets this (never inferred from equal text): the
    # final repeats a reply the viewer already painted, so it settles in place.
    captured_result = (captured or {}).get('result') or {}
    if response and captured_result.get('response_reused'):
        complete['response_reused'] = True
    # The committed row addresses of the turn: a viewer binds the streamed reply to
    # its stored row, so a transcript read racing this frame never paints it twice.
    # ``submission_id`` names whose turn it is: the sending viewer binds its optimistic
    # prompt (``user-<submission_id>``), which the queued admission ack could not name.
    if isinstance(captured_result.get('persisted_turn'), dict):
        complete['persisted_turn'] = {**captured_result['persisted_turn'],
                                      'submission_id': row['request_id']}
    return complete


def publish_terminal(authority, ref, row, settled, response, captured):
    """Cleanup cannot withhold a committed outcome; publication is attempted once."""
    live = authority.sessions[ref.session_id]
    try:
        live.controls.snapshot(ref.session_id, None)
    except Exception:
        logger.exception('Terminal control cleanup failed for admission %s', row['admission_id'])
    try:
        from gateway.session_ingress_media import release_admission_media
        release_admission_media(authority.db, row['admission_id'])
    except Exception:
        logger.exception('Terminal media cleanup failed for admission %s', row['admission_id'])
    live.event_stream.publish(ref.session_id, completion_payload(row, settled, response, captured))
    # Idle follows the completion so a viewer cannot mistake a settled turn for a lost frame.
    authority._publish_pending(ref)
