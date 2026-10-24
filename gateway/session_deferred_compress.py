"""A /compress sent while a turn runs waits for that turn instead of being refused.

The request is authorized and parsed when it arrives and kept on the live session (one per
session; a newer one replaces it). The drain runs it at the next turn boundary, before it claims
the next queued message, through ``session.mutate``'s own compress path, so it sees the turn's
reply. A bare ``/compress`` is dropped when the turn already compacted the context on its own:
compressing twice in a row only throws away more of the history.
"""
import contextlib
import logging

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

QUEUED_MESSAGE = '/compress queued: it runs when the current reply finishes.'


def _compactions(authority, ref, live):
    from gateway.session_usage import carried_compression_count
    agent = None
    with contextlib.suppress(AttributeError, KeyError):  # a runner without an agent cache has none
        agent = authority.agent(ref)
    return carried_compression_count(live, agent)


def defer_compress(authority, actor, ref, owner, params, live):
    """Keep *params* (an authorized compress refused only because a turn holds the session) for the
    next boundary. ``None`` for a ``--preview``: a read-only report has nothing to wait for."""
    from gateway.session_mutation_compress import compress_request
    if compress_request(params['payload']).preview:
        return None
    row = authority.db.get_session(ref.session_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    live.deferred_compress = {'actor': actor, 'owner': owner, 'request_id': params['request_id'],
                              'payload': params['payload'], 'compactions': _compactions(authority, ref, live)}
    return {'session_id': ref.session_id, 'operation': 'compress', 'status': 'deferred', 'message': QUEUED_MESSAGE,
            'revision': row['runtime_revision'], 'execution_generation': row['runtime_generation']}


def _announce(live, ref, kind, text):
    live.event_stream.publish(ref.session_id, {'kind': kind, 'text': text}, event_type='status.update')


async def apply_deferred_compress(authority, ref, live):
    """Run a deferred compress. The drain calls this between turns, holding neither lock."""
    import asyncio
    from gateway.session_mutations import _mutate_session
    pending = live.deferred_compress
    if pending is None:
        return
    live.deferred_compress = None
    if not pending['payload'] and _compactions(authority, ref, live) > pending['compactions']:
        _announce(live, ref, 'compacted', 'Queued /compress skipped: the context was already compacted during the reply.')
        return
    row = await asyncio.to_thread(authority.db.get_session, ref.session_id)
    if row is None:
        return
    params = {'session_id': ref.session_id, 'request_id': 'deferred-compress:' + pending['request_id'],
              'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
              'operation': 'compress', 'payload': pending['payload']}
    try:
        async with live.claim_gate:
            result = await _mutate_session(authority, pending['actor'], ref, params,
                                           commit_lock=live.mutation_lock, owner=pending['owner'])
    except RuntimeStoreError as exc:
        if exc.reason == 'session_busy':
            live.deferred_compress = pending  # a worker still holds the transcript: next boundary
            return
        _announce(live, ref, 'warn', f'Queued /compress did not run: {exc.reason}')
        return
    except Exception:  # health: allow BLE001 -- runs before the claim: escaping would wedge the drain
        logger.exception('Deferred compress on %s failed', ref.session_id)
        _announce(live, ref, 'warn', 'Queued /compress failed; run /compress again.')
        return
    summary = result.get('summary') or {}
    _announce(live, ref, 'compacted', summary.get('headline') or 'Queued /compress complete.')
