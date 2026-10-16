"""Shared read status belongs to the runtime owner, never a second DB writer."""
import asyncio
import concurrent.futures
import sqlite3

from hermes_state_runtime import RuntimeStoreError

FORWARD_TIMEOUT_SECONDS = 10.0


def operator_read_status(actor, params):
    """An authenticated dashboard operator may acknowledge any visible chat.

    This changes only the shared unread marker. It confers no history binding,
    execution permission, title/archive control or native transport identity.
    """
    payload = params.get('payload')
    return ('session:operator' in actor.capabilities and params.get('operation') == 'sidebar'
            and isinstance(payload, dict) and set(payload) == {'unread'}
            and type(payload['unread']) is bool)


async def _apply(runner, envelope):
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_mutations import mutate_session

    if set(envelope) != {'instance_id', 'profile_id', 'principal_id', 'params'}:
        raise RuntimeStoreError('invalid_params')
    descriptor = runner.session_runtime_descriptor
    if (descriptor.get('state') != 'ready' or runner._draining
            or envelope['instance_id'] != descriptor.get('instance_id')):
        raise RuntimeStoreError('runtime_coordination_required')
    profile_id, subject = envelope['profile_id'], envelope['principal_id']
    if not isinstance(profile_id, str) or not isinstance(subject, str) or not subject or len(subject) > 1024:
        raise RuntimeStoreError('invalid_params')
    authority = authority_for_profile_id(runner, profile_id)
    if authority is None:
        raise RuntimeStoreError('profile_mismatch')
    actor = Principal(subject, authority.profile_id,
        frozenset({'session:read', 'session:control', 'session:operator'}), 'dashboard-read-status')
    params = envelope['params']
    if not isinstance(params, dict) or not operator_read_status(actor, params):
        raise RuntimeStoreError('invalid_params')
    with owner_scope(authority):
        return await mutate_session(authority, actor,
            SessionRef(authority.profile_id, params.get('session_id')), params)


def install_read_status_forwarder(runner, server, loop):
    """Only the existing same-OS-user private control channel accepts attestations.

    The dashboard authenticates its operator before forwarding. Preserve that
    principal for receipts; do not impersonate the OS owner or mint native grants.
    """
    def forward(envelope, peer):
        if not peer:
            raise PermissionError('authenticated private request required')
        future = asyncio.run_coroutine_threadsafe(_apply(runner, envelope), loop)
        try:
            return {'result': future.result(timeout=FORWARD_TIMEOUT_SECONDS)}
        except RuntimeStoreError as exc:
            return {'error': exc.reason}
        except sqlite3.Error:
            return {'error': 'storage_unavailable'}
        except concurrent.futures.TimeoutError:
            # The owner retains its mutation task through settlement. Do not
            # cancel, repeat or send the write through offline maintenance.
            return {'error': 'mutation_outcome_unknown'}
    server.private_handlers['session-read-status'] = forward
