"""Closed delegation-ledger operations in the existing worker receipt transaction.

The caller must supply the verified registration's execution/PID/birth, never
operation payload identity. Each operation is one ledger verb of the profile's
storage interface handle, run on the worker's payload as sent.
"""
from dataclasses import asdict

from persistence.core.errors import AuthorityRefusal

# The delivery operations and the ledger verb each one runs.
_DELIVERY = {'claim': 'claim_delegation_delivery', 'ack': 'ack_delegation_delivery',
             'drop': 'drop_delegation_delivery', 'defer': 'defer_delegation_delivery',
             'release': 'release_delegation_delivery'}


def _probe(worker_pid, worker_birth):
    """``(owner_started_at, refusal)`` of the dispatching worker's process: its start time when the
    process is alive with *worker_birth*, else ``permission_denied`` (another birth) or
    ``worker_not_live`` (gone)."""
    import psutil
    from gateway.status import get_process_start_time
    from hermes_cli import process_identity
    try:
        if process_identity.worker_birth(psutil.Process(worker_pid)) != worker_birth:
            return None, 'permission_denied'
        return get_process_start_time(worker_pid), None
    except psutil.Error:
        return None, 'worker_not_live'


def persist_worker_delegation(store, *, epoch, execution_id, session_id, generation,
                              sequence, operation, payload, worker_pid, worker_birth):
    """Closed ledger sibling of the worker's ``worker.persist`` operations, sharing its receipt stream.
    *store* is the profile's storage interface handle; returns the receipt, ``{'value': ...}``.

    Gateway calls this ONLY after its ordinary worker claim verification. The
    verified process identity is deliberately separate from the operation payload.
    """
    from persistence.workers.delegations import WorkerDelegationPayload
    workers = store.worker_executions
    name = operation.removeprefix('delegation.') if isinstance(operation, str) and operation.startswith('delegation.') else None
    receipt = dict(epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation,
                   sequence=sequence, worker_pid=worker_pid, worker_birth=worker_birth)
    sent = WorkerDelegationPayload(payload)
    if name == 'dispatch':
        owner_started_at, refused = _probe(worker_pid, worker_birth)
        workers.dispatch_delegation(**receipt, dispatch=sent, owner_started_at=owner_started_at,
                                    probe_refused=refused)
        return {'value': None}
    if name == 'complete':
        workers.complete_delegation(**receipt, completion=sent)
        return {'value': None}
    if name == 'child':
        workers.record_delegation_child(**receipt, child=sent)
        return {'value': None}
    if name == 'read':
        return {'value': asdict(workers.read_delegation(**receipt, delegation_id=sent))}
    if name == 'unscheduled':
        workers.withdraw_delegation(**receipt, delegation_id=sent)
        return {'value': None}
    if name in _DELIVERY:
        return {'value': getattr(workers, _DELIVERY[name])(**receipt, claim=sent)}
    raise AuthorityRefusal('invalid_params')


async def worker_delegation_request(connection, ref, params):
    """worker.persist delegation-family hook; all authentication stays mandatory."""
    import asyncio
    from gateway.session_worker import _SCOPE, _claim, _verify
    if set(params) != _SCOPE | {'epoch', 'sequence', 'operation', 'payload'}:
        raise AuthorityRefusal('invalid_params')
    claim = _claim(connection, ref, params)
    authority = connection.authority
    if 'worker:adopt' not in connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    _verify(connection, ref, params, claim)

    from gateway.session_persistence import authority_persistence
    store = authority_persistence(authority)

    def persist_and_publish():
        result = persist_worker_delegation(store,
            **{key: params[key] for key in ('epoch', 'execution_id', 'session_id', 'generation',
                                           'sequence', 'operation', 'payload')},
            worker_pid=params['pid'], worker_birth=params['birth'])
        if params['operation'] == 'delegation.complete':
            event = store.worker_executions.pending_delegation_event(
                execution_id=params['execution_id'], session_id=ref.session_id,
                delegation_id=params['payload']['event']['delegation_id'])
            if event is not None:
                from tools.process_registry import process_registry
                # Queueing after commit can repeat after a lost ACK. Existing
                # token-guarded claims arbitrate delivery; never ACK at publication.
                process_registry.completion_queue.put(dict(event) | {'restored': True})
        return result
    return await asyncio.to_thread(persist_and_publish)
