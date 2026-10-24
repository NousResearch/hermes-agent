"""Receipt-scoped progress hints. A hint is never commit proof or a deadline lease."""
import asyncio

from gateway.session_admission import admission_fingerprint


def receipt_identity(params):
    identity = {key: params[key] for key in (
        'profile_id', 'session_id', 'execution_id', 'generation', 'epoch', 'sequence', 'operation'
    ) if key in params}
    identity['payload_digest'] = admission_fingerprint(
        canonical_target=params.get('session_id', ''),
        payload={'operation': params.get('operation'), 'payload': params.get('payload')})
    return identity


async def observe_write(connection, params, operation):
    """Keep physical write tracking in tracked_write; cancel only this observer."""
    transport = getattr(connection, 'transport', None)
    if transport is None:
        return await operation
    identity = receipt_identity(params)
    def report(phase):
        transport.write({'jsonrpc': '2.0', 'method': 'worker.persist.progress',
                         'params': dict(identity, phase=phase)})
    # The validated owner accepted this request. No storage result is asserted.
    report('storage_pending')
    result = await operation
    report('storage_settled')
    return result
