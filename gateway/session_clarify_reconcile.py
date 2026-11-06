"""Read-only reconciliation for the canonical batch-clarify request registry."""
from hermes_state_runtime import RuntimeStoreError


async def clarify_reconcile(connection, ref, params):
    from gateway.session_server_requests import RequestIdentityMismatch, may_settle, reconcile

    request_id = params.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeStoreError('invalid_params')
    authority, actor = connection.authority, connection.actor

    def entitled(session_id, events):
        return may_settle(authority, actor, session_id, events)

    try:
        return reconcile(request_id, session_ok=entitled,
                         authority_epoch=params.get('authority_epoch'),
                         execution_generation=params.get('execution_generation'))
    except RequestIdentityMismatch as exc:
        raise RuntimeStoreError(exc.reason) from None
