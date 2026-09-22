"""Receipt-only adapter ingress: the authority, not an HTTP/WS task, owns execution."""
from contextlib import asynccontextmanager

from hermes_state_runtime import RuntimeStoreError


@asynccontextmanager
async def producer_scope(adapter, event):
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_ingress_context import native_callback

    from gateway.session_authorities import authority_for_home
    runner = getattr(adapter._message_handler, '__self__', None)
    if getattr(runner, 'session_authority', None) is None:
        raise RuntimeStoreError('not_found')
    registered, profile = runner._owning_profile(adapter, adapter.platform)
    home = getattr(runner, '_native_transport_homes', {}).get(profile)
    if not registered or home is None:
        raise RuntimeStoreError('profile_mismatch')
    source = event.source
    if profile:
        runner._stamp_event_profile(event, profile)
    elif not source.profile and not runner._stamp_routed_profile(source):
        source.profile_route_rejected = True
    source._authorization_profile_home = home
    runtime_home = runner._resolve_profile_home_for_source(source) if source.profile else home
    # The routed profile's own ledger admits the delivery; an unserved route is refused.
    authority = authority_for_home(runner, runtime_home)
    if authority is None:
        raise RuntimeStoreError('profile_mismatch')
    with native_callback(runner, event, home, profile):
        async with _async_profile_runtime_scope(runtime_home):
            yield authority


async def admit_producer(adapter, event):
    async with producer_scope(adapter, event) as authority:
        from gateway.config import Platform
        if adapter.platform == Platform.WEBHOOK:
            prior = await _webhook_retry(authority, event)
            if prior is not None:
                return prior
        return await authority.admit_native(event)


async def _webhook_retry(authority, event):
    # One-shot finalization ends the route. Resolve an existing provider receipt
    # before SessionStore would rotate it into a new physical session.
    import json
    from gateway.session_envelope import prepare_native, restore_native
    from hermes_state_runtime import get_session_admission
    payload = await prepare_native(authority.runner, event)
    source = restore_native(payload).source
    identity = json.dumps([source.profile, source.platform.value, source.chat_id,
                           source.thread_id, source.user_id], separators=(',', ':'))
    rows = authority.db._read_all('SELECT admission_id FROM session_admissions '
        'WHERE principal_id=? AND request_id=?', ('messaging:' + identity, event.message_id))
    if not rows:
        return None
    row = get_session_admission(authority.db, admission_id=rows[0]['admission_id'])
    retained = row['payload']['native_text_v1']
    if 'webhook_delivery' not in retained and 'webhook_route' not in retained:
        # A pre-destination receipt is still a receipt, not permission to infer
        # again. Do not retrofit today's destination into its queued execution.
        payload['native_text_v1'].pop('webhook_delivery')
        payload['native_text_v1'].pop('webhook_route')
        if 'provenance' not in retained:
            payload['native_text_v1'].pop('provenance', None)
    if len(rows) != 1 or row['payload'] != payload:
        raise RuntimeStoreError('admission_conflict')
    event._webhook_duplicate = True
    receipt = authority._receipt(row)
    finalize_webhook(authority, receipt)
    return receipt


def finalize_webhook(authority, receipt):
    """End only the settled admitted generation, atomically with the epoch fence."""
    import time
    if receipt.status != 'terminal' or receipt.execution_generation is None:
        return False
    sid = receipt.ref.session_id
    def write(conn):
        return authority.db._end_and_bump(conn, """
            UPDATE sessions SET ended_at=?, end_reason=?
            WHERE id=? AND ended_at IS NULL AND runtime_generation=?
              AND EXISTS (SELECT 1 FROM runtime_epoch WHERE singleton=1 AND epoch=?)
              AND EXISTS (SELECT 1 FROM session_admissions WHERE admission_id=?
                AND target_session_id=? AND status='terminal' AND generation=? AND owner_epoch=?)
              AND NOT EXISTS (SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal')
            """, (time.time(), 'webhook_complete', sid, receipt.execution_generation,
                  authority.epoch, receipt.admission_id, sid, receipt.execution_generation,
                  receipt.authority_epoch, sid), sid, 'webhook_complete')
    return bool(authority.db._execute_write(write))


async def recover_webhook_finalizations(authority):
    """Close settled original webhooks only after current route authorization succeeds."""
    from gateway.session_envelope import check_native_route
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError, get_session_admission

    rows = authority.db._read_all("""
        SELECT a.admission_id FROM sessions AS s
        JOIN session_admissions AS a ON a.target_session_id=s.id
        WHERE s.source='webhook' AND s.ended_at IS NULL
          AND a.status='terminal' AND a.generation IS NOT NULL
          AND json_extract(a.payload_json, '$.native_text_v1.source.platform')='webhook'
          AND json_extract(a.payload_json, '$.native_text_v1.automation') IS NULL
        """)
    results = {}
    for candidate in rows:
        row = get_session_admission(authority.db, admission_id=candidate['admission_id'])
        if row is None:
            continue
        sid = row['target_session_id']
        try:
            envelope = row['payload']['native_text_v1']
            entry = authority.runner.session_store.lookup_by_session_key(envelope['route'])
            if entry is None or authority.logical_owner(entry.session_id) != sid:
                raise RuntimeStoreError('admission_conflict')
            adapter = authority.runner._adapter_for_source(entry.origin) if entry.origin is not None else None
            target = authority.physical_target(SessionRef(authority.profile_id, sid))
            await check_native_route(authority.runner, row['payload'], target, entry.origin, adapter)
            results[sid] = 'finalized' if finalize_webhook(
                authority, authority._receipt(row)) else 'unchanged'
        except (KeyError, RuntimeStoreError):
            results[sid] = 'unavailable'
    return results
