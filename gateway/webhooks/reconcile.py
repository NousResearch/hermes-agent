"""Bounded, fair original-final reconciliation on the existing gateway loop."""
import asyncio
import logging

log = logging.getLogger(__name__)
PASS_LIMIT = 8


def _candidates(authority):
    cursor = getattr(authority, '_webhook_final_cursor', '')
    from gateway.session_persistence import authority_persistence
    from persistence.authority.webhooks import WebhookLedger
    def read(after):
        return authority_persistence(authority).webhook_deliveries.candidates(WebhookLedger.FINAL, after=after, limit=PASS_LIMIT)
    rows = read(cursor)
    if not rows and cursor:
        rows = read('')
    authority._webhook_final_cursor = rows[-1] if rows else ''
    return rows


def restored_adapter(authority, payload):
    """A revoked provenance is no adapter, so the intent's authorizer records refusal."""
    from gateway.session_envelope import restore_native
    from persistence.core.errors import AuthorityRefusal
    try:
        event = restore_native(payload, authority.runner)
    except (AuthorityRefusal, ValueError, KeyError, TypeError):
        return None
    return authority.runner._adapter_for_source(event.source)


async def reconcile_finals(authority):
    from gateway.session_authorities import async_owner_scope
    from gateway.webhooks.presentation import deliver_settled
    from gateway.webhooks.intents import read_intent, _state
    from gateway.session_runtime_workers import tracked_write
    from gateway.session_persistence import authority_persistence
    from gateway.session_authority import admission_row
    if getattr(authority, 'retiring', False) or getattr(authority, 'retired', False):
        return
    async with async_owner_scope(authority):
        rows = await asyncio.to_thread(_candidates, authority)
        for candidate in rows:
            if getattr(authority, 'retiring', False):
                return
            admission_id = candidate
            try:
                row = admission_row(await asyncio.to_thread(authority_persistence(authority).admissions.get, admission_id=admission_id))
                if row is None:
                    value = await asyncio.to_thread(read_intent, authority, admission_id)
                    if value:
                        await tracked_write(authority, lambda: _state(authority, value, 'refused', 'source_unavailable'))
                    continue
                adapter = restored_adapter(authority, row['payload'])
                # A missing/revoked connector must turn an existing obligation into refusal.
                await deliver_settled(authority, adapter, None, admission_id)
            except Exception:  # health: allow BLE001 -- isolate one unavailable origin without exposing its text
                log.warning('Webhook final reconciliation unavailable for source=%s', admission_id)
        from gateway.webhooks.origins import reconcile_origins
        await reconcile_origins(authority)


def schedule_reconciliation(runner, loop):
    """Called from housekeeping's thread; admission/ownership remains on the gateway loop."""
    if runner is None or loop is None or loop.is_closed():
        return
    from gateway.session_authorities import serving_authorities
    from gateway.session_runtime_workers import track_mutation
    def schedule():
        for authority in serving_authorities(runner):
            old = getattr(authority, '_webhook_reconcile_task', None)
            if old is None or old.done():
                authority._webhook_reconcile_task = track_mutation(authority, reconcile_finals(authority))
    loop.call_soon_threadsafe(schedule)
