"""Bounded, fair original-final reconciliation on the existing gateway loop."""
import asyncio
import logging
from gateway.webhook_final_intent import PREFIX

log = logging.getLogger(__name__)
PASS_LIMIT = 8


def _candidates(authority):
    cursor = getattr(authority, '_webhook_final_cursor', '')
    query = """SELECT origin_id FROM (
        SELECT a.admission_id AS origin_id FROM session_admissions AS a
        WHERE a.status IN ('terminal','unknown')
          AND json_extract(a.payload_json, '$.native_text_v1.source.platform')='webhook'
          AND (json_extract(a.payload_json, '$.native_text_v1.automation') IS NULL
            OR json_extract(a.payload_json, '$.native_text_v1.webhook_root') IS NOT NULL)
          AND (json_extract(a.payload_json, '$.native_text_v1.webhook_delivery.deliver')='bot-chat'
            OR json_extract(a.payload_json, '$.native_text_v1.webhook_delivery.deliver') LIKE 'bot-chat:%')
          AND NOT EXISTS (SELECT 1 FROM state_meta AS m WHERE m.key=? || a.admission_id)
        UNION
        SELECT json_extract(value, '$.origin_id') AS origin_id FROM state_meta
        WHERE key>=? AND key<? AND json_extract(value, '$.state') NOT IN ('settled','suppressed','refused','failed')
        ) WHERE origin_id>? ORDER BY origin_id LIMIT ?"""
    def read(after):
        return authority.db._read_all(query, (PREFIX, PREFIX, PREFIX + '~', after, PASS_LIMIT))
    rows = read(cursor)
    if not rows and cursor:
        rows = read('')
    authority._webhook_final_cursor = rows[-1]['origin_id'] if rows else ''
    return rows


def restored_adapter(authority, payload):
    """A revoked provenance is no adapter, so the intent's authorizer records refusal."""
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import RuntimeStoreError
    try:
        event = restore_native(payload, authority.runner)
    except (RuntimeStoreError, ValueError, KeyError, TypeError):
        return None
    return authority.runner._adapter_for_source(event.source)


async def reconcile_finals(authority):
    from gateway.session_authorities import async_owner_scope
    from gateway.platforms.webhook_bot_chat import deliver_settled
    from gateway.webhook_final_intent import read_intent, _state
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import get_session_admission
    if getattr(authority, 'retiring', False) or getattr(authority, 'retired', False):
        return
    async with async_owner_scope(authority):
        rows = await asyncio.to_thread(_candidates, authority)
        for candidate in rows:
            if getattr(authority, 'retiring', False):
                return
            admission_id = candidate['origin_id']
            try:
                row = await asyncio.to_thread(get_session_admission, authority.db, admission_id=admission_id)
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
        from gateway.webhook_origins import reconcile_origins
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
