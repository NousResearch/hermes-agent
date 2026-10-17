"""Receipt-only adapter ingress: the authority, not an HTTP/WS task, owns execution."""
import asyncio
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
    # Identity is pinned through the runner's single canonicalization seam (a secondary's own
    # bot, or the primary's routed profile); a rejected primary route is refused downstream.
    if profile:
        runner._canonicalize(source, transport_profile=profile)
        runtime_home = runner._resolve_profile_home_for_source(source) if source.profile else home
    else:
        runtime_home = runner._admit_primary_source(source, home)
        if runtime_home is None:
            _refuse_parked(runner, getattr(source, '_rejected_route_profile', None))
            source.profile_route_rejected = True
            runtime_home = home
    source._authorization_profile_home = home
    # The routed profile's own ledger admits the delivery; an unserved route is refused.
    authority = authority_for_home(runner, runtime_home)
    if authority is None:
        _refuse_parked(runner, getattr(source, 'profile', None) or profile)
        raise RuntimeStoreError('profile_mismatch')
    with native_callback(runner, event, home, profile):
        async with _async_profile_runtime_scope(runtime_home):
            yield authority


def _refuse_parked(runner, profile):
    """A delivery routed to a parked profile is retried, never refused for good: a rescan serves the
    profile again, so it is ``runtime_draining`` (no relay ACK; the connector redelivers), not the
    permanent ``profile_mismatch`` of a route to a profile this gateway does not serve."""
    from gateway.run_runtime import profile_parked
    if profile_parked(runner, profile):
        raise RuntimeStoreError('runtime_draining')


def parked_profile_refusal(runner, profile):
    """A retryable 503 for a webhook to a parked ``/p/<profile>/``, else None. Answered before the
    delivery id is recorded: coalesced, ``deliver_only`` and ``cron_job`` routes act before (or
    without) an admission, so the in-memory id is their only replay guard, and a recorded id would
    answer the provider's retry as a duplicate once the profile is served again."""
    from gateway.run_runtime import profile_parked
    if not profile_parked(runner, profile):
        return None
    from aiohttp import web
    return web.json_response({'error': f'Profile {profile} is not being served; retry this delivery'},
                             status=503, headers={'Retry-After': '30'})


async def admit_producer(adapter, event):
    """Commit a producer delivery; ``None`` when a plugin skipped or consumed it.

    ``pre_gateway_dispatch`` and ``post_gateway_admission`` (#129958) fire once per admission, like
    ``GatewayInboundMixin._handle_message``: before the commit (a skipped or consumed delivery is
    never admitted, a rewrite is what gets committed; a consumer's reply rides
    ``event._consumer_reply`` for the caller's own reply leg), and never for a replay of an
    already-committed delivery (provider retry, relay re-send) nor at execution (PLG-5)."""
    async with producer_scope(adapter, event) as authority:
        from gateway.config import Platform
        from gateway.run_inbound_consumer import POST_ADMISSION_HOOK
        from hermes_cli.lifecycle import has_hook
        webhook = adapter.platform == Platform.WEBHOOK
        hooks = [name for name in ('pre_gateway_dispatch', POST_ADMISSION_HOOK) if has_hook(name)]
        if not (webhook or hooks):
            return await authority.admit_native(event)
        from gateway.session_authority import native_write_started
        payload, principal = await _prepared(authority, event)
        writing = native_write_started.set(False)
        try:
            # An identical delivery (provider retry, relay re-send) overlapping this one waits here, then
            # rechecks the ledger: the retry check, the hooks and the commit run once per delivery.
            async with _delivery_turn((authority.profile_id, principal, event.message_id)):
                committed = payload, _committed(authority, principal, event.message_id)
                return await _admit_once(authority, event, webhook, hooks, committed)
        except (Exception, asyncio.CancelledError) as exc:
            # Refused after preparing (a command-shaped rewrite, a drain during the hooks, a conflict),
            # or cancelled (relay disconnect, shutdown) before the admission write began. A cancelled
            # write still commits (it is shielded): the bytes it commits are never released here.
            if not (isinstance(exc, asyncio.CancelledError) and native_write_started.get()):
                await asyncio.shield(_release_preparation(authority, payload))
            raise
        finally:
            native_write_started.reset(writing)


async def _admit_once(authority, event, webhook, hooks, committed):
    from gateway.run_inbound_consumer import POST_ADMISSION_HOOK
    if webhook:
        prior = await _webhook_retry(authority, event, *committed)
        if prior is not None:
            return prior
    admitted = event
    if hooks and (webhook or not committed[1]):
        runner = authority.runner
        if 'pre_gateway_dispatch' in hooks:
            admitted = await runner._hm_pre_gateway_dispatch_hook(event, event.source)
            if admitted is None:
                await _release_preparation(authority, committed[0])
                return None
            if admitted is not event:
                # The rewrite is committed with the fingerprint of the delivery as received:
                # a provider retry or relay re-send resends that, and resolves to this row.
                from gateway.session_ingress_media import ingress_fingerprint
                admitted._ingress_sha256 = ingress_fingerprint(committed[0], event.raw_message)
        if POST_ADMISSION_HOOK in hooks:
            from gateway.run_inbound_consumer import run_post_admission_hook
            consumed, event._consumer_reply = await run_post_admission_hook(
                runner, admitted, admitted.source, runner._session_key_for_source(admitted.source))
            if consumed:
                await _release_preparation(authority, committed[0])
                return None
    return await authority.admit_native(admitted)


# Producer deliveries in flight by (profile, principal, provider id) -> [lock, holder + waiters]. This
# gateway is the only owner of producer ingress, so in-process coordination covers every request.
_deliveries = {}


@asynccontextmanager
async def _delivery_turn(key):
    if not key[2]:  # no provider id: admission mints one, nothing can overlap
        yield
        return
    entry = _deliveries.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            del _deliveries[key]


async def _release_preparation(authority, payload):
    """A delivery a plugin skipped or consumed, or one refused or cancelled after preparing, is never
    admitted: release the media its preparation (``_prepared``) copied into the retained store. A path
    another admission or an in-flight capture holds is kept, as for a refused admission."""
    from gateway.session_ingress_media import release_unheld_media
    if media := payload['native_text_v1'].get('media'):
        await asyncio.to_thread(release_unheld_media, authority.db, media, retain_history=False)


async def _prepared(authority, event):
    """``(validated payload, principal)`` of this delivery. Preparing the payload runs the same
    authorization as admission, so a refused sender raises here."""
    import json
    from gateway.session_envelope import prepare_native, restore_native
    payload = await prepare_native(authority.runner, event)
    source = restore_native(payload).source
    identity = json.dumps([source.profile, source.platform.value, source.chat_id,
                           source.thread_id, source.user_id], separators=(',', ':'))
    return payload, 'messaging:' + identity


def _committed(authority, principal, request_id):
    """The rows of this delivery's identity in the FIFO."""
    return authority.db._read_all('SELECT admission_id FROM session_admissions '
        'WHERE principal_id=? AND request_id=?', (principal, request_id))

async def _webhook_retry(authority, event, payload, rows):
    # One-shot finalization ends the route. Resolve an existing provider receipt
    # before SessionStore would rotate it into a new physical session.
    from gateway.session_ingress_media import ingress_fingerprint
    from hermes_state_runtime import get_session_admission
    if not rows:
        return None
    row = get_session_admission(authority.db, admission_id=rows[0]['admission_id'])
    retained = row['payload']['native_text_v1']
    if 'ingress_sha256' in retained:
        # Committed rewritten: the provider resends the delivery as received.
        same = retained['ingress_sha256'] == ingress_fingerprint(payload, event.raw_message)
    else:
        if 'webhook_delivery' not in retained and 'webhook_route' not in retained:
            # A pre-destination receipt is still a receipt, not permission to infer
            # again. Do not retrofit today's destination into its queued execution.
            payload['native_text_v1'].pop('webhook_delivery')
            payload['native_text_v1'].pop('webhook_route')
            if 'provenance' not in retained:
                payload['native_text_v1'].pop('provenance', None)
        same = row['payload'] == payload
    if len(rows) != 1 or not same:
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
