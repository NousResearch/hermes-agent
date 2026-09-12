"""Trusted messaging admission and the existing TurnRunner invocation boundary."""
import asyncio
from contextvars import ContextVar
from contextlib import nullcontext
from dataclasses import replace

from gateway.platforms.event import MessageEvent
from gateway.session_envelope import restore_native

admission_author = ContextVar('admission_author', default=None)
executing_admission = ContextVar('executing_admission', default=False)


async def admit_message(authority, event):
    receipt = await authority.admit_native(event)
    if receipt.status == 'terminal':
        return None
    # Only the delivery waiter is process-local; execution reads the committed snapshot.
    authority.native_waiters.add(receipt.admission_id)
    waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
    return await asyncio.shield(waiter)


async def execute_admission(authority, ref, row):
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is not None and policy.source == 'cron':
        from gateway.session_cron import execute
        return await execute(authority, ref, row, policy)
    from gateway.session_managed_worker import managed_policy, execute_managed
    policy = managed_policy(authority, ref)
    if policy is not None:
        return await execute_managed(authority, ref, row, policy)
    live = authority.sessions[ref.session_id]
    native = row['admission_id'] in authority.native_waiters
    authority.native_waiters.discard(row['admission_id'])
    if 'native_text_v1' in row['payload']:
        event = restore_native(row['payload'], authority.runner)
    else:
        from gateway.session_ingress_media import restore_attachments
        event = MessageEvent(text=row['payload']['text'], source=live.source,
                             message_id=row['admission_id'], **restore_attachments(row['payload']))
        if 'local_automation_v1' in row['payload']:
            from gateway.session_automation import restore_local_automation
            event = restore_local_automation(authority, ref, row)
    provenance = row['payload'].get('native_text_v1', {}).get('provenance')
    from gateway.run import _profile_runtime_scope
    # Under multiplex, owner-side execution runs under the OWNING profile's home (agent build,
    # config, secrets, state.db), never the launch profile's ambient scope; native provenance
    # refines it. A single-profile gateway keeps its ambient scope byte-for-byte.
    scope = nullcontext()
    if getattr(getattr(authority.runner, 'config', None), 'multiplex_profiles', False):
        from gateway.session_authorities import owner_scope
        scope = owner_scope(authority, hydrate_secrets=True)
    if provenance is not None:
        from gateway.session_ingress_context import restore_provenance
        home = restore_provenance(authority.runner, event.source, provenance)
        scope = _profile_runtime_scope(home)
    from gateway.config import Platform
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.session_results import execution_result
    is_api = live.source.platform == Platform.API_SERVER
    prepared = prepare_api_execution(authority, ref, row['payload']) if is_api else None
    if is_api:
        if isinstance(event.text, list):
            event.text = '\n'.join(part['text'] for part in event.text if part.get('type') == 'text')
        event.allow_gateway_control = False
        event.internal = True  # trust comes from the private binding and preclaim, never client JSON
    author_token = admission_author.set(event.metadata.get('turn_author'))
    api_token = api_execution.set(prepared)
    captured = {}
    result_token = execution_result.set(captured)
    token = executing_admission.set(True)
    try:
        with scope:
            response = await authority.runner._handle_message(event)
            result = captured.get('result') or {'final_response': response or '', 'messages': []}
            # The drain commits this under the stream lock so no viewer reads `terminal`
            # before the completion event exists in the replay ring.
            authority.pending_results[row['admission_id']] = {'result': result, 'usage': captured.get('usage', {})}
            if not native and not is_api and response:
                adapter = authority.runner._adapter_for_source(event.source)
                if adapter is not None:
                    await deliver_response(adapter, event, live.route, response)
            return response
    finally:
        executing_admission.reset(token)
        execution_result.reset(result_token)
        api_execution.reset(api_token)
        admission_author.reset(author_token)


async def deliver_response(adapter, event, session_key, response):
    from gateway.platforms.base import _thread_metadata_for_event, _mark_notify_metadata
    text, ttl = adapter._unwrap_ephemeral(response)
    if not text:
        return
    extracted = await adapter._extract_response_content(text, event, session_key, is_ephemeral_response=ttl > 0)
    metadata = _mark_notify_metadata(_thread_metadata_for_event(event))
    results = []
    if extracted.text_content:
        await adapter._send_final_text(event, session_key, extracted.text_content,
                                       metadata, ttl > 0, ttl, results.append)
    await adapter._deliver_attachments(event, extracted, metadata, anything_sent=bool(results),
                                       record_delivery=results.append)


async def dispatch_shared_busy(adapter, event, session_key):
    delivery_event = replace(event, source=replace(event.source))
    response = await adapter._message_handler(event)
    await deliver_response(adapter, delivery_event, session_key, response)
