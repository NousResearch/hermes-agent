"""Typed terminal process notices, not a new meaning for arbitrary adapter.send."""
import asyncio
from functools import partial
import json

from hermes_state_runtime import RuntimeStoreError


async def present_process_final(runner, watcher, session, mode):
    from gateway.warning_notifications import present_notification
    selected = mode in {'concise', 'all', 'result'} or (mode == 'error' and session.exit_code not in {0, None})
    platform = watcher.get('platform', '')
    if platform == 'webhook' and await _webhook_process(runner, watcher, session, mode, selected):
        return
    if selected:
        message = runner._format_process_final_message(watcher['session_id'], session, mode)
        await present_notification(lambda: runner._send_watcher_message(platform, watcher.get('chat_id', ''),
            watcher.get('thread_id', ''), message, watcher), platform=platform,
            diagnostic=session.exit_code not in {0, None})


async def _webhook_process(runner, watcher, session, mode, selected):
    from gateway.session_authorities import active_authority
    from gateway.session_automation import snapshot_automation
    from gateway.platforms.event import MessageEvent
    from gateway.webhook_origins import read_origin, _identity, _begin, _finish, _reconcile_origin
    from gateway.platforms.webhook_ingress import _delivery_turn
    from gateway.session_runtime_workers import tracked_write, track_mutation
    from gateway.warning_notifications import present_notification
    authority = active_authority(runner)
    if authority is None:
        return False
    source = await asyncio.to_thread(runner._build_process_event_source, watcher)
    if source is None or source.platform.value != 'webhook':
        return False
    identity = runner._completion_delivery_identity(dict(type='completion', session_id=session.id, started_at=session.started_at))
    if identity is None or not session.exited:
        return True  # no incarnation, no new Bot Chat turn
    adapter = runner._adapter_for_source(source)
    event = MessageEvent(text='', source=source, internal=True, metadata={
        'gateway_session_key': watcher.get('session_key'),
        'gateway_session_id': watcher.get('parent_session_id') or session.parent_session_id})
    request = json.dumps(identity, separators=(',', ':'))
    try:
        payload, _entry = snapshot_automation(authority, adapter, event, request)
    except RuntimeStoreError:
        return True
    native = payload['native_text_v1']
    if 'webhook_root' not in native:
        return False
    principal = 'webhook-process:' + native['route']
    key = _identity(authority, principal, request, native)
    event.message_id = request
    async def run():
        async with _delivery_turn((authority.profile_id, principal, request)):
            value = read_origin(authority, key)
            if value is None:
                # Formatting, policy and commit remain in one owned task. The process already
                # exited; no agent result/admission is fabricated to act as its provenance.
                message = runner._format_process_final_message(session.id, session, mode) if selected else ''
                owed = []
                async def present():
                    owed.append(True)
                if selected:
                    await present_notification(present, platform='webhook', diagnostic=session.exit_code not in {0, None})
                value = await tracked_write(authority, partial(_begin, authority, key, principal, event, native, request))
                value.update(process_id=session.id, process_started_at=session.started_at)
                await tracked_write(authority, partial(_finish, authority, value, 'process_final', message,
                    outcome='failed' if session.exit_code not in {0, None} else 'completed', muted=not owed))
            return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))
    await asyncio.shield(track_mutation(authority, run()))
    return True
