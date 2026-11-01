"""Authority-owned native image transfers; public queued controls remain text-only."""
import asyncio
import base64
import logging
from dataclasses import replace
from functools import partial

from hermes_state_runtime import RuntimeStoreError, get_session_admission, revert_steer_input, steer_session_input
from gateway.session_queued_controls import _receiving_agent, _register, _unregister
from gateway.session_runtime_workers import tracked_write

logger = logging.getLogger(__name__)


def _prepare(authority, row, agent):
    from agent.auxiliary_client import scoped_runtime_main
    from agent.image_routing import build_native_content_parts, _transcode_to_png
    from gateway.session_envelope import restore_native
    from gateway.run import _event_media_is_image

    runner = authority.runner
    event = restore_native(row['payload'], runner)
    key = row['payload']['native_text_v1']['route']
    # Ingress workers inherit the caller's context, not the adopted agent's runtime.
    runtime = {name: getattr(agent, name, '') for name in (
        'provider', 'model', 'requested_provider', 'base_url', 'api_key', 'api_mode', 'auth_mode',
    )}
    with runner._profile_scope_for_source(event.source), scoped_runtime_main(runtime):
        if (not event.media_urls or event.internal or not event.allow_gateway_control
                or runner._effective_busy_input_mode(event.source) != 'steer'
                or agent.api_mode == 'codex_app_server' or not agent._model_supports_vision()
                or not all(_event_media_is_image(event, i) for i in range(len(event.media_urls)))
                or runner._decide_image_input_mode(source=event.source, session_key=key,
                                                  provider=agent.provider, model=agent.model) != 'native'):
            return None
        text = (event.text or '').strip() or '[The user attached images to correct the current task.]'
        content, skipped = build_native_content_parts(runner._steer_text_with_origin(text, event), event.media_urls)
        images = [part for part in content if part.get('type') == 'image_url']
        if skipped or len(images) != len(event.media_urls):
            return None
        # Event-level PHOTO and MIME labels do not prove that every sibling decodes.
        for part in images:
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            if _transcode_to_png(raw) is None:
                return None
        return content


async def transfer_native_images(authority, ref, row, event):
    """Acknowledge the actual transfer, never the pre-admission text-only demotion."""
    pending_ack = event.__dict__.pop('_native_image_busy_ack', None)
    delivered = await _transfer_native_images(authority, ref, row, event)
    if pending_ack is None or delivered is None:
        return
    if not delivered:
        current = get_session_admission(authority.db, admission_id=row['admission_id'])
        if current is None or current['status'] != 'queued':
            return  # settlement or another generation already decided this input
    key, adapter, state, agent, outcome = pending_ack
    outcome = replace(outcome, effective_mode='steer' if delivered else 'queue', steered=delivered)
    with authority.runner._profile_scope_for_source(event.source):
        await authority.runner._busy_interrupt_and_ack(
            event, key, adapter, state, agent, outcome, steer_parent_only=True,
        )


async def _transfer_native_images(authority, ref, row, event) -> bool | None:
    """Transfer only this committed row to the agent already adopted by its running generation.

    The enclosing admission's tracked write shields preparation and transfer from caller cancellation.
    Registration fences settlement while storage waits off-loop. An uncertain write leaves that fence
    intact; settlement, not a new admission or a caption comparison, decides the original FIFO row.
    Return actual acceptance, refusal, or None when no acknowledgement can be inferred.
    """
    if (row['status'] != 'queued' or not row['payload'].get('native_text_v1', {}).get('media')
            or getattr(event, '_fifo_only', False) or authority.runner._draining):
        return
    sid, ticket = ref.session_id, row['admission_id']
    live = authority.sessions[sid]
    with live.event_stream.lock:
        generation = authority.executing.get(sid)
        agent = _receiving_agent(authority, sid, generation)
        if agent is None or not callable(getattr(agent, 'steer_content', None)):
            return False
    try:
        content = await asyncio.to_thread(_prepare, authority, row, agent)
    except Exception:
        logger.debug('Native image preparation unavailable; admission stays queued', exc_info=True)
        return False
    if not content:
        return False
    with live.event_stream.lock:
        if authority.runner._draining or _receiving_agent(authority, sid, generation) is not agent:
            return False
        _register(authority, sid, generation, ticket)
    try:
        await tracked_write(authority, partial(steer_session_input, authority.db, epoch=authority.epoch,
                                               admission_id=ticket, generation=generation))
    except RuntimeStoreError:
        with live.event_stream.lock:
            _unregister(authority, sid, generation, ticket)
        return False
    except Exception:
        # Commit-success/error is not proof of refusal. Never deliver or replay speculatively.
        logger.warning('Native image transfer write uncertain; claim settlement owns it', exc_info=True)
        return
    with live.event_stream.lock:
        entry = authority.steer_transfers.get(sid)
        delivered = False
        if (entry is not None and entry[0] == generation and ticket in entry[1]
                and _receiving_agent(authority, sid, generation) is agent):
            delivered = entry[1][ticket]
            if not delivered:
                delivered = entry[1][ticket] = bool(agent.steer_content(content, correction_id=ticket))
                if delivered:
                    authority.runner._fold_into_running_turn(
                        agent, row['payload']['native_text_v1']['route'], event)
    if not delivered:
        await tracked_write(authority, partial(revert_steer_input, authority.db, epoch=authority.epoch,
                                               admission_id=ticket, generation=generation))
        with live.event_stream.lock:
            _unregister(authority, sid, generation, ticket)
    authority._publish_pending(ref)
    return delivered
