"""Trusted messaging admission and the existing TurnRunner invocation boundary."""
import asyncio
import logging
from contextvars import ContextVar
from contextlib import nullcontext
from dataclasses import replace

from gateway.platforms.event import MessageEvent
from gateway.session_envelope import restore_native
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

admission_author = ContextVar('admission_author', default=None)
executing_admission = ContextVar('executing_admission', default=False)
# Set by a busy-path caller that must be released once its input is durably queued.
busy_acceptance = ContextVar('busy_acceptance', default=None)


async def admit_message(authority, event):
    accepted = busy_acceptance.get()
    busy_acceptance.set(None)  # the drain task this schedules must not inherit the busy caller's future
    receipt = await authority.admit_native(event)
    if receipt.status == 'terminal':
        return None
    # Only the delivery waiter is process-local; execution reads the committed snapshot.
    authority.native_waiters.add(receipt.admission_id)
    waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
    adopters = _marker_adopters(authority)
    if getattr(event, '_turn_marker_handoff', False):
        # The adapter lifecycle that sends this reply releases the turn's crash marker only once
        # the reply is in the delivery ledger; the drain hands its marker to this event.
        adopters[receipt.admission_id] = event
    # A drain already running can claim, run and settle the row before this caller resumes; one
    # that settled it already sent (or owes) the reply itself and resolves no waiter registered now.
    from hermes_state_runtime import get_session_admission
    row = get_session_admission(authority.db, admission_id=receipt.admission_id)
    status = 'terminal' if row is None else row['status']  # None: deleted with its chat, nothing owed
    if status in ('terminal', 'unknown'):
        authority.native_waiters.discard(receipt.admission_id)
        if authority.waiters.get(receipt.admission_id) is waiter:
            del authority.waiters[receipt.admission_id]
        if adopters.get(receipt.admission_id) is event:
            del adopters[receipt.admission_id]
        return None if status == 'terminal' else pause_notice(authority, receipt.ref, 'unknown_execution')
    if accepted is not None and not accepted.done():
        accepted.set_result(None)
    try:
        return await asyncio.shield(waiter)
    except RuntimeStoreError as exc:
        return pause_notice(authority, receipt.ref, exc.reason)
    finally:
        if adopters.get(receipt.admission_id) is event:
            del adopters[receipt.admission_id]


def _marker_adopters(authority):
    adopters = getattr(authority, 'marker_adopters', None)
    if adopters is None:
        adopters = authority.marker_adopters = {}
    return adopters


_MARKER_FIELDS = ('_gateway_active_turn_session_key', '_gateway_active_turn_token')


async def _hand_over_turn_marker(authority, admission_id, event):
    """The executed turn's durable active-turn marker outlives the handler, as the in-process
    turn's does on main, and is released only once its reply is in the delivery ledger (or nothing
    is owed). A kill between the terminal commit and that ledger write then leaves a marked turn
    whose persisted reply the next unclean boot ledgers and sends once, without new inference,
    instead of a settled answer nobody owes. The waiting adapter lifecycle adopts it; a recovered
    no-waiter reply keeps it until ``deliver_settled``; any other turn releases it here."""
    adopter = _marker_adopters(authority).get(admission_id)
    if adopter is not None:
        _move_marker(event, adopter)
        return
    if admission_id in authority.pending_deliveries:
        # Released by ``deliver_settled``, also when settlement failed and dropped the delivery.
        _held_markers(authority)[admission_id] = event
        return
    await _release_turn_marker(authority, event)


def _move_marker(source, target):
    for name in _MARKER_FIELDS:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
            delattr(source, name)


def _held_markers(authority):
    held = getattr(authority, 'held_turn_markers', None)
    if held is None:
        held = authority.held_turn_markers = {}
    return held


def _registry(authority, name):
    found = getattr(authority, name, None)
    if found is None:
        found = {}
        setattr(authority, name, found)
    return found


def waiter_replies(authority):
    """admission -> the reply a live native waiter is about to send, kept only until the drain
    settles the turn, so a turn fenced ``unknown`` can still owe it to the chat."""
    return _registry(authority, 'waiter_replies')


def retain_unsettled_delivery(authority, admission_id):
    """Settlement could not commit and the turn is fenced ``unknown``, but this owner keeps its
    captured answer for Discard (``resolve_unknown``). The reply stays owed to its destination,
    and so does the turn's crash marker: the live waiter is released with a pause notice, so the
    marker is taken back from it. Restart while unknown clears the marker (no answer is owed,
    ``release_unknown_turn_markers``); a Discard that commits the captured answer makes it owed,
    sent once by ``deliver_resolved`` or, after a kill before that send, by the boot ledger."""
    delivery = waiter_replies(authority).pop(admission_id, None)
    if delivery is not None:
        adopter = _marker_adopters(authority).get(admission_id)
        if adopter is not None:
            _move_marker(adopter, delivery[1])
            _held_markers(authority)[admission_id] = delivery[1]
    else:
        delivery = authority.pending_deliveries.pop(admission_id, None)
    if delivery is not None:
        _registry(authority, 'unsettled_deliveries')[admission_id] = delivery


async def deliver_resolved(authority, admission_id, *, owed):
    """After a Discard committed: send the retained reply when its captured answer committed with
    the resolution (``owed``), else release its marker and drop it. Popped first, so a repeated
    resolution never sends twice; a turn the drain still owns is delivered by the drain."""
    delivery = _registry(authority, 'unsettled_deliveries').pop(admission_id, None)
    if delivery is None:
        return
    if owed:
        authority.pending_deliveries[admission_id] = delivery
    await deliver_settled(authority, admission_id)


async def _release_turn_marker(authority, event):
    if getattr(event, '_gateway_active_turn_token', None):
        await authority.runner._clear_durable_active_turn(event)


def pause_notice(authority, ref, reason):
    """The committed input stays queued behind a paused FIFO (a turn lost across an owner
    restart, or a head the preflight refused). Tell the platform user once per pause episode
    through the ordinary reply path; later messages onto the same pause are admitted silently."""
    live = authority.sessions[ref.session_id]
    if live.pause_notified:
        return None
    live.pause_notified = True
    if reason == 'runtime_draining':
        # Transient: the row runs when the restarted owner drains it, so /reset would be wrong advice.
        return '⏳ Hermes is restarting — your message is saved and will be answered when it is back.'
    if reason == 'session_busy':
        # Transient: queued behind work already running on this conversation (a worker execution).
        return '⏳ Your message is saved and will be answered when the work running in this conversation finishes.'
    if reason == 'unknown_execution':
        cause = 'a previous turn did not finish when Hermes restarted, so nothing queued after it will run'
    else:
        cause = f'a queued message could not be re-authorized ({reason})'
    return (f'⏸️ This conversation is paused: {cause}. Your message is saved but will not be answered here. '
            'Send /reset to start a fresh conversation, or ask the operator to resume this one.')


def row_turn_author(policy, row):
    """Who wrote the admitted input, for memory attribution only: the producer's stamp when it
    left one, else the forwarded peer bound into the session policy; never grants anything."""
    payload = row['payload']
    for key in ('local_automation_v1', 'api_turn_v1'):
        author = (payload.get(key) or {}).get('turn_author')
        if author is not None:
            return author
    from gateway.session_a2a import forward_author
    return forward_author(policy)


async def execute_admission(authority, ref, row):
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is not None and policy.source == 'cron':
        from gateway.session_cron import execute
        return await execute(authority, ref, row, policy)
    local_policy = policy
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
    home = None
    if provenance is not None:
        from gateway.session_ingress_context import restore_provenance
        home = restore_provenance(authority.runner, event.source, provenance)
    scope = _execution_scope(authority, home)
    from gateway.config import Platform
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.session_results import execution_result
    is_api = live.source.platform == Platform.API_SERVER
    author = event.metadata.get('turn_author')
    author_token = admission_author.set(author if author is not None else row_turn_author(local_policy, row))
    api_token = api_execution.set(None)
    captured = {}
    result_token = execution_result.set(captured)
    token = executing_admission.set(True)
    # This drain, not the handler's unwind, releases the turn's crash marker (see below).
    event._turn_marker_handoff = True
    handed = False
    try:
        with scope:
            if is_api:
                # Committed media resolves under the owning profile's home, like admission did.
                prepared = prepare_api_execution(authority, ref, row['payload'])
                api_execution.set(prepared)
                if isinstance(event.text, list):
                    # The durable transcript keeps the committed media references (the same
                    # ``[Image attached ...]`` hints native transports persist), not only the caption.
                    event.text = '\n'.join(part['text'] for part in prepared['content'] if part.get('type') == 'text')
                event.allow_gateway_control = False
                event.internal = True  # trust comes from the private binding and preclaim, never client JSON
            response = await authority.runner._handle_message(event)
            result = captured.get('terminal_result', captured.get('result'))
            if result is None:
                # No TurnRunner result means the handler answered without executing the turn.
                # A turn that never ran because it failed (agent initialization raised, history
                # unreadable) recorded its own failure via ``record_unexecuted_failure`` on every
                # surface; any other reply here is a deliberate notice. An API caller asked for
                # work, so its receipt is a failure either way, never a completed apology; so did
                # a finite (``chat -q`` / ``-z``) prompt, whose exit code is the only verdict a
                # script reads, unless the prompt was a command and the reply is its answer.
                result = {'final_response': response or '', 'messages': []}
                if is_api or (row['payload'].get('finite') is True and not event.get_command()):
                    result = {'final_response': '', 'messages': [], 'failed': True, 'completed': False,
                              'error': response or 'The admitted turn failed.'}
            # The drain commits this under the stream lock so no viewer reads `terminal`
            # before the completion event exists in the replay ring.
            authority.pending_results[row['admission_id']] = {'result': result, 'usage': captured.get('usage', {})}
            if not is_api and response:
                adapter = authority.runner._adapter_for_source(event.source)
                if adapter is not None:
                    # A recovered turn with no live delivery waiter: the drain sends this only after
                    # it commits the terminal outcome (deliver_settled), so a failed settlement never
                    # leaves an externally answered admission started -> unknown. A live waiter
                    # sends its own; the record only outlives a settlement fenced unknown.
                    delivery = (adapter, event, live.route, response, home)
                    if native:
                        waiter_replies(authority)[row['admission_id']] = delivery
                    else:
                        authority.pending_deliveries[row['admission_id']] = delivery
            handed = True
            await _hand_over_turn_marker(authority, row['admission_id'], event)
            return response
    finally:
        if not handed:
            await _release_turn_marker(authority, event)
        executing_admission.reset(token)
        execution_result.reset(result_token)
        api_execution.reset(api_token)
        admission_author.reset(author_token)


def _execution_scope(authority, home):
    """Under multiplex, owner-side execution runs under the OWNING profile's home (agent build,
    config, secrets, state.db), never the launch profile's ambient scope; native provenance
    (``home``) refines it. A single-profile gateway keeps its ambient scope byte-for-byte."""
    if home is not None:
        from gateway.run import _profile_runtime_scope
        return _profile_runtime_scope(home)
    if getattr(getattr(authority.runner, 'config', None), 'multiplex_profiles', False):
        from gateway.session_authorities import owner_scope
        return owner_scope(authority, hydrate_secrets=True)
    return nullcontext()


async def deliver_settled(authority, admission_id):
    """Send a recovered no-waiter turn's reply after its terminal outcome committed. A send
    failure cannot rewrite that outcome; it is logged, and inference never runs again."""
    held = _held_markers(authority).pop(admission_id, None)
    delivery = authority.pending_deliveries.get(admission_id)
    if delivery is not None and admission_id in authority.native_waiters:
        # ``admit_message`` registered its delivery waiter after this turn started: the drain
        # hands that waiter the reply and its adapter lifecycle sends it, so this send is not owed.
        authority.native_waiters.discard(admission_id)
        authority.pending_deliveries.pop(admission_id)
        adopter = _marker_adopters(authority).get(admission_id)
        if held is not None and adopter is not None:
            _move_marker(held, adopter)
            held = None
        delivery = None
    if delivery is None:
        if held is not None:
            # Fenced unknown: no answer is owed, so no later boot may deliver one either.
            await _release_turn_marker(authority, held)
        return
    adapter, event, session_key, response, home = delivery
    try:
        with _execution_scope(authority, home):
            await deliver_response(adapter, event, session_key, response)
    except Exception:
        logger.exception('Delivery of settled admission %s failed', admission_id)
    finally:
        # Held until the send returns: an empty map means every settled reply was handed over.
        authority.pending_deliveries.pop(admission_id, None)
        # The ledgered send already released the marker; an empty or failed send owes no more.
        await _release_turn_marker(authority, event)


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
    """Serial receive loops (IRC, Signal) await this before reading their next frame, so return
    once the follow-up is committed to the FIFO; a tracked observer delivers its one reply."""
    delivery_event = replace(event, source=replace(event.source))
    accepted = asyncio.get_running_loop().create_future()

    async def observe():
        try:
            response = await adapter._message_handler(event)
            await deliver_response(adapter, delivery_event, session_key, response)
        except Exception as exc:
            if not accepted.done():
                raise  # nothing was queued: the receive loop sees the failure as before
            logger.error('[%s] Queued busy follow-up failed: %s', adapter.name, exc, exc_info=True)
            await adapter._notify_turn_error(delivery_event, exc)

    token = busy_acceptance.set(accepted)
    try:
        task = asyncio.create_task(observe())
    finally:
        busy_acceptance.reset(token)
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    try:
        await asyncio.wait({accepted, task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        if not accepted.done():
            task.cancel()  # cancelled before commit: the input was never ACKed
        raise
    if task.done():
        task.result()  # refused, consumed or answered inline before any durable acceptance
