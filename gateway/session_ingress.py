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
# The live session while the executing turn is one its viewer submitted (``is_viewer_input``), else None.
viewer_turn: ContextVar = ContextVar('viewer_turn', default=None)
# Set by a busy-path caller that must be released once its input is durably queued.
busy_acceptance = ContextVar('busy_acceptance', default=None)
# Set beside it by a caller that must also know when the row is durably committed (a busy burst's
# commit task, which command barriers and /restart wait on): ``busy_acceptance`` may already resolve
# once the input's FIFO position is fixed behind an earlier input of its chat still capturing.
busy_commit = ContextVar('busy_commit', default=None)
# ``_validate_native`` clauses about the message's own content, safe to name to the sender's chat.
_CONTENT_CLAUSES = frozenset({
    'text', 'auto_skill', 'channel_prompt', 'channel_context', 'reply_expected',
    'media_urls', 'media_types', 'media_text_inlined', 'media_types_length', 'media_text_inlined_length',
})

# Nothing was committed (main's busy-queue cap); /retry answers it before rewriting (#139257).
QUEUE_FULL_REPLY = ('⏳ Too many messages are already waiting in this conversation; '
                    'send this again after the current reply.')


async def admit_message(authority, event):
    from gateway.session_authority import messaging_admission
    accepted, committed = busy_acceptance.get(), busy_commit.get()
    busy_acceptance.set(None)  # the drain task this schedules must not inherit the busy caller's future
    busy_commit.set(None)
    ingress = messaging_admission.set(True if accepted is None else accepted)
    try:
        receipt = await authority.admit_native(event)
        if committed is not None and not committed.done():
            committed.set_result(None)  # the row is durable (any status): a barrier may go on
    except RuntimeStoreError as exc:
        if exc.reason == 'stale_generation' and exc.field == 'route':
            return None  # a later /stop or /new of the chat fenced it, as main cancels a queued row
        if exc.reason == 'invalid_params' and exc.field:
            # Refused before commit (_validate_native): a refusal notice instead of a generic turn error.
            # The reply reaches the whole chat, so only content clauses are named; trust and provenance
            # clauses (and any new one) stay generic, and gateway.log keeps the clause (``_refuse``).
            named = f'{exc.reason}: {exc.field}' if exc.field in _CONTENT_CLAUSES else exc.reason
            return (f'⚠️ Hermes refused this message ({named}), so it was not saved '
                    'or answered. Ask the operator to check gateway.log.')
        if exc.reason != 'queue_full':
            raise
        # Nothing was committed (main's busy-queue cap): say so instead of a generic turn error.
        return QUEUE_FULL_REPLY
    finally:
        messaging_admission.reset(ingress)
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
    for key in ('local_automation_v1', 'api_turn_v1', 'cli_turn_v1'):
        author = (payload.get(key) or {}).get('turn_author')
        if author is not None:
            return author
    from gateway.session_a2a import forward_author
    return forward_author(policy)


async def execute_admission(authority, ref, row):
    from gateway.session_completion_notes import note_if_unviewed
    noted = await note_if_unviewed(authority, ref, row)  # #134889 LC-7: no viewer, no turn
    if noted is not None:
        return noted
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
    if event.internal and 'gateway_session_id' in event.metadata:
        # Main's ownership proof (802fd5782f3): the session the pre-claim check just validated, so the
        # turn's resolver never re-reads (and on a transient failure drops) an admitted completion.
        event.metadata['gateway_proven_session_id'] = (
            authority.physical_target(ref) if 'native_text_v1' in row['payload'] else event.metadata['gateway_session_id'])
    provenance = row['payload'].get('native_text_v1', {}).get('provenance')
    home = None
    if provenance is not None:
        from gateway.session_ingress_context import restore_provenance
        home = restore_provenance(authority.runner, event.source, provenance)
    from gateway.config import Platform
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.session_results import execution_result
    is_api = live.source.platform == Platform.API_SERVER
    if not (event.internal or is_api or live.source.platform == Platform.LOCAL):
        # The claim on a chat route (native or a viewer's submit): a /new before the turn resolves
        # its session must not move it onto the fresh conversation (main's /new cancelled its task).
        event._claimed_owner = (ref.session_id, row['generation'])
    author = event.metadata.get('turn_author')
    author_token = admission_author.set(author if author is not None else row_turn_author(local_policy, row))
    from gateway.session_operator import is_viewer_input
    viewer_token = viewer_turn.set(live if is_viewer_input(row) else None)
    api_token = api_execution.set(None)
    captured = {}
    result_token = execution_result.set(captured)
    token = executing_admission.set(True)
    # This drain, not the handler's unwind, releases the turn's crash marker (see below).
    event._turn_marker_handoff = True
    handed = False
    try:
        with await _execution_scope(authority, home):
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
            authority.pending_results[row['admission_id']] = {
                'result': result, 'usage': captured.get('usage', {}),
                **({'session_usage': captured['session_usage']} if 'session_usage' in captured else {})}
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
        viewer_turn.reset(viewer_token)


async def _execution_scope(authority, home):
    """Under multiplex, owner-side execution runs under the OWNING profile's home (agent build,
    config, secrets, state.db), never the launch profile's ambient scope; native provenance
    (``home``) refines it. A single-profile gateway keeps its ambient scope byte-for-byte.
    Secret sources (external CLIs) hydrate off the loop, as main's per-message scope does (#99519)."""
    multiplex = getattr(getattr(authority.runner, 'config', None), 'multiplex_profiles', False)
    if home is None and not multiplex:
        return nullcontext()
    from gateway.run import _load_profile_secret_scope, _profile_runtime_scope

    def load():
        from pathlib import Path
        scope_home = home
        if scope_home is None:  # owner_scope's home; symbolic test ids keep the ambient scope
            owner = Path(str(authority.profile_id))
            scope_home = owner if owner.is_absolute() and owner.is_dir() else None
        return scope_home, scope_home and _load_profile_secret_scope(scope_home)

    scope_home, secrets = await asyncio.to_thread(load)
    return nullcontext() if scope_home is None else _profile_runtime_scope(scope_home, secrets)


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
        with await _execution_scope(authority, home):
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


async def dispatch_shared_busy(adapter, event, session_key, *, durable=False):
    """Serial receive loops (IRC, Signal) await this before reading their next frame, so return
    once the follow-up is committed to the FIFO, or queued behind an earlier input of its chat still
    capturing (its position is then fixed; acceptance still waits for the commit); a tracked
    observer delivers its one reply. *durable*: return only once the row is committed (or the input
    was refused, fenced or answered), never on the earlier position release: a busy burst's commit
    task, which a command barrier (``/restart`` closing admission) waits on. Otherwise an input
    released before its commit is held until it commits or its handler ends, and a serial reader's
    next command waits for that too (``burst_gate``): a non-burst busy input (a second video, a voice
    note, text outside queue mode) has no burst commit task to wait on."""
    delivery_event = replace(event, source=replace(event.source))
    accepted = asyncio.get_running_loop().create_future()
    committed = asyncio.get_running_loop().create_future()

    async def observe():
        # Like the first message's lifecycle, this sender adopts the turn's crash marker and
        # releases it only once the reply is ledgered (the marker moves onto ``event``).
        event._turn_marker_handoff = getattr(adapter, 'gateway_runner', None) is not None
        try:
            response = await adapter._message_handler(event)
            # The ledgered send releases the marker off the event it sends.
            _move_marker(event, delivery_event)
            delivery_event._turn_marker_handoff = event._turn_marker_handoff
            await deliver_response(adapter, delivery_event, session_key, response)
        except Exception as exc:
            if not accepted.done():
                raise  # nothing was queued: the receive loop sees the failure as before
            logger.error('[%s] Queued busy follow-up failed: %s', adapter.name, exc, exc_info=True)
            await adapter._notify_turn_error(delivery_event, exc)
        finally:
            # Clears the marker if no ledgered send did, then wakes the drain's bounded wait,
            # which listens on the adopter (``event``).
            await adapter._release_turn_marker(delivery_event)
            await adapter._release_turn_marker(event)

    token, commit_token = busy_acceptance.set(accepted), busy_commit.set(committed)
    try:
        task = asyncio.create_task(observe())
    finally:
        busy_acceptance.reset(token)
        busy_commit.reset(commit_token)
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    try:
        await asyncio.wait({accepted, task}, return_when=asyncio.FIRST_COMPLETED)
        if durable and not task.done():
            await asyncio.wait({committed, task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        if not accepted.done():
            task.cancel()  # cancelled before commit: the input was never ACKed
        raise
    if task.done():
        task.result()  # refused, consumed or answered inline before any durable acceptance
    elif not committed.done():
        _hold_until_committed(adapter, committed, task)


def _hold_until_committed(adapter, committed, task):
    """Released before its commit: the reader reads on, but this input is not durable yet. A serial
    reader's next command waits for its commit or its handler's end (``burst_gate``), so /restart never
    closes admission, and /new never cancels the chat's rows, before it lands."""
    settled = asyncio.get_running_loop().create_future()

    def settle(_):
        if not settled.done():
            settled.set_result(None)
    committed.add_done_callback(settle)
    task.add_done_callback(settle)
    held = adapter.__dict__.setdefault('_uncommitted_inputs', set())
    held.add(settled)
    settled.add_done_callback(held.discard)
