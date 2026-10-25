"""Trusted messaging admission and the existing TurnRunner invocation boundary."""
import asyncio
import logging
from contextvars import ContextVar
from contextlib import nullcontext
from dataclasses import replace

from gateway.platforms.event import MessageEvent
from gateway.session_envelope import restore_native
from gateway.turn_delivery import (
    adopt_turn_delivery,
    current_turn_delivery,
    fire_turn_delivery,
    turn_delivery_for_event,
)
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


def _refusal_reply(exc: 'RuntimeStoreError') -> tuple[bool, str | None]:
    """Map an admission refusal to ``(handled, reply)``.

    ``handled=False`` means the caller must re-raise: only the refusals main answers with a notice
    are listed here, and the trust/provenance clauses stay generic on purpose.
    """
    if exc.reason == 'stale_generation' and exc.field == 'route':
        return True, None  # a later /stop or /new of the chat fenced it, as main cancels a queued row
    if exc.reason == 'invalid_params' and exc.field:
        # Refused before commit (_validate_native): a refusal notice instead of a generic turn error.
        # The reply reaches the whole chat, so only content clauses are named; trust and provenance
        # clauses (and any new one) stay generic, and gateway.log keeps the clause (``_refuse``).
        named = f'{exc.reason}: {exc.field}' if exc.field in _CONTENT_CLAUSES else exc.reason
        return True, (f'⚠️ Hermes refused this message ({named}), so it was not saved '
                      'or answered. Ask the operator to check gateway.log.')
    if exc.reason == 'queue_full':
        # Nothing was committed (main's busy-queue cap): say so instead of a generic turn error.
        return True, QUEUE_FULL_REPLY
    return False, None


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
        handled, reply = _refusal_reply(exc)
        if not handled:
            raise
        return reply
    finally:
        messaging_admission.reset(ingress)
    if receipt.status == 'terminal':
        return None
    # Only the delivery waiter is process-local; execution reads the committed snapshot.
    authority.native_waiters.add(receipt.admission_id)
    # Publish the turn's record under its ADMISSION ID before the drain can claim the row (the
    # drain may already have run: ``bind`` adopts whatever record the registry settled on). The
    # ContextVar stays set for the REST OF THIS TASK, so this sender reads the exact record back
    # after ``_message_handler`` returns — an event rewrite cannot redirect it to another object.
    pre_existing = _registry(authority, 'turn_deliveries').get(receipt.admission_id)
    turn_delivery = bind_turn_delivery(
        authority, receipt.admission_id, event,
        sender_record=current_turn_delivery.get(), adapter_sends=True)
    current_turn_delivery.set(turn_delivery)
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
        if pre_existing is None:
            # The row settled before a turn registered a waiter: the record we just created
            # belongs to a turn whose reply nobody will send. Drop it rather than strand it.
            pop_turn_delivery(authority, receipt.admission_id)
        return None if status == 'terminal' else pause_notice(authority, receipt.ref, 'unknown_execution')
    if accepted is not None and not accepted.done():
        accepted.set_result(None)
    try:
        result = await asyncio.shield(waiter)
    except RuntimeStoreError as exc:
        if pre_existing is None and exc.reason != 'unknown_execution':
            # No turn will send a reply on this lane and none is retained for a Discard.
            pop_turn_delivery(authority, receipt.admission_id)
        return pause_notice(authority, receipt.ref, exc.reason)
    except BaseException:
        # The sender task is being cancelled (stop, shutdown) or the wait was aborted: no adapter
        # lifecycle is left to send this reply or fire its record.  Hand both back to the drain --
        # clear the carried bit, and move a reply the drain already captured for THIS adapter into
        # ``pending_deliveries`` so settlement sends it instead of losing it with the cancelled
        # task (the shield kept the waiter alive; nobody else would resolve or deliver it).
        authority.native_waiters.discard(receipt.admission_id)
        if authority.waiters.get(receipt.admission_id) is waiter:
            del authority.waiters[receipt.admission_id]
        record = turn_deliveries(authority).get(receipt.admission_id)
        if record is not None:
            record.adapter_sends = False
        captured_reply = waiter_replies(authority).pop(receipt.admission_id, None)
        if captured_reply is not None:
            authority.pending_deliveries[receipt.admission_id] = captured_reply
        raise
    finally:
        if adopters.get(receipt.admission_id) is event:
            del adopters[receipt.admission_id]
    # The drain may have created the record after our publish: adopt whichever one the registry
    # settled on, hand it to this sender task, and stop tracking it by id (the sender owns firing).
    turn_delivery = bind_turn_delivery(
        authority, receipt.admission_id, event,
        sender_record=current_turn_delivery.get(), adapter_sends=True)
    current_turn_delivery.set(turn_delivery)
    if _registry(authority, 'turn_deliveries').get(receipt.admission_id) is turn_delivery:
        _registry(authority, 'turn_deliveries').pop(receipt.admission_id, None)
    return result


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


def turn_deliveries(authority):
    """admission_id -> the turn's deferred-delivery record (process-local, like the waiters)."""
    return _registry(authority, 'turn_deliveries')


def bind_turn_delivery(authority, admission_id, event=None, *, sender_record=None,
                       adapter_sends=None):
    """The ONE record for this turn, keyed by its admission id.

    Either lane can arrive first — ``admit_message`` (the adapter / busy-observer task that owns
    the delivery waiter, which passes the record it created) or ``execute_admission`` (the drain,
    which passes none and adopts the registry's record or creates its own from the restored
    event).  ``setdefault`` makes both share ONE object, so a ``pre_gateway_dispatch`` rewrite
    between them can never leave the SENDER holding a different record from the one the turn
    registers on: the admission id is the shared identity, not the MessageEvent (which
    ``dataclasses.replace`` rebuilds).  The record is also attached to ``event`` for any lane that
    resolves it from an event, but no lane depends on that.

    ``adapter_sends`` is the CARRIED "a native adapter lifecycle owns this reply" bit: the lane
    that knows (either one) stamps it, and ``deliver_settled`` reads it instead of inferring
    ownership from the multi-tenant ``authority.waiters``."""
    registry = _registry(authority, 'turn_deliveries')
    record = registry.get(admission_id)
    if record is None:
        record = sender_record
        if record is None and event is not None:
            record = turn_delivery_for_event(event)
        if record is None:
            return None
        registry[admission_id] = record
    if adapter_sends:
        record.adapter_sends = True
    if event is not None:
        try:
            event._turn_delivery = record
        except Exception:
            logger.debug('Event refused the turn delivery record', exc_info=True)
    return record


def pop_turn_delivery(authority, admission_id):
    return _registry(authority, 'turn_deliveries').pop(admission_id, None)


def close_turn_delivery(authority, admission_id):
    """CLOSED-drop a turn's record because nothing will ever deliver its reply.

    Release/cleanup callbacks (``run_on_drop``) run now -- they free a queue slot or delete
    per-turn UI that no delivery will release -- and reply-ordered callbacks are discarded.
    Idempotent, and a later ``fire()`` is a no-op because the callbacks were consumed, so closing
    is safe while a sender task is still unwinding."""
    record = pop_turn_delivery(authority, admission_id)
    if record is not None:
        record.drop()
    return record


def close_unretained_turn_delivery(authority, admission_id):
    """Close the record unless a Discard may still deliver this turn's reply.

    ``retain_unsettled_delivery`` keeps the reply -- and therefore its record -- for
    ``deliver_resolved``, which closes it when the Discard owes nothing.  Every other exit that
    abandons an admission (a cancelled drain, a lost managed worker, settlement recovery that
    lost ownership) leaves no lane to fire the record, so it must not stay in the registry."""
    if admission_id in _registry(authority, 'unsettled_deliveries'):
        return
    close_turn_delivery(authority, admission_id)


async def fire_settled_turn_delivery(authority, admission_id):
    """Fire the record of a reply just handed to its live delivery waiter.

    ``deliver_settled`` defers to this when a waiter -- not a native adapter lifecycle -- owns the
    reply, so the deferred callbacks follow the handover instead of preceding it (a goal notice
    must not reach the chat before the reply it annotates).  A native waiter is skipped: its
    adapter task fires the record itself, after it sends."""
    record = turn_deliveries(authority).get(admission_id)
    if record is None or getattr(record, 'adapter_sends', False):
        return
    pop_turn_delivery(authority, admission_id)
    await record.fire()


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
        # The Discard released the adapter lifecycle that would have sent this reply, so the
        # drain is the sender now -- and therefore the lane that fires the record. Clearing the
        # carried bit keeps ``deliver_settled`` from handing it back to a sender that is gone.
        record = turn_deliveries(authority).get(admission_id)
        if record is not None:
            record.adapter_sends = False
        authority.pending_deliveries[admission_id] = delivery
    else:
        # A Discard owes no answer, so it owes no status line either -- but it does owe the turn's
        # release/cleanup callbacks (``run_on_drop``): no delivery will ever run them, and they
        # free a queue slot or delete per-turn UI.  ``drop()`` runs exactly those and discards
        # everything reply-ordered, so the record must be dropped, not merely discarded.
        record = pop_turn_delivery(authority, admission_id)
        if record is not None:
            record.drop()
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
        # GW-10: the way out is in this chat. /discard acknowledges the lost turn (never replayed)
        # through the same fenced resolve_unknown as Desktop/TUI, and the saved messages then run.
        # Slack registers no native /discard (its slash cap): the umbrella `/hermes discard` reaches it.
        slack = getattr(getattr(live.source, 'platform', None), 'value', None) == 'slack'
        discard = '/hermes discard' if slack else '/discard'
        return ('⏸️ This conversation is paused: a previous turn did not finish when Hermes restarted, '
                f'and only you know whether to send it again. Your message is saved. Send {discard} to drop '
                'the lost turn and continue (it is not re-run), or /reset to start a fresh conversation.')
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
    from gateway.session_managed_worker import managed_policy, admit_managed
    policy = managed_policy(authority, ref)
    if policy is not None:
        return await admit_managed(authority, ref, row, policy)
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
    # Adopt the record the sender lane published under this admission id (or create it here for a
    # recovered no-waiter turn). The id is the identity; ``event`` is only a carrier for lanes
    # that resolve it from one, and it may be a different object from the sender's event.
    # ``native`` (claimed above) is the CARRIED ownership bit: only a native adapter lifecycle
    # sends this turn's reply and fires the record itself. The API, Webhook, Bot, Hosted and
    # Managed lanes do not set it, so their turns are fired by whichever lane ends up delivering.
    turn_delivery = bind_turn_delivery(
        authority, row['admission_id'], event, adapter_sends=native)
    delivery_token = current_turn_delivery.set(turn_delivery)
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
        current_turn_delivery.reset(delivery_token)
        executing_admission.reset(token)
        execution_result.reset(result_token)
        api_execution.reset(api_token)
        admission_author.reset(author_token)
        viewer_turn.reset(viewer_token)


def _launch_body_if_launch_owner(authority):
    """A standalone owner's turn binds nothing, but the launch profile's is marked at entry: a second
    profile switching this gateway to multiplexing mid-turn must not fail its later credential reads
    closed. Any other owner (a hosted named profile) stays unmarked, never reading the launch env."""
    from pathlib import Path
    from agent.secret_scope import _is_process_home
    owner = Path(str(authority.profile_id))
    if not owner.is_absolute() or not _is_process_home(owner):
        return nullcontext()
    from tui_gateway.launch_profile_policy import launch_body
    return launch_body()


async def _execution_scope(authority, home):
    """Under multiplex, owner-side execution runs under the OWNING profile's home (agent build,
    config, secrets, state.db), never the launch profile's ambient scope; native provenance
    (``home``) refines it. A single-profile gateway keeps its ambient scope byte-for-byte.
    Secret sources (external CLIs) hydrate off the loop, as main's per-message scope does (#99519)."""
    multiplex = getattr(getattr(authority.runner, 'config', None), 'multiplex_profiles', False)
    if home is None and not multiplex:
        return _launch_body_if_launch_owner(authority)
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
    failure cannot rewrite that outcome; it is logged, and inference never runs again.

    The turn's deferred callbacks are fired by whichever lane actually delivers the reply: this
    one after its send, the native adapter lifecycle that owns the reply
    (``record.adapter_sends``), or -- for a reply handed to a live waiter (API, Webhook, Bot,
    Hosted) -- ``fire_settled_turn_delivery`` right after that handover.  No lane infers that
    from ``authority.waiters``, which those lanes also fill -- an is_api turn has a waiter there
    but no adapter send, and judging by it left the record neither sent, fired, nor dropped."""
    held = _held_markers(authority).pop(admission_id, None)
    record = turn_deliveries(authority).get(admission_id)
    # CARRIED ownership bit, plus the late-registration fallback: only ``admit_message`` fills
    # ``native_waiters`` (the other lanes only touch ``waiters``), so a native waiter that
    # registered while the turn ran is still recognised as the sender.
    adapter_sends = (bool(getattr(record, 'adapter_sends', False))
                     or admission_id in authority.native_waiters)
    if adapter_sends:
        # A native adapter lifecycle owns this reply: its task sends it and fires the SAME record
        # object afterwards, so this lane neither sends nor fires (and must not strand the record
        # by dropping it -- the sender task reads it back from its context).
        authority.native_waiters.discard(admission_id)
        authority.pending_deliveries.pop(admission_id, None)
        adopter = _marker_adopters(authority).get(admission_id)
        if held is not None and adopter is not None:
            _move_marker(held, adopter)
            held = None
        if held is not None:
            await _release_turn_marker(authority, held)
        return
    delivery = authority.pending_deliveries.get(admission_id)
    if delivery is None:
        if held is not None:
            # Fenced unknown: no answer is owed YET, so the record stays for the later resolved
            # send (``deliver_resolved``), which fires it.
            await _release_turn_marker(authority, held)
            return
        if admission_id in authority.waiters:
            # A live waiter (API, Webhook, Bot, Hosted) is about to be handed this reply by
            # ``_resolve_settled_waiter``: fire the record AFTER that handover, never before it.
            return
        # Nothing on this lane sends (no reply, API/local, or a failed settlement): the turn's
        # deferred callbacks must not be stranded in the registry -- fire them here.
        record = pop_turn_delivery(authority, admission_id)
        if record is not None:
            await record.fire()
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
        # Fire BY ID: this lane has no adapter lifecycle, and the record -- not the event object,
        # which a rewrite may have replaced -- is the turn's identity.
        record = pop_turn_delivery(authority, admission_id)
        try:
            # The ledgered send already released the marker; an empty or failed send owes no more.
            # Release it BEFORE firing: a slow callback must not delay the durable release (nor
            # skip it by cancellation after the fire).
            await _release_turn_marker(authority, event)
        finally:
            # Even a failed/empty send owns its turn's deferred callbacks: never strand the notice.
            if record is not None:
                await record.fire()
            else:
                await fire_turn_delivery(event)


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
        turn_delivery = turn_delivery_for_event(event)
        delivery_token = current_turn_delivery.set(turn_delivery)
        try:
            response = await adapter._message_handler(event)
            # ``admit_message`` may have adopted the drain's record for this admission: fire what
            # the handler actually registered on, never the event's opening record.
            turn_delivery = current_turn_delivery.get() or turn_delivery
            # The ledgered send releases the marker off the event it sends.
            _move_marker(event, delivery_event)
            delivery_event._turn_marker_handoff = event._turn_marker_handoff
            # ``delivery_event`` is a ``dataclasses.replace`` copy, which does not carry the
            # record; adopt it before the send so the busy lane's send owns the turn's callbacks.
            adopt_turn_delivery(event, delivery_event)
            await deliver_response(adapter, delivery_event, session_key, response)
        except Exception as exc:
            if not accepted.done():
                raise  # nothing was queued: the receive loop sees the failure as before
            logger.error('[%s] Queued busy follow-up failed: %s', adapter.name, exc, exc_info=True)
            await adapter._notify_turn_error(delivery_event, exc)
        finally:
            current_turn_delivery.reset(delivery_token)
            try:
                # Clears the marker if no ledgered send did, then wakes the drain's bounded wait,
                # which listens on the adopter (``event``).  Released BEFORE the callbacks fire:
                # a slow callback must not delay it (nor skip it by cancellation after the fire).
                await adapter._release_turn_marker(delivery_event)
                await adapter._release_turn_marker(event)
            finally:
                # A failed/short reply still owns its deferred callbacks; never strand them.
                # Fired by OBJECT: the admission-keyed record, not the (possibly rewritten) event.
                await turn_delivery.fire()

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
