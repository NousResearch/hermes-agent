"""Authorized native commands share closeout receipts and pin their primary review to the old ref."""
import asyncio
from dataclasses import asdict
from functools import partial
import hashlib
import logging
import uuid

from hermes_cli.conversation_closeout_jobs import PRESERVATION_WAIT_SECONDS

logger = logging.getLogger(__name__)


async def _publish_native_closeout(runner, event, intent, db, authority, ref, receipt):
    from gateway.session_closeout_publication import publish_native_route
    from hermes_cli.conversation_closeout import CloseoutStore, public_receipt
    source = event.source
    epoch = authority.epoch if authority is not None else None
    published = await asyncio.to_thread(publish_native_route, runner.session_store, source,
                                        receipt['closeout_id'], epoch=epoch)
    if published is None:
        return False
    operation = CloseoutStore(db, epoch=epoch).read_current(receipt['closeout_id'])
    if operation is None:
        return False
    receipt = public_receipt(operation)
    # Both native close commands leave a blank independent route for the next message.
    # Its deterministic request key makes interrupted/local retries converge.
    fresh = published[1]
    if fresh is not None:
        title = event.get_command_args().strip() if intent == 'new' else ''
        if title and callable(getattr(runner, '_reset_titled_header', None)):
            await runner._reset_titled_header('', fresh.session_id, title)
    if CloseoutStore(db, epoch=epoch).read_current(receipt['closeout_id']) is None:
        return False
    if authority is not None and receipt['settled']:
        live = authority.sessions.get(ref.session_id)
        if live is not None and not live.subscribers:
            from gateway.session_viewer_close import close_viewer_session
            await close_viewer_session(authority, ref.session_id, 'new_session' if intent == 'new' else 'archive')
    if CloseoutStore(db, epoch=epoch).read_current(receipt['closeout_id']) is None:
        return False
    banner = ('A fresh conversation is ready.' if intent == 'new' else 'The previous conversation is archived.')
    outcomes = receipt['destinations']
    try:
        await runner._send_command_ack(source, banner + ' Memory: ' + outcomes['memory'] +
            '; task ledger: ' + outcomes['tasks'] + '.', 'closeout')
    except Exception:  # health: allow BLE001 -- transport failures can contain credentials; retain the durable receipt without their payloads
        logger.warning('Closeout notice delivery is unconfirmed; durable receipts are retained', exc_info=False)
    return True


async def start_native_closeout(runner, event, intent):
    from gateway.session_authorities import active_authority
    from gateway.session_closeout import canonical_settled
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from hermes_cli.conversation_closeout import CloseoutStore
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    from hermes_constants import profile_name_for_home
    source = event.source
    if not runner._is_user_authorized_for_source(source):
        return 'This conversation could not be closed: authorization is unavailable.'
    denied = runner._check_slash_access(source, 'new' if intent == 'new' else 'archive')
    if denied is not None:
        return denied
    route = runner._session_key_for_source(source)
    db = runner.session_store._db_for_key(route)
    if db is None:
        return 'This conversation could not be preserved locally; its existing route is retained.'
    entry = await runner.async_session_store.get_or_create_session(source)
    old = old_identity(db, entry.session_id, profile=profile_name_for_home(db.db_path.parent) or 'default')
    authority = active_authority(runner)
    ref = SessionRef(authority.profile_id, old['logical_session_id']) if authority is not None else None
    if authority is not None and ref.session_id not in authority.sessions:
        authority._publish_route(source, ref.session_id, route)
    identity = '\n'.join((route, str(source.user_id), str(event.message_id or uuid.uuid4().hex)))
    request_id = 'native-closeout-' + hashlib.sha256(identity.encode()).hexdigest()
    prior = CloseoutStore(db, epoch=authority.epoch if authority is not None else None).for_request(request_id)
    if prior is not None:
        old = prior['old']
        ref = SessionRef(authority.profile_id, old['logical_session_id']) if authority is not None else None

    async def submit(prompt, operation):
        if authority is None:
            return {'error': 'primary_owner_unavailable'}
        actor = SessionActor(str(source.user_id), authority.profile_id,
                             frozenset({'session:submit'}), 'closeout:' + operation)
        # The exact ref is retained even after this chat's route names a fresh conversation.
        receipt = await authority.submit(actor, Submission('closeout-review-' + operation, ref,
                                                           {'text': prompt}, 'queue'))
        return asdict(receipt)

    publish = partial(_publish_native_closeout, runner, event, intent, db, authority, ref)

    def launch(coroutine):
        task = asyncio.create_task(coroutine)
        owner = authority if authority is not None else runner
        jobs = owner.__dict__.setdefault('_closeout_jobs', set())
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    receipt = begin_closeout(db, old, request_id, intent, submit=submit,
        settled=(lambda: canonical_settled(authority, ref)) if authority is not None else (lambda: False),
        launch=launch, publish=publish, epoch=authority.epoch if authority is not None else None,
        wait_seconds=PRESERVATION_WAIT_SECONDS)
    if receipt['status'] == 'preserved' and receipt['archived']:
        if await publish(receipt) is False:
            return 'That closeout was released or superseded; the current conversation is unchanged.'
        return 'The conversation is archived; its preservation receipts are retained.'
    return 'Preserving this conversation before closing it. Online destinations may remain pending.'
