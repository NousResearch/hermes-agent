"""Bot Chat ingress into the existing authority FIFO; viewers never own delivery.

The mailbox is a delivery receipt, not an execution queue. A committed canonical
admission is the only consumer; unknown execution is never retried as inference.
"""
import asyncio
from functools import partial
import logging
from pathlib import Path
import time

from agent.turn_author import parse_turn_author

from gateway.session_contract import Principal, SessionRef
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tools.bot_failure_reasons import HELD_NOT_SENT
from tools.bot_live_delivery import _TERMINAL, _delivery_id, _read
logger = logging.getLogger(__name__)
from gateway.session_bot_mailbox import (mailbox_lock, read_receipt, write_receipt,
    scan_receipts, remember_receipt, track_receipt_task, withdraw_pin, refuse_pin, requeue_pin)


_CANONICAL_RECEIPT_STATUSES = frozenset({
    'canonical', 'queued', 'claimed', 'ambiguous', 'settled', 'failed', 'cancelled',
})
_BOT_DM_MAX_CHARS = 16200
# Admission refusals about the owner's state, not the delivery: with nothing committed the pin is
# withdrawn and a same-id resend admits once it clears (a full FIFO, a drain for restart/update or
# retirement, a fenced epoch or generation, storage, a live CLI holding the session, or a target
# that moved before the write: the resend resolves it again). Every other refusal (permission_denied,
# invalid_params, admission_conflict, profile_mismatch, ...) is final for that delivery: ``refuse_pin``.
_OWNER_STATE_REFUSALS = frozenset({
    'queue_full', 'runtime_draining', 'runtime_retired', 'stale_epoch', 'stale_generation',
    'storage_unavailable', 'runtime_coordination_required', 'session_busy', 'not_found',
})
_PRODUCER_MAX_CHARS = 1024 * 1024


def _canonical_record_error(path, record, required_strings=('admission_id', 'profile_home', 'session_id', 'principal_id')):
    """Return why an admission-backed receipt is unsafe for bulk recovery."""
    delivery_id = record.get('delivery_id')
    if not isinstance(delivery_id, str) or delivery_id != path.stem:
        return 'delivery id does not match filename'
    try:
        _delivery_id(delivery_id)
    except ValueError:
        return 'delivery id is invalid'
    missing = [key for key in required_strings
               if not isinstance(record.get(key), str) or not record[key]]
    if missing:
        return 'missing canonical fields: ' + ', '.join(missing)
    if not isinstance(record.get('message'), str):
        return 'message is not a string'
    status = record.get('status')
    if not isinstance(status, str) or status not in _CANONICAL_RECEIPT_STATUSES:
        return f'unknown canonical status {status!r}'
    return None


def _record_shape_error(path, record):
    """Validate the known canonical, refused or legacy receipt shape for directory scans."""
    if record.get('refused') is True and 'admission_id' not in record:
        # ``refuse_pin``: the canonical pin fields, terminal ``failed``, never admitted.
        if record.get('status') != 'failed':
            return 'refused receipt is not failed'
        return _canonical_record_error(path, record, ('profile_home', 'session_id', 'principal_id'))
    if 'admission_id' in record or record.get('status') == 'canonical' or 'principal_id' in record:
        return _canonical_record_error(path, record)
    if 'owner' in record:
        from tools.bot_live_delivery import _ticket_shape_error
        return _ticket_shape_error(path, record)
    return 'unknown receipt schema'


def _is_pre_admission_pin(path, record):
    """The ``ambiguous`` pin ``_admit`` writes before it admits: no admission id, never refused."""
    return (record.get('status') == 'ambiguous' and 'admission_id' not in record and 'refused' not in record
            and _canonical_record_error(path, record, ('profile_home', 'session_id', 'principal_id')) is None)


def _warn_malformed(path, problem):
    logging.getLogger(__name__).warning("Skipping malformed Bot mailbox receipt %s (%s)", path, problem)


def _scan_records(root, pins=None):
    """Bulk mailbox scan: isolate unreadable/malformed records; exact-id reads still fail closed.
    An owner scan passes ``pins``: pre-admission pins land there with their mtime, unwarned, since
    only the owner can tell whether their admission committed (``_scan_owner_receipts``)."""
    for path in root.glob('*.json'):
        try:
            record = _read(path)
            problem = None if record is None else _record_shape_error(path, record)
            if problem is not None and pins is not None and _is_pre_admission_pin(path, record):
                pins.append((path, record, path.stat().st_mtime))
                continue
        except (OSError, ValueError) as exc:
            record, problem = None, str(exc)
        if problem is not None:
            _warn_malformed(path, problem)
            continue
        if record is not None:
            yield path, record


# An ``_admit`` pins, commits and withdraws under the mailbox lock in well under a second: a
# pre-admission pin older than this is no in-flight admission, and the owner scan may withdraw it.
_PIN_GRACE_SECONDS = 300


async def _scan_owner_receipts(authority, home):
    """The owner's bulk scan, under the mailbox lock ``_admit`` holds. A pre-admission pin past the
    grace window whose ``bot:<key>`` never committed (refused by an older build, a death before the
    commit, a transient refusal on the legacy migration path) is withdrawn as ``_admit`` withdraws
    one: the scan stops warning and a same-id resend admits once. A pin laid over a legacy ticket
    (``owner``) is that never-run message's only copy: it is requeued instead, for ``_migrate`` to
    admit once under the same id. A pin whose admission committed stays the explicit unknown record
    and keeps its warning.

    Also returns the FIFO barrier: the lowest ``_legacy_order`` of a legacy pin of this home that is
    still unresolved (inside the grace window, or its requeue lost to a concurrent change). No
    legacy ticket ordered after it may be admitted by this pass. It holds for one scan only: past
    the grace window the pin is requeued and released, and a failed requeue is re-read by the next
    scan (gone, admitted or a ticket again), so the barrier never outlives the pin."""
    pins, barrier = [], None
    root, records = await scan_receipts(home, pins)
    for path, record, mtime in pins:
        key, age, legacy = record['delivery_id'], time.time() - mtime, 'owner' in record
        if (record['profile_home'] != str(home)
                or _pin_committed(authority, record, 'bot:' + key)):
            _warn_malformed(path, _record_shape_error(path, record))
            continue
        cleared = False
        if age >= _PIN_GRACE_SECONDS and legacy:
            ticket = await requeue_pin(home, key)
            cleared = ticket is not None and _record_shape_error(path, ticket) is None
            if cleared:
                records.append((path, ticket))
        elif age >= _PIN_GRACE_SECONDS:
            cleared = await withdraw_pin(home, key)
        if not cleared:
            if legacy:
                order = _legacy_order(path, record)
                barrier = order if barrier is None else min(barrier, order)
            _warn_malformed(path, _record_shape_error(path, record))
            continue
        logging.getLogger(__name__).info('%s Bot delivery %s pre-admission pin (age %ds): nothing was admitted',
                                         'Requeued' if legacy else 'Withdrew', key, age)
    return root, records, barrier


def _pin_committed(authority, record, request_id):
    """Whether the pin's admission may have committed: a live row, or a deleted target (its
    admissions moved to retired receipts keyed by the route principal the pin never recorded)."""
    if _admission_at(authority, record['session_id'], request_id) is not None:
        return True
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with authority.db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                            (RETIRED_PREFIX + record['session_id'],)).fetchone() is not None


def _legacy_order(path, record):
    """``_migrate``'s FIFO key of a legacy ticket, or of the pin laid over one (it keeps the ticket's
    fields). A pin without an integer order holds every legacy ticket until it resolves."""
    order = record.get('sequence', record.get('created_at'))
    return (order, path.name) if isinstance(order, int) and not isinstance(order, bool) else (float('-inf'), '')


def _home(authority, actor, profile):
    from gateway.session_authorities import served_profile_name
    home = Path(authority.db.db_path).parent.resolve()
    name = served_profile_name(home)
    if actor.profile_id != authority.profile_id or profile not in (name, 'hermes' if name == 'default' else name):
        raise RuntimeStoreError('profile_mismatch')
    if 'session:submit' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    return home


async def _target(authority, actor):
    ref = lane = None
    row = authority.db.get_session_by_title('Bot Chat')
    if row is None or (displaced := displaceable_bot_chat(authority, row)) is not None:
        # None: a concurrent delivery created it first. A title held by an API conversation no FIFO
        # can own is moved aside first, under the same lock (MSG-21, #140769).
        ref = await _create_bot_chat(authority, actor, displace=None if row is None else displaced)
    if ref is None:
        row = authority.db.get_session_by_title('Bot Chat')
        tip = authority.db.get_compression_tip(row['id'])
        target = authority.db.get_session(tip)
        if target is None:
            raise RuntimeStoreError('not_found')
        from gateway.session_local_migration import resolve_local_target
        from gateway.session_route_recovery import handed_off_ref
        # A Bot Chat handed off to a messaging lane stays the bot's chat there: delivery follows it.
        ref = lane = handed_off_ref(authority, target['id'])
        if ref is None:
            ref = resolve_local_target(authority, actor, target['id'])
    authority.authorize(actor, ref, 'session:submit')
    live = authority.sessions[ref.session_id]
    entry = authority.runner.session_store.lookup_by_session_key(live.route)
    if (live.source.platform != Platform.LOCAL and lane is None) or entry is None:
        raise RuntimeStoreError('admission_conflict')
    return ref, live, entry


def displaceable_bot_chat(authority, row):
    """The id holding the canonical title when the Bot Chat is a conversation no FIFO can own (an
    ``api_`` row an older gateway or a plain ``POST /api/sessions`` minted, a messaging row, an
    unbound CLI/TUI/Desktop row no native open can adopt), else None. Such a row can never be
    adopted (``validate_legacy_row``), renamed (the canonical title is its identity) or renamed
    over the API, so it would refuse every delivery for good. A bound local chat, old local history
    a native open adopts and a local row a login stamped as its own are never moved."""
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX

    if str(row.get('chat_id') or '').startswith('local-'):
        return None  # every delivery asks: a bound local chat answers from its own row, with no read
    db = authority.db
    tip = db.get_session(db.get_compression_tip(row['id']) or row['id'])
    from gateway.session_route_recovery import handed_off_ref, superseded_hand_off
    if tip is not None and (lane := handed_off_ref(authority, tip['id'])) is not None:
        # Handed off (``/handoff``): still the bot's chat while its lane holds it; history once that
        # lane moved on (``/new``), so a fresh Bot Chat takes the title (#143286).
        return row['id'] if superseded_hand_off(authority, lane) else None
    for candidate in (row, tip):
        if candidate is None or str(candidate.get('chat_id') or '').startswith('local-'):
            return None
        chat_id = str(candidate.get('chat_id') or '')
        with db._read_ctx() as conn:
            if conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)',
                            (POLICY_PREFIX + (chat_id or candidate['id']),
                             LEGACY_PREFIX + (chat_id or candidate['id']))).fetchone():
                return None
        if _kept_local_history(db, candidate):
            return None
    return row['id']


def release_superseded_bot_chat(db, store, route, session_id):
    """The lane on *route* moved off *session_id* (its ``/new``, a reset, another ``/resume`` or
    ``/handoff``). A Bot Chat handed off there (``/handoff``) is that lane's history now: free the
    canonical title at once (``_displace_bot_chat``), so the Bots roster and its registry lookup both
    stop naming it and the next click or delivery mints a fresh Bot Chat (#143286). Every route
    transition calls this, and so does boot (an older build's transition)."""
    from gateway.session_route_recovery import _messaging_platform
    from hermes_state_local_migration import LEGACY_LOCAL_SOURCES
    root = (db.get_compression_lineage(session_id) or [session_id])[0]
    row = db.get_session(root)
    if (row is None or row.get('title') != 'Bot Chat' or row.get('created_source') not in LEGACY_LOCAL_SOURCES
            or _messaging_platform(row.get('source')) is None):
        return
    with store._lock:
        entry = store._entry_locked(route)
    if entry is None or (db.get_compression_lineage(entry.session_id) or [entry.session_id])[0] == root:
        return  # the lane still holds it
    if _displace_bot_chat(db, root):
        logger.info("Bot Chat %s was handed off to %s and that lane moved on: moved it aside", root, route)


def release_superseded_bot_chat_at_boot(db, store):
    row = db.get_session_by_title('Bot Chat')
    if row is not None:
        tip = db.get_session(db.get_compression_tip(row['id']) or row['id']) or row
        release_superseded_bot_chat(db, store, tip.get('session_key'), row['id'])


def _kept_local_history(db, row):
    """Whether an unbound row is local history the Bot Chat door must leave in place: a CLI, TUI
    or Desktop row a native open adopts, or one a login stamped as its owner. A local-source row
    no open can ever adopt (a route column left by a crash or reconnect, another profile's stamp)
    is not: it held the title and refused every delivery for good (#141243)."""
    from hermes_state_local_migration import LEGACY_LOCAL_SOURCES, legacy_account_label, validate_legacy_row
    if row.get('source') not in LEGACY_LOCAL_SOURCES:
        return False
    if row.get('user_id') and not legacy_account_label(row):
        return True
    try:
        validate_legacy_row(db, row)
    except RuntimeStoreError:
        return False
    return True


def _moved_aside_by_this_door(authority, holder, pinned):
    """Whether *pinned*, the Bot Chat the producer discovered, is the title holder this delivery's
    ``_target`` just moved aside. The producer pins the id it read before asking, so the first
    delivery after a move would otherwise be refused ``admission_conflict`` for the chat the door
    itself replaced, and land only on the next beat."""
    if holder is None:
        return False
    db = authority.db
    if pinned not in (holder['id'], db.get_compression_tip(holder['id'])):
        return False
    return str((db.get_session(holder['id']) or {}).get('title') or '').startswith('Bot Chat (')


def _displace_bot_chat(db, sid):
    """Free the canonical title from *sid*: it keeps its id and history under a visible name the
    user can find, rename or delete: ``Bot Chat (API <id>)`` for an API conversation, else named by
    its platform (``Bot Chat (telegram <id>)``). False when the title moved meanwhile."""
    source = str((db.get_session(sid) or {}).get('source') or '')
    label = 'API' if source in ('', 'api_server', 'bot_peer_dm') else source
    name = f'Bot Chat ({label} {sid})'
    if len(name) > db.MAX_TITLE_LENGTH:
        name = name[:db.MAX_TITLE_LENGTH - 2] + '…)'

    def write(conn):
        return conn.execute("UPDATE sessions SET title=?, title_source=?, hidden=0 WHERE id=? AND title='Bot Chat'",
                            (name, db.TITLE_SOURCE_USER, sid)).rowcount
    return bool(db._execute_write(write))


async def _create_bot_chat(authority, actor, *, displace=None):
    """No Bot Chat yet (or *displace*, an API conversation holding its title, is moved aside first): mint it the way ``hermes chat --in ~ -c "Bot Chat" --create-if-missing``
    did on the CLI lane this door replaced, so a fresh profile's first delivery has somewhere
    to land instead of being recorded as failed. Same owner step as ``session.create`` with a
    title: lookup, creation and title run under the authority's titled-create lock, with the
    commits off the owner loop (N90). Creation is the actor's own capability; a submit-only
    principal (owner recovery) still gets ``not_found``."""
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('not_found')
    from gateway.session_local import stage_local_session, staged_creation_follows
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_local_title import title_new_session
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_local import commit_local_session
    lock = getattr(authority, '_titled_create_lock', None)
    if lock is None:
        lock = authority._titled_create_lock = asyncio.Lock()
    async with lock:
        held = authority.db.get_session_by_title('Bot Chat')
        if held is not None:
            if displace is None or held['id'] != displace or displaceable_bot_chat(authority, held) != displace:
                return None
            logger.warning("Bot Chat title held by %s, a conversation no FIFO can own: moving it aside", displace)
            if not await tracked_write(authority, partial(_displace_bot_chat, authority.db, displace)):
                return None
        # The Bot Chat follows its profile whoever mints it (a Bot or cron delivery, a peer DM),
        # the same contract Desktop's create asks for. A later profile change reaches it.
        request_id = _bot_chat_creation_id(authority, actor)
        params = {'request_id': request_id, 'source': 'cli', 'cwd': str(Path.home()),
                  'follow_profile_config': True}
        # A creation staged by an older build, before this flag, resumes as it was staged.
        if not staged_creation_follows(authority, actor, request_id):
            params.pop('follow_profile_config')
        ref = stage_local_session(authority, actor, params)
        if isinstance(ref, dict):
            await tracked_write(authority, partial(commit_local_session, authority.db, epoch=authority.epoch, receipt=ref))
            ref = restore_local_session(authority, ref['session_id'])
        await tracked_write(authority, partial(title_new_session, authority, ref, 'Bot Chat'))
        return ref


def _bot_chat_creation_id(authority, actor):
    """The first creation identity whose session is not spent. Generation 0 keeps the historical
    ``bot-chat:<profile>`` id; a generation is spent once its session was deleted (retired: a retry
    must never resurrect it), renamed (titled, but not "Bot Chat" — or the lookup had found it) or
    handed off to a messaging lane, and a replacement takes the next one. Derived from durable rows
    only, so a restart and two concurrent deliveries (serialized by the titled-create lock) pick the
    same identity. An existing UNtitled local session is the same creation interrupted before its
    title: it is reused."""
    from gateway.session_local_recovery import local_identity
    from gateway.session_route_recovery import _messaging_platform
    from hermes_state_mutation_retirement import first_generation
    base = 'bot-chat:' + authority.profile_id
    request_id = lambda generation: base if generation == 0 else f'{base}:{generation}'

    def spent(conn, sid):
        row = conn.execute('SELECT title, source FROM sessions WHERE id=?', (sid,)).fetchone()
        # Handed off (``/handoff``), it is a messaging lane's conversation: its receipt went with it,
        # and a reset's head carried its name there (``hermes_state_local.hand_off_conversations``).
        return row is not None and (bool(row[0]) or _messaging_platform(row[1]) is not None)
    generation, _sid = first_generation(
        authority.db, lambda n: local_identity(authority.profile_id, actor.subject, request_id(n)), spent=spent)
    return request_id(generation)


def _admission_outcome(authority, admission_id, fallback=None):
    """``(status, reply)`` of one admission, read from its committed result, never transcript recency."""
    row = get_session_admission(authority.db, admission_id=admission_id)
    if row is None:
        return _expired_outcome(authority, fallback)
    # A terminal row without a result blob is still definitive when the outcome says the
    # input never ran (cancelled) or was refused/failed; ambiguous is reserved for the
    # durable unknown state and for missing evidence (completed/interrupted without result).
    status = {'queued': 'queued', 'started': 'claimed', 'unknown': 'ambiguous',
              'terminal': {'cancelled': 'cancelled', 'rejected': 'failed', 'failed': 'failed'}.get(row['outcome'], 'ambiguous')}[row['status']]
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, admission_id)
    reply = (fallback or {}).get('reply', '')
    if saved is not None:
        reply = saved['result'].get('final_response', '')
        status = 'settled' if row['outcome'] == 'completed' else 'failed'
        # A successful bare silence marker is a delivery decision (same rule as the gateway and
        # the live Bot Chat completion): the turn stays in the target's transcript, the sender
        # never sees the marker as prose.
        from gateway.response_filters import is_intentional_silence_response
        if status == 'settled' and is_intentional_silence_response(reply):
            reply = ''
    elif fallback is not None and fallback.get('status') in {'settled', 'failed'}:
        status = fallback['status']
    return status, reply


def _expired_outcome(authority, record):
    """A receipt outlives its admission once a deleted target's tombstone passes the retry horizon
    (``prune_retired_tombstones``). A final receipt keeps the answer it already holds; a pending one
    whose session was deleted (retired, or its row gone) can never learn its outcome and becomes
    ``ambiguous``, never resent. A live session missing its admission is still a storage fault."""
    from hermes_state_mutation_retirement import retired_session
    record = record or {}
    if record.get('status') in _TERMINAL:
        return record['status'], record.get('reply', '')
    session_id = record.get('session_id')
    if session_id and retired_session(authority.db, session_id):
        return 'ambiguous', record.get('reply', '')
    if session_id and authority.db.get_session(session_id) is None:
        # Never resent either way, but a row gone without its fence is not a plain delete: store damage,
        # or a marker an older release reclaimed. Logged once: the receipt is final from here on.
        logging.getLogger(__name__).warning(
            'Bot delivery %s: session %s is gone with no deletion marker; settled as ambiguous',
            record.get('delivery_id'), session_id)
        return 'ambiguous', record.get('reply', '')
    raise RuntimeStoreError('storage_unavailable')


def _retry_admission(authority, record):
    """The delivery's one retry admission id, read from the FIFO by its derived identity, so every
    reader (an in-memory record, the receipt file, an owner restart) follows the same retry."""
    return _admission_at(authority, record['session_id'], _retry_identity(record['delivery_id']))


def _admission_at(authority, session_id, request_id):
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT admission_id FROM session_admissions WHERE target_session_id=? AND request_id=?',
                           (session_id, request_id)).fetchone()
    return row[0] if row else None


def _result(authority, record):
    """The delivery's receipt. Once the one transient-failure retry was admitted, the sender's
    outcome is the retry's: ``status``/``reply`` follow it (``retry_admission_id``).

    A ``failed`` receipt carries ``error`` + a typed ``reason`` (``tools.bot_failure_reasons``):
    the relay's Desktop forwards ``res.reason`` to the sender, which cannot re-derive it from a
    reply that lacks the provider text."""
    retry_id = _retry_admission(authority, record)
    current = retry_id or record['admission_id']
    if retry_id:
        status, reply = _admission_outcome(authority, retry_id)
    else:
        status, reply = _admission_outcome(authority, record['admission_id'], record)
    error = reason = None
    if status == 'failed':
        try:
            error, reason = _failure(authority, current)
        except RuntimeStoreError as exc:
            if exc.reason != 'not_found':
                raise
            # Its admission aged out after the chat was deleted: the receipt's own answer stands.
            from tools.bot_failure_reasons import UNKNOWN
            error, reason = record.get('error') or 'Bot Chat delivery failed', record.get('reason') or UNKNOWN
    return {k: v for k, v in dict(status=status, delivery_id=record['delivery_id'],
        profile_home=record['profile_home'], session_id=record['session_id'],
        admission_id=record['admission_id'], message=record['message'], reply=reply,
        retry_admission_id=retry_id, error=error, reason=reason).items() if v is not None}


def _failure(authority, admission_id):
    """``(error, reason)`` of a failed admission from its committed result: the raw provider error
    and the turn loop's typed verdict, classified like every other Bot lane. A refusal that never
    ran (no result) has no provider text: ``unknown``."""
    from gateway.session_results import admission_result
    from tools.bot_failure_reasons import classify_agent_error, turn_failure_text
    saved = (admission_result(authority.db, admission_id) or {}).get('result') or {}
    error = saved.get('error') or saved.get('final_response') or 'Bot Chat delivery failed'
    return error, classify_agent_error(turn_failure_text(saved.get('error'), saved.get('failure_reason')) or error)


def _delivery_result(authority, record):
    result = _result(authority, record)
    if result['status'] == 'failed' and _retry_eligible(authority, record):
        result['status'] = 'claimed'  # owner receipt writer still owes the one allowed retry
    return result


def peer_wait_admission(authority, record):
    """``(admission_id, interim)`` a live waiter must follow; ``(None, False)`` once the delivery
    settled for good. The retry, once admitted, is the only admission whose future fires again.
    An original that failed while its one retry is still eligible is ``interim``: pending, never
    the answer, while the owner's receipt task admits the retry (or records why it would not;
    the receipt file is re-read for that, the caller's in-memory record is stale)."""
    from tools.bot_live_delivery import read_delivery_result
    retry_id = _retry_admission(authority, record)
    if retry_id:
        return (retry_id, False) if _admission_outcome(authority, retry_id)[0] in {'queued', 'claimed'} else (None, False)
    status = _admission_outcome(authority, record['admission_id'], record)[0]
    if status in {'queued', 'claimed'}:
        return record['admission_id'], False
    if status == 'failed':
        saved = read_delivery_result(record['profile_home'], record['delivery_id']) or record
        if _retry_eligible(authority, {**record, 'retry': saved.get('retry')}):
            return record['admission_id'], True
    return None, False


def _retry_identity(key):
    return 'bot:' + key + ':retry'


def _retry_eligible(authority, record):
    """Exactly the main-lane gate (``tools.bot_failure_reasons.result_retry_action``): the original
    admission SETTLED ``failed`` with a committed result whose error classifies transient (429/5xx/
    context overflow). queued/started/unknown admissions never qualify — unknown execution is never
    replayed — and the retry itself is never retried."""
    if (record.get('retry') or {}).get('refused') or _retry_admission(authority, record):
        return False
    row = get_session_admission(authority.db, admission_id=record['admission_id'])
    if row is None or row['status'] != 'terminal' or row['outcome'] != 'failed':
        return False
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, record['admission_id'])
    if saved is None:
        return False
    from tools.bot_failure_reasons import RETRY_NONE, result_retry_action
    return result_retry_action(saved['result']) != RETRY_NONE


async def _maybe_retry(authority, home, path, record):
    """Admit the delivery's one retry under a DERIVED identity recorded on the same receipt, so a
    repeated deliver(), an owner restart or a second settle re-reads that admission instead of
    minting another. The identity is written before the admission: a death in that two-store window
    re-admits the SAME identity, which the FIFO answers with the existing row (never a second turn).

    No user-row adoption: a transient owner-side failure persists the DM row and then CLOSES it with the
    durable failed-turn boundary (``agent.conversation_loop._close_durable_failed_turn`` /
    ``_hmwa_close_failed_turn``, #107070), so ``agent.session_persistence.adopt_unanswered_turn`` would
    decline by contract (a plain assistant row follows the DM) and the retry is a fresh turn after that
    boundary. A failure that left the DM as an OPEN tail (the context-pressure classes skip the closer)
    is not retried: the owner execution has no adoption seam, and the retry's DM would merge into it."""
    if not _retry_eligible(authority, record):
        return
    identity = _retry_identity(record['delivery_id'])
    # The retry keeps the access its delivery was admitted on: ``operator`` is recorded only when the
    # transport granted ``session:operator`` (a local delivery into api-peer's Bot Chat), and the
    # session check below still pins the retry to the admitted target.
    operator = {'session:operator'} if record.get('operator') is True else set()
    actor = Principal(record['principal_id'], authority.profile_id,
                      frozenset({'session:submit', 'session:read'} | operator), 'bot-delivery-retry')
    try:
        authority._require_admission_open()
        ref, live, entry = await _target(authority, actor)
        if ref.session_id != record['session_id']:
            raise RuntimeStoreError('admission_conflict')
        if authority.db.latest_conversation_role(entry.session_id) == 'user':
            raise RuntimeStoreError('open_user_tail')
    except RuntimeStoreError as exc:
        if exc.reason == 'runtime_draining':
            # No retry was attempted. Keep the local projection pending until startup recovery
            # can perform the same derived retry, rather than exposing an interim failure.
            record.update(status='claimed', reply='')
            return
        record['retry'] = {'identity': identity, 'refused': exc.reason}
        await write_receipt(home, record)
        return
    event = MessageEvent(text=record['message'], source=live.source, internal=True,
        message_id=identity, metadata={'gateway_session_key': live.route, 'gateway_session_id': entry.session_id})
    from gateway.session_automation import automation_notification_metadata
    event.metadata.update(automation_notification_metadata(record))
    if record.get('author') is not None:
        event.metadata['turn_author'] = dict(record['author'])
    record['retry'] = {'identity': identity}
    record.update(status='queued', reply='')
    await write_receipt(home, record)
    try:
        receipt = await authority.admit_automation(authority.runner._adapter_for_source(live.source), event, identity)
    except RuntimeStoreError as exc:
        if exc.reason != 'queue_full':
            raise
        # Nothing was admitted: keep the delivery pending (as for a drain) so the next wake or
        # delivery performs the same derived retry once the FIFO has room.
        record.update(status='claimed', reply='')
        return
    record.update(_result(authority, record))
    if record['status'] in {'queued', 'claimed'}:
        _watch_reply(authority, home, record['delivery_id'], receipt.admission_id)
    await write_receipt(home, record)
    remember_receipt(authority, home, record)


async def _record_reply(authority, home, key, future):
    paused = False
    try:
        await asyncio.shield(future)
    except RuntimeStoreError:
        # A paused FIFO is still queued/unknown. Project that durable state;
        # never turn a wakeup refusal into a failed delivery or a fresh retry.
        paused = True
    async with mailbox_lock(authority):
        root, record = await read_receipt(home, key)
        if record is None:
            return
        path = root / f'{key}.json'
        record.update(_result(authority, record))
        if not paused:
            await _maybe_retry(authority, home, path, record)
        record.update(_delivery_result(authority, record))
        await write_receipt(home, record)
        remember_receipt(authority, home, record)


async def refresh_receipt(authority, home, key):
    async with mailbox_lock(authority):
        root, record = await read_receipt(home, key)
        if record is None:
            return
        record.update(_result(authority, record))
        if record['status'] in {'queued', 'claimed'}:
            _watch_reply(authority, home, key, record.get('retry_admission_id') or record['admission_id'])
        await _maybe_retry(authority, home, root / f'{key}.json', record)
        await write_receipt(home, record)
        remember_receipt(authority, home, record)


def relay_operation(connection, operation, params):
    authority, actor = connection.authority, connection.actor
    from gateway.session_authorities import require_serving, served_profile_name
    require_serving(authority)  # the outbox/roster/replies belong to the home's next owner
    home = Path(authority.db.db_path).parent.resolve()
    name = served_profile_name(home)
    _home(authority, actor, name)
    if 'session:control' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    fields = {'roster': {'agents'}, 'outbox': set(), 'reply': {'id', 'reply', 'error', 'reason'}}
    if set(params) - fields[operation]:
        raise RuntimeStoreError('invalid_params')
    from tools.bot_relay import write_remote_roster, claim_pending_envelopes, write_reply
    if operation == 'roster':
        return {'count': write_remote_roster(home, params.get('agents'))}
    if operation == 'outbox':
        return {'envelopes': claim_pending_envelopes(home)}
    try:
        write_reply(home, params.get('id'), reply=params.get('reply', ''),
                    error=params.get('error', ''), reason=params.get('reason', ''))
    except ValueError as exc:
        raise RuntimeStoreError('admission_conflict') from exc
    return {'ok': True}


async def _migrate(authority, actor, home, root):
    # FIFO: a legacy ticket queued after an unresolved legacy pin waits for that pin's requeue. An
    # owner-state refusal below raises out of the pass, so nothing after the refused ticket runs.
    root, records, barrier = await _scan_owner_receipts(authority, home)
    legacy = [(path, record) for path, record in records
              if record and 'owner' in record and not record.get('admission_id')
              and record['status'] in {'queued', 'claimed'}
              and (barrier is None or _legacy_order(path, record) < barrier)]
    if not legacy:
        return
    ref, live, entry = await _target(authority, actor)
    for path, record in sorted(legacy, key=lambda item: _legacy_order(*item)):
        owner = record['owner']
        if owner['profile_home'] != str(home):
            continue
        if authority.db.get_compression_tip(owner['session_id']) != entry.session_id:
            continue
        from hermes_cli.active_sessions import active_session_liveness_guard
        with active_session_liveness_guard(owner['session_id'], registry_home=home) as active:
            if active:
                raise RuntimeStoreError('runtime_coordination_required')
        if record['status'] == 'claimed':
            record.update(status='ambiguous', reason='unknown_execution')
            await write_receipt(home, record, expect='claimed')
            continue
        await _admit(authority, actor, home, root, _delivery_id(record['delivery_id']),
                     record['message'], ref, live, entry, author=parse_turn_author(record.get('author')), legacy=record,
                     notification_category=record.get('notification_category', 'result'))


async def recover_bot_deliveries(authority):
    """Rebuild derivative replies and queued legacy admissions at owner startup."""
    home = Path(authority.db.db_path).parent.resolve()
    async with mailbox_lock(authority):
        root, records, _ = await _scan_owner_receipts(authority, home)
        for path, record in records:
            if not record or record.get('profile_home') != str(home) or not record.get('admission_id'):
                continue
            try:
                record.update(_result(authority, record))
                if record['status'] in {'queued', 'claimed'}:
                    _watch_reply(authority, home, record['delivery_id'],
                                 record.get('retry_admission_id') or record['admission_id'])
                if record['status'] not in {'queued', 'claimed'}:
                    await _maybe_retry(authority, home, path, record)
            except RuntimeStoreError as exc:
                # One receipt the store cannot answer must not stop the owner (and, multiplexed, every
                # profile with it): log it and recover the rest. The next start reads it again (a
                # retry already pinned on it is then admitted once).
                logging.getLogger(__name__).error('Bot delivery %s not recovered: %s (%s)',
                                                  record.get('delivery_id'), exc.reason, path, exc_info=True)
                continue
            await write_receipt(home, record)
            remember_receipt(authority, home, record)
        row = authority.db.get_session_by_title('Bot Chat')
        if row is None:
            return
        target = authority.db.get_session(authority.db.get_compression_tip(row['id']))
        if target is None:
            return
        from hermes_state_local import POLICY_PREFIX
        with authority.db._read_ctx() as conn:
            bound = conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                (POLICY_PREFIX + (target.get('chat_id') or target['id']),)).fetchone()
        if not bound:
            return  # Unowned history requires a native ticket, never the first remote sender.
        actor = Principal(target['user_id'], authority.profile_id,
                          frozenset({'session:submit', 'session:read'}), 'bot-owner-recovery')
        await _migrate(authority, actor, home, root)


def _watch_reply(authority, home, key, admission_id):
    watchers = getattr(authority, '_bot_reply_watchers', None)
    if watchers is None:
        watchers = authority._bot_reply_watchers = {}
    previous = watchers.get(key)
    if previous is not None and previous[0] == admission_id and not previous[2].done():
        return
    future = authority.waiters.setdefault(admission_id, asyncio.get_running_loop().create_future())
    task = asyncio.create_task(_record_reply(authority, home, key, future))
    watchers[key] = (admission_id, task, future)
    def done(finished):
        if watchers.get(key) == (admission_id, finished, future):
            watchers.pop(key, None)
    task.add_done_callback(done)
    track_receipt_task(authority, task)


async def deliver(connection, params):
    from gateway.session_runtime_workers import track_mutation
    connection.authority._require_admission_open()
    return await asyncio.shield(track_mutation(connection.authority, _deliver(connection, params)))


def _require_same_delivery(record, actor, message, author, category):
    """A delivery id names one delivery: the same message, principal, author and category (the
    mailbox writer's identity), whether its receipt is admission-backed or refused."""
    if (record['message'] != message or record['principal_id'] != actor.subject
            or record.get('author') != author
            or record.get('notification_category', 'result') != category):
        raise RuntimeStoreError('admission_conflict')


def _delivery_params(authority, actor, params):
    """``(home, key, message, author, category)`` of one ``deliver``/``settle`` request: the
    delivery's full identity, validated the same way for both verbs."""
    home = _home(authority, actor, params.get('profile'))
    if set(params) - {'id', 'profile', 'message', 'session_id', 'author', 'notification_category'}:
        raise RuntimeStoreError('invalid_params')
    from gateway.session_automation import automation_notification_metadata
    notification = automation_notification_metadata(params)
    category = notification.get('notification_category', 'result')
    try:
        key = _delivery_id(params.get('id'))
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    message = params.get('message')
    # Main capped only a relayed bot DM (MESSAGE_MAX_CHARS + attribution headroom), at the relay
    # edge; a trusted producer's report (cron bot-chat, its deferred drain) carries no author and
    # was admitted whole, so it gets a resource bound only.
    limit = _BOT_DM_MAX_CHARS if params.get('author') is not None else _PRODUCER_MAX_CHARS
    if not isinstance(message, str) or not message.strip() or len(message) > limit:
        raise RuntimeStoreError('invalid_params')
    author = parse_turn_author(params.get('author'))
    if params.get('author') is not None and (not isinstance(params['author'], dict) or author is None):
        raise RuntimeStoreError('invalid_params')
    return home, key, message, author, category


def _refused_answer(record):
    return {k: record[k] for k in ('status', 'refused', 'delivery_id', 'profile_home', 'session_id',
                                   'message', 'reply', 'error', 'reason')}


async def _receipt_answer(authority, actor, home, record, identity, retry_path=None):
    """The answer an id's existing receipt gives, or None when it has none yet: an admitted id's
    published result (after its one retry when ``retry_path`` is given), or a terminal refusal
    (``refuse_pin``, ``settle``), which is the id's answer for good. Both check the identity."""
    if record is None or not (record.get('admission_id') or record.get('refused') is True):
        return None
    _require_same_delivery(record, actor, *identity)
    if not record.get('admission_id'):
        return _refused_answer(record)
    authority.authorize(actor, SessionRef(authority.profile_id, record['session_id']), 'session:submit')
    if retry_path is not None:
        await _maybe_retry(authority, home, retry_path, record)
    return await _publish_result(authority, home, record)


async def _deliver(connection, params):
    authority, actor = connection.authority, connection.actor
    home, key, message, author, category = _delivery_params(authority, actor, params)
    identity = (message, author, category)
    authority._require_admission_open()
    async with mailbox_lock(authority):
        root, record = await read_receipt(home, key)
        path = root / f'{key}.json'
        answer = await _receipt_answer(authority, actor, home, record, identity, retry_path=path)
        if answer is not None:
            return answer
        try:
            await _migrate(authority, actor, home, root)
            _, record = await read_receipt(home, key)
            answer = await _receipt_answer(authority, actor, home, record, identity, retry_path=path)
            if answer is not None:
                return answer
            if record is not None:
                raise RuntimeStoreError('unknown_execution')
            holder = authority.db.get_session_by_title('Bot Chat')
            ref, live, entry = await _target(authority, actor)
            pinned = params.get('session_id', entry.session_id)
            if pinned != entry.session_id and not _moved_aside_by_this_door(authority, holder, pinned):
                raise RuntimeStoreError('admission_conflict')
            return await _admit(authority, actor, home, root, key, message, ref, live, entry, author=author,
                                notification_category=category)
        except RuntimeStoreError as exc:
            raise await _held_by_another_writer(home, key, exc) or exc


async def settle(connection, params):
    from gateway.session_authorities import require_serving
    from gateway.session_runtime_workers import track_mutation
    # Retirement joins tracked work but cannot fence a later write: receipts carry no epoch and
    # mailbox_lock is this authority's own, so a stale connection is refused before it schedules.
    require_serving(connection.authority)
    connection.authority._require_admission_open()
    return await asyncio.shield(track_mutation(connection.authority, _settle(connection, params)))


async def _settle(connection, params):
    """Finalize a delivery the sender was told may not have been admitted: under the mailbox lock
    every ``_admit`` holds, either settle the id as never sent or answer with what it already is.

    A ``deliver`` refusal marked ``admitted=False`` proves absence only while that request held
    the lock; a same-id attempt may admit right after. The sender calls this before it says "NOT
    sent": with no receipt and no admission row for ``bot:<id>``, a terminal ``refused`` receipt
    bound to the full identity is written, and every later ``deliver`` of the id answers with it
    instead of admitting. An admitted id gets its receipt; an in-flight pin stays unknown. Only the
    Bot DM tool calls this, so cron's same-id retries of a refusal are unchanged.

    The writer is not re-checked: it may release between the refusal and this call, and "nothing
    was admitted" is all that is serialized here. So the receipt claims only that, and any
    ``session:submit`` principal of the profile can settle an id it never delivered: the trust
    ``deliver`` already grants it over that id."""
    authority, actor = connection.authority, connection.actor
    home, key, message, author, category = _delivery_params(authority, actor, params)
    async with mailbox_lock(authority):
        _, record = await read_receipt(home, key)
        if record is None:
            record = _not_sent_record(authority, actor, home, key, message, author, category)
            await write_receipt(home, record)
            logging.getLogger(__name__).info('Bot Chat delivery %s settled as not sent (%s)', key, record['reason'])
        _require_same_delivery(record, actor, message, author, category)
        answer = await _receipt_answer(authority, actor, home, record, (message, author, category))
        if answer is None:
            raise RuntimeStoreError('unknown_execution')
        return answer


def _not_sent_record(authority, actor, home, key, message, author, category):
    """The never-admitted id's terminal receipt, in ``refuse_pin``'s shape. Refuses (unknown) when
    the FIFO holds a ``bot:<id>`` row anyway or there is no Bot Chat to name."""
    with authority.db._read_ctx() as conn:
        admitted = conn.execute('SELECT 1 FROM session_admissions WHERE request_id=?', ('bot:' + key,)).fetchone()
    row = authority.db.get_session_by_title('Bot Chat')
    if admitted is not None or row is None:
        raise RuntimeStoreError('unknown_execution')
    from gateway.session_automation import automation_notification_metadata
    record = dict(delivery_id=key, profile_home=str(home), session_id=authority.db.get_compression_tip(row['id']),
                  principal_id=actor.subject, message=message, status='failed', refused=True, reply='',
                  error='Settled as not sent after a session-writer refusal', reason=HELD_NOT_SENT)
    record.update(automation_notification_metadata({'notification_category': category}))
    if author is not None:
        record['author'] = dict(author)
    return record


async def _held_by_another_writer(home, key, exc):
    """The settleable form of a ``runtime_coordination_required`` refusal, or None.

    Another session writer holds the chat (a pre-authority Desktop backend's lease on a legacy Bot
    Chat, a live CLI). Nothing was admitted under this id when it still has no receipt: ``_admit``
    pins one before it admits and withdraws it only when nothing committed. That holds only while
    this request has the lock, so ``admitted=False`` is a hint, not the verdict: the sender asks
    ``settle`` before it reports a no-send. Any other refusal keeps its unknown outcome."""
    if exc.reason != HELD_NOT_SENT or (await read_receipt(home, key))[1] is not None:
        return None
    held = RuntimeStoreError(exc.reason, field='session_writer',
                             detail='another session writer holds the Bot Chat; nothing was admitted yet')
    held.admitted = False
    return held


async def _admit(authority, actor, home, root, key, message, ref, live, entry, author=None, legacy=None,
                 notification_category='result'):
    from gateway.session_automation import automation_notification_metadata
    notification = automation_notification_metadata({'notification_category': notification_category})
    path = root / f'{key}.json'
    event = MessageEvent(text=message, source=live.source, internal=True,
        message_id='bot:' + key, metadata={'gateway_session_key': live.route,
                                         'gateway_session_id': entry.session_id})
    event.metadata.update(notification)
    if author is not None:
        event.metadata['turn_author'] = dict(author)
    # Pin the physical target before committing. A process death in this
    # two-store window leaves an explicit unknown record, never a new target.
    record = dict(legacy or {}, delivery_id=key, profile_home=str(home), session_id=ref.session_id,
        principal_id=actor.subject, message=message, status='ambiguous')
    record.update(notification)
    if 'session:operator' in actor.capabilities:
        record['operator'] = True  # admitted on operator access: its one retry keeps it (``_maybe_retry``)
    if author is not None:
        record['author'] = dict(author)
    # Preserve a foreign cancellation or claim that won after our legacy scan.
    if not await write_receipt(home, record, expect=(legacy or {}).get('status')):
        if legacy is not None:
            return None
        raise RuntimeStoreError('admission_conflict')
    try:
        receipt = await authority.admit_automation(authority.runner._adapter_for_source(live.source), event, 'bot:' + key)
    except RuntimeStoreError as exc:
        if _admission_at(authority, ref.session_id, 'bot:' + key) is not None:
            raise  # committed after all: the pin stays the explicit unknown record
        if exc.reason in _OWNER_STATE_REFUSALS:
            if legacy is None:
                # Refused before any INSERT, not for the delivery itself: the same-id resend admits.
                await withdraw_pin(home, key)
        else:
            # Refused for good with nothing committed: a terminal receipt, not an unknown pin.
            logging.getLogger(__name__).info('Bot Chat delivery %s refused (%s)', key, exc.reason)
            await refuse_pin(home, key)
        raise
    record.update(status='canonical', admission_id=receipt.admission_id)
    record.update(_result(authority, record))
    if record['status'] in {'queued', 'claimed'}:
        _watch_reply(authority, home, key, receipt.admission_id)
    if record['status'] not in {'queued', 'claimed'}:
        await _maybe_retry(authority, home, path, record)
    return await _publish_result(authority, home, record)


async def _publish_result(authority, home, record):
    """Publish the delivery's projection and answer with exactly that projection. The write is
    off-loop, so the admission may settle while it runs; re-deriving the answer afterwards would
    hand the sender a reply the receipt file does not hold yet (the reply watcher publishes it
    once this delivery releases the mailbox lock)."""
    result = _delivery_result(authority, record)
    record.update(result)
    await write_receipt(home, record)
    remember_receipt(authority, home, record)
    return result
