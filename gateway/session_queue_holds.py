"""Why a queued admission waits, and the holds a session's drain honours before it claims its head.

A queued row is never refused for a reason that passes: the drain parks on its head and each pending
snapshot names the reason (``hold_reason``) so Desktop and ``/queue list`` can say why it waits.

- ``slot``: ``max_concurrent_sessions`` is full. The head waits instead of being answered with the
  refusal notice, and a freed slot goes to the best waiter: people before automation (cron, kanban,
  a2a, webhook, automation notices), then oldest first. Waiters are process-wide (one cap across
  every profile this runner serves) and refreshed on each poll, so a drain that stopped waiting
  (cancelled head, retirement) never keeps a slot from the others. The winner RESERVES the slot
  until its turn shows up as running (or ``_SLOT_RESERVE_S`` passes): its claim and turn start
  await, and another drain polling meanwhile must not count the same slot as free. The owning drain
  drops it when that turn ends or when it exits without one (``release_slot``), so a short turn no
  other drain saw running never pins the slot.
- ``editing``: a client is editing the row (``prompt.hold_queued``). The drain does not start it
  until the edit is saved or abandoned; the hold lapses after ``EDIT_HOLD_S`` so a crashed client
  cannot pin the FIFO. Each editor (its holder token under the signed-in subject, else its
  connection) holds separately: one editor closing never releases another's hold, and a reconnected
  editor still releases its own.
- ``session_busy`` (a turn or worker runs on the session), ``restarting``, ``paused`` (a lost turn
  waits for the user): read from the drain's own pause reasons.

Other admission gates name their reason by writing ``authority.hold_reasons[session_id]`` while they
hold the head; the drain clears it once the FIFO moves.
"""
import logging
import time

logger = logging.getLogger(__name__)

EDIT_HOLD_S = 120.0
# A waiter not refreshed for this long stopped polling (its drain exited): it no longer ranks.
_WAITER_STALE_S = 2.0
# A won slot stays reserved for its drain this long at most while the claim and turn start await.
_SLOT_RESERVE_S = 30.0
_AUTOMATION_PRINCIPALS = ('cron-owner', 'automation:', 'a2a:', 'msgraph:')
_AUTOMATION_PAYLOADS = ('kanban_card_v1', 'local_automation_v1')
_PAUSE_REASONS = {'session_busy': 'session_busy', 'runtime_draining': 'restarting', 'unknown_execution': 'paused'}


def _holds(authority, admission_id, now):
    """The admission's live ``{holder: expiry}`` (lapsed holders dropped)."""
    holders = vars(authority).get('edit_holds', {}).get(admission_id) or {}
    for holder in [h for h, expiry in holders.items() if expiry <= now]:
        del holders[holder]
    return holders


def edit_held(authority, admission_id) -> bool:
    if _holds(authority, admission_id, time.monotonic()):
        return True
    vars(authority).get('edit_holds', {}).pop(admission_id, None)
    return False


def set_edit_hold(authority, admission_id, held: bool, holder=None) -> None:
    """Hold or release ``admission_id`` for ``holder``; releasing with no holder ends every hold
    (the row was replaced, cancelled or started)."""
    now = time.monotonic()
    for stale in [aid for aid in authority.edit_holds if not _holds(authority, aid, now)]:
        del authority.edit_holds[stale]
    if held:
        authority.edit_holds.setdefault(admission_id, {})[holder] = now + EDIT_HOLD_S
    elif holder is None:
        authority.edit_holds.pop(admission_id, None)
    else:
        holders = authority.edit_holds.get(admission_id, {})
        holders.pop(holder, None)
        if not holders:
            authority.edit_holds.pop(admission_id, None)


def is_automation(authority, row, live) -> bool:
    """A turn no person is waiting at a chat for: it yields a freed slot to people. A local session
    launched by a non-interactive surface (a2a forwards, cron, kanban, bot rooms) is automation
    whoever submits into it; that frozen launch source is the trusted producer marker."""
    payload = row.get('payload') or {}
    platform = getattr(getattr(live, 'source', None), 'platform', None)
    if (str(row.get('principal_id') or '').startswith(_AUTOMATION_PRINCIPALS)
            or any(key in payload for key in _AUTOMATION_PAYLOADS)
            or getattr(platform, 'value', platform) == 'webhook'):
        return True
    if getattr(platform, 'value', platform) != 'local':
        return False
    from gateway.session_queued_controls import _interactive
    return not _interactive(authority, row['target_session_id'])


def _slot_blocked(authority, ref, live, row) -> bool:
    from hermes_cli.active_sessions import resolve_max_concurrent_sessions
    runner = authority.runner
    cap = resolve_max_concurrent_sessions(getattr(runner, 'config', None))
    if cap is None:
        return False
    waiters = runner.__dict__.setdefault('_queue_slot_waiters', {})
    reserved = runner.__dict__.setdefault('_queue_slot_reserved', {})
    key = (authority.profile_id, ref.session_id)
    reserved.pop(key, None)  # this drain polls again: its earlier win is spent or retried here
    if runner._is_session_running(live.route):
        waiters.pop(key, None)
        return False
    now = time.monotonic()
    # A reserved slot frees once its turn counts as running (or the reservation lapses).
    for other in [k for k, (route, until) in reserved.items() if until <= now or runner._is_session_running(route)]:
        del reserved[other]
    rank = (1 if is_automation(authority, row, live) else 0, waiters.get(key, (0, now))[1], key)
    waiters[key] = (*rank, now)
    for other in [k for k, entry in waiters.items() if now - entry[3] > _WAITER_STALE_S]:
        del waiters[other]
    free = cap - runner._running_agent_count() - len(reserved)
    ahead = sum(1 for entry in waiters.values() if entry[:3] < rank)
    if ahead < free:
        waiters.pop(key, None)
        reserved[key] = (live.route, now + _SLOT_RESERVE_S)
        return False
    return True


def release_slot(authority, ref) -> None:
    """``ref``'s drain no longer needs its won slot reserved: its turn ended, or it exited without one."""
    runner = getattr(authority, 'runner', None)  # an authority with no runner holds no reservations
    reserved = vars(runner).get('_queue_slot_reserved') if runner is not None else None
    if reserved:
        reserved.pop((authority.profile_id, ref.session_id), None)


def head_hold(authority, ref, live, row):
    """The reason the drain must not claim ``row`` (the queued head) yet, or None."""
    if edit_held(authority, row['admission_id']):
        return 'editing'
    if _slot_blocked(authority, ref, live, row):
        return 'slot'
    return None


def note_hold(authority, ref, reason) -> None:
    """The drain parks on its head for ``reason`` (None: it moved). Published once per change."""
    sid = ref.session_id
    previous = authority.hold_reasons.get(sid)
    reason = _PAUSE_REASONS.get(reason, reason)
    if reason == previous:
        return
    if reason is None:
        authority.hold_reasons.pop(sid, None)
    else:
        authority.hold_reasons[sid] = reason
        logger.info('Session %s queue held: %s', sid, reason)
    if sid in authority.sessions:
        authority._publish_pending(ref)


def hold_reason(authority, row):
    """Why the queued ``row`` is not running yet (``PendingAdmission.hold_reason``); None otherwise."""
    if row.get('status') != 'queued':
        return None
    if edit_held(authority, row['admission_id']):
        return 'editing'
    # ``vars``: an authority projected for a resumed CLI's snapshot carries no live drain state.
    sid, state = row['target_session_id'], vars(authority)
    if sid in state.get('executing', ()):
        return 'session_busy'
    return state.get('hold_reasons', {}).get(sid)
