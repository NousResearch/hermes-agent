"""A local viewer's turn on a messaging session (TUI-8, NousResearch/hermes-agent#134889).

Always: a viewer turn runs on whichever session the lane resolves to (``get_or_create_session`` keys by
source), so a viewer's turn on a row the lane has certainly moved past is refused (``lane_state``): a
historical row is never silently continued in the lane's current session.

Local divergence, opt-in through ``gateway.session.viewer_replies_stay_local`` (default false = upstream):
the turn answers in the panel, never on the platform; a historical row is adopted as the viewer's own
local session (``gateway/session_viewer_adoption.py``); and a lane with no routing entry admits a viewer
turn only when recovery would land it in that very row.

Before the canonical authority (#106742) a viewer resumed a Telegram/Weixin/... row in its own
in-process TUI session (``platform="tui"``): reply, progress and prompts stayed in the panel. Through
the authority that turn runs on the row's messaging source, and every egress of the turn (final reply,
typing, tool progress, stream drafts, approval and clarify cards) resolves its transport through
``_delivery_adapter_for``. While a viewer's admission executes, that seam answers for the session's
own chat with the profile's ``LocalSessionAdapter``, the adapter local sessions already deliver
through: it ACKs sends (the event stream carries the turn) and prompt cards, which the turn runner then
registers in the session's ``PendingControls`` for a viewer to answer. Egress the turn addresses to
any other chat resolves as usual. Platform input and automation never carry ``local_operator_v1`` and
keep their platform.

``API_SERVER`` sessions are excluded on purpose: their reply is the HTTP response of the API transport,
so a viewer continuing an api_server row is still answered on that transport (out of TUI-8's scope).
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging

from gateway.config import Platform
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

# The viewer admission executing in this context, else None.
viewer_turn = ContextVar('viewer_turn', default=None)
# ``viewer_delivery_adapter``'s answer for egress that is not the viewer turn's own chat.
NOT_A_VIEWER_TURN = object()
REFUSED_TEXT = ('Not run: this conversation is no longer its chat\'s current session (it moved on with /new) '
                'before this message started. The message was refused; nothing is left queued.')


@dataclass
class _ViewerTurn:
    authority: object
    session_id: str
    notified: bool = False


def viewer_replies_stay_local(authority):
    """``gateway.session.viewer_replies_stay_local`` from the owning profile's config.yaml, read raw like
    every gateway setting (no DEFAULT_CONFIG merge): absent, or anything but ``true``, is upstream.

    Reads the loader ``gateway.run._load_gateway_config`` wraps, not that facade: tests stub the facade
    with zero-argument callables (``tests/gateway/conftest.py``, ``test_session_operator_scope.py``),
    which the per-profile path this read needs under multiplex would break. Bypassing the facade also
    means those stubs do not reach this read: it always sees the profile's real config.yaml.
    Fail-open like the facade."""
    from pathlib import Path
    from hermes_cli.config import cfg_get
    from hermes_cli.config_effective import load_user_config_effective
    home = Path(str(authority.profile_id))
    # Under multiplex the authority's profile id is its home; symbolic ids (tests) read the active one.
    path = home / 'config.yaml' if home.is_absolute() and home.is_dir() else None
    try:
        config = load_user_config_effective(path)
    except Exception:
        logger.debug('Could not load gateway config from %s', path, exc_info=True)
        config = {}
    return cfg_get(config, 'gateway', 'session', 'viewer_replies_stay_local', default=False) is True


def _messaging(source):
    platform = getattr(source, 'platform', None)
    return isinstance(platform, Platform) and platform not in (Platform.LOCAL, Platform.API_SERVER)


@contextmanager
def viewer_turn_scope(authority, ref, source, payload):
    """Mark the execution of a viewer's admission on a messaging session whose replies stay local
    (see the module doc)."""
    viewer = ('local_operator_v1' in payload and _messaging(source)
              and viewer_replies_stay_local(authority))
    token = viewer_turn.set(_ViewerTurn(authority, ref.session_id) if viewer else None)
    try:
        yield
    finally:
        viewer_turn.reset(token)


def viewer_delivery_adapter(source):
    """The adapter a viewer turn's egress to its own chat answers through, ``NOT_A_VIEWER_TURN`` for
    any other egress. None (deliver nothing) rather than a platform bot when the local slot is held by
    anything else; the viewers are told once instead of waiting on a reply that never comes."""
    turn = viewer_turn.get()
    live = turn.authority.sessions.get(turn.session_id) if turn is not None else None
    if (live is None or getattr(source, 'platform', None) != live.source.platform
            or str(source.chat_id) != str(live.source.chat_id)):
        return NOT_A_VIEWER_TURN
    from gateway.session_local_recovery import local_session_adapter
    adapter = local_session_adapter(turn.authority)
    if adapter is None and not turn.notified:
        turn.notified = True
        live.event_stream.publish(turn.session_id, {
            'text': 'This turn has no local delivery channel; its replies cannot be shown here.',
            'level': 'warning', 'kind': 'sticky', 'key': 'viewer-delivery:' + turn.session_id},
            event_type='notification.show')
    return adapter


def lane_state(authority, ref):
    """Which session *ref*'s messaging lane runs its next turn in, answered from the routing transition
    ``SessionStore._get_or_create_session_impl`` will run (``get_or_create_session`` keys by source, so
    a viewer turn lands wherever that transition resolves). Sessions compare by ``logical_owner``
    (compression continuations copy ``session_key``, so a raw id would split one lineage):

    * ``'moved'``: the next turn certainly lands elsewhere:

      1. the route names another logical owner (``/new``, a reset, ``/resume`` of another row). An entry
         on a hard-deleted row names no owner: it is a stale hit like 3., and recovery decides;
      2. the route is suspended and no compression heal rewrites it: ``_route_reset_reason``
         (``session_lifecycle.py:92-94``) answers ``'suspended'``, so ``_apply_route_checks`` pops the
         entry and schedules a reset (``session.py:1012-1017``) and a fresh session is minted, never
         recovered (``session.py:960``);
      3. the routed row ended and recovery will not reopen it: ``_is_session_ended_in_db`` (row gone or
         ``end_reason`` set) on an entry no compression heal rewrites is a stale hit (``session.py:1001``);
         the entry is popped (``session.py:1016``) and recovery (``peek_recovery``) continues only a row
         its peer finder admits (``_RECOVERABLE_END_REASONS`` or a completed handoff, newest per key),
         so a row ended by a deliberate boundary, or older than the finder's pick, is never the one;
    * ``'current'``: the route names *ref* and none of the above holds. The heal is judged first, as
      ``_apply_route_checks`` does (``session.py:997-1002``): an entry left on a compressed parent is
      rewritten to its tip, which voids the suspension and stale checks taken on the parent, so the
      turn runs in the tip (same logical owner) even on a suspended route;
    * no entry at all (pruned, or a row ``session_route_recovery`` mounted from itself): recovery decides,
      exactly as for the platform's own next message. By default that is ``'current'``, upstream's
      admission (#134889). With ``viewer_replies_stay_local`` it is a dry run of that recovery
      (``peek_recovery``, ``session_recovery.py`` ``_query_recoverable_row``): ``'current'`` only when
      it continues *ref*'s logical owner, ``'moved'`` when it picks another row or mints one;
    * ``'unknown'``: the routing index or the store owning the route cannot be read (no session store,
      an index that fails to load, a state.db replaced underneath it: any error), recovery could not be
      told without side effects, or its lookup failed (the real transition swallows that failure and
      mints a session; never adopt on it). Both causes share ``REFUSED_TEXT`` when the switch refuses,
      whose mention of ``/new`` is imprecise for an unreadable lane;
    * None: not a messaging session (local and API sessions route by their own receipts).

    ``viewer_turn_refused`` is the one verdict the submission gate, the pre-claim refusal and viewer
    adoption read. It never raises.

    Known residual race (shared with upstream, not reproduced): nothing fences ``refuse_moved_head``'s
    pre-claim check (``session_authority.py``) from the turn resolving its session
    (``run_turn_prepare.py``), so a ``/new`` landing in between still runs the turn in the successor."""
    try:
        live = authority.sessions[ref.session_id]
        if not _messaging(live.source):
            return None
        store = authority.runner.session_store
        with store._lock:
            entry = store._entry_locked(live.route)
        owner = authority.logical_owner(ref.session_id)
        if entry is None:
            if store._db_for_key(live.route) is None:
                return 'unknown'
            if not viewer_replies_stay_local(authority):
                return 'current'
            return _recovery_state(authority, store, live, owner)
        if _row_exists(store, live.route, entry.session_id):
            if authority.logical_owner(entry.session_id) != owner:
                return 'moved'
            if store._compression_tip_for_session_id(entry.session_id) not in (None, entry.session_id):
                return 'current'
        if entry.suspended:
            return 'moved'
        if not store._is_session_ended_in_db(entry.session_id, session_key=live.route):
            return 'current'
        return _recovery_state(authority, store, live, owner)
    except Exception:
        # No store / a store without the index API (bare runners), an index that fails to load, a
        # KeyError for a session no longer mounted, or any other read failure: this verdict fails
        # closed rather than raising. A state.db replaced underneath the process surfaces on the
        # write path (StateDbReplacedError / DeletedWalGenerationError, RuntimeError subclasses);
        # reads against the replaced file answer from it or raise sqlite3.Error, covered here too.
        logger.warning('Lane of %s could not be resolved', ref.session_id, exc_info=True)
        return 'unknown'


def _row_exists(store, route, session_id):
    """False only when the route's own store answers that the row is gone (hard-deleted): its id then
    names no lineage, and upstream treats the entry as stale (``_is_session_ended_in_db``), so
    recovery decides.

    A route whose owning store cannot be read (``_db_for_key`` -> None) keeps the entry authoritative
    — fail-open, matching ``_is_session_ended_in_db``'s own ``db is None`` answer and routing's use of
    the entry's id. The sibling branch (no entry at all) answers ``'unknown'`` instead, because there
    the entry carries no id to trust; with the switch off both admit (upstream behaviour), with it on
    the no-entry branch refuses visibly."""
    db = store._db_for_key(route)
    return db is None or db.get_session(session_id) is not None


def _recovery_state(authority, store, live, owner):
    # peek_recovery's lookup errors raise by its own contract (session_recovery.py); any of them,
    # plus any other read failure, becomes lane_state's 'unknown'.
    known, recovered = store.peek_recovery(live.route, live.source)
    if not known:
        return 'unknown'
    return 'current' if recovered and authority.logical_owner(recovered) == owner else 'moved'


def viewer_turn_refused(authority, ref):
    """A viewer turn the lane would run in another session; with ``viewer_replies_stay_local`` also one
    whose lane cannot be resolved (fail closed). Upstream admits ``'unknown'`` as it always did."""
    state = lane_state(authority, ref)
    return state == 'moved' or (state == 'unknown' and viewer_replies_stay_local(authority))


def require_current_lane(authority, ref):
    """Refuse a viewer turn ``viewer_turn_refused`` names: never continue it elsewhere."""
    if viewer_turn_refused(authority, ref):
        raise RuntimeStoreError('session_not_current')


async def refuse_moved_head(authority, ref, row):
    """Settle a queued viewer turn whose lane moved on between admission and claim as a visible
    terminal refusal (never left queued behind a paused FIFO); False when it may run."""
    if not viewer_turn_refused(authority, ref):
        return False
    import asyncio
    from hermes_state_runtime import cancel_session_input
    settled = await asyncio.to_thread(cancel_session_input, authority.db, epoch=authority.epoch,
                                      admission_id=row['admission_id'], outcome='refused')
    if settled['status'] == 'terminal':
        authority._settle_cancelled(ref, row['admission_id'], outcome='refused', text=REFUSED_TEXT)
    return True
