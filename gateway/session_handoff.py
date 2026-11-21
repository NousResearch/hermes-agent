"""``handoff.request`` / ``handoff.state`` / ``handoff.fail`` on the shared owner (Desktop and CLI
``/handoff``), as the per-TUI sidecar serves them (``tui_gateway/methods_session.py``).

The owner only writes the session row's ``pending`` handoff; the gateway's handoff watcher claims it,
repoints the platform's home route at the session and runs the synthetic turn there. ``dispatch``
runs each handler under the owning profile's scope, so the gateway config read here (platform and
home channel) is that profile's, the one the watcher delivers through. SQLite runs off the owner loop.
"""
import asyncio
from functools import partial

from gateway.session_busy_controls import authorize
from gateway.session_persistence import authority_persistence
from persistence.authority.coordination import HandoffStatus
from persistence.core.errors import AuthorityRefusal as RuntimeStoreError, HandoffUnsettled


def handlers(connection):
    return {'handoff.request': partial(request, connection), 'handoff.state': partial(state, connection),
            'handoff.fail': partial(fail, connection)}


def _session(connection, ref, params, fields, capability):
    if set(params) - fields or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, capability)


class HandoffRefusal(RuntimeStoreError):
    """A refusal whose wire ``message`` is the human reason, as the sidecar's handoff errors were:
    Desktop shows ``error.message`` verbatim, and a bare ``invalid_params`` never says what to fix
    (``data`` still carries the reason code, field and detail)."""

    def __init__(self, reason, detail, field=None):
        super().__init__(reason, field=field, detail=detail)
        self.wire_message = self.detail


def _destination(platform_name):
    """The home channel the watcher will deliver to, by the CLI's own rule (relay-fronted platforms
    included): an unconfigured platform or a missing home channel would otherwise sit ``pending``
    until the client gives up."""
    from gateway.handoff_target import HandoffTargetError, resolve_handoff_home
    try:
        return resolve_handoff_home(platform_name)
    except HandoffTargetError as exc:
        raise HandoffRefusal('invalid_params', _REFUSALS[exc.reason].format(platform=platform_name),
                       field='platform') from None


_REFUSALS = {'unknown_platform': "unknown platform '{platform}'",
             'platform_not_enabled': "platform '{platform}' is not configured/enabled in the gateway",
             'no_home_channel': 'no home channel configured for {platform}; set one with /sethome on the '
                                'destination chat first'}


_UNSETTLED = {'session_busy': 'wait for the current turn to finish, then retry the handoff',
              'unknown_execution': 'a turn of this session did not finish when Hermes restarted; send /discard '
                                   'in it (or resume it) first, then retry the handoff'}


def _unsettled(exc):
    """The human refusal for :class:`HandoffUnsettled`: durable work still owns the session (any
    queued, started or unknown admission, a live worker, or a live turn lease on its runtime lineage).
    A recovered ``unknown`` turn has no resident drain task, yet moving its route would hand the lane
    a turn whose outcome nobody resolved. The store checks it in its own write transaction: inside
    ``request_handoff``'s, and again when the watcher repoints (``check_settled``), so a turn admitted
    after the request still wins."""
    return HandoffRefusal(exc.reason, _UNSETTLED[exc.reason])


async def check_settled(store, session_id):
    """The watcher's re-check of the settled condition right before it repoints the route; a refusal
    fails the handoff with the human reason (the CLI and Desktop show ``handoff_error``)."""
    if store is None:
        return
    try:
        await store.aio.coordination.check_handoff_settled(session_id=session_id)
    except HandoffUnsettled as exc:
        raise RuntimeError(_unsettled(exc).detail) from None


async def request(connection, ref, params):
    _session(connection, ref, params, {'session_id', 'profile', 'platform'}, 'session:control')
    platform_name = params.get('platform')
    if not isinstance(platform_name, str) or not platform_name.strip():
        raise HandoffRefusal('invalid_params', 'platform required', field='platform')
    platform_name = platform_name.strip().lower()
    live = connection.authority.sessions[ref.session_id]
    if live.task is not None and not live.task.done():
        # The watcher's repoint and synthetic turn would race the running one.
        raise HandoffRefusal('session_busy', _UNSETTLED['session_busy'])
    coordination = authority_persistence(connection.authority).coordination

    def write():
        home = _destination(platform_name)
        try:
            queued = coordination.request_handoff(session_id=ref.session_id, platform=platform_name, require_settled=True)
        except HandoffUnsettled as exc:
            raise _unsettled(exc) from None
        if not queued:
            raise HandoffRefusal('session_busy', 'a handoff of this session is already in flight; wait for it to settle',
                           field='handoff')
        return home
    home = await asyncio.to_thread(write)
    return {'queued': True, 'session_key': ref.session_id, 'platform': platform_name, 'home_name': home.name}


async def state(connection, ref, params):
    _session(connection, ref, params, {'session_id', 'profile'}, 'session:read')
    coordination = authority_persistence(connection.authority).coordination
    record = await asyncio.to_thread(coordination.get_handoff, session_id=ref.session_id)
    return {field: str(getattr(record, field, None) or '') for field in ('state', 'platform', 'error')}


async def fail(connection, ref, params):
    """A client's poll timeout: fails a handoff no watcher claimed yet. A claimed (``running``) row is
    the watcher's to settle, so the write is a compare-and-swap on ``pending``."""
    _session(connection, ref, params, {'session_id', 'profile', 'error'}, 'session:control')
    reason = params.get('error')
    reason = (reason.strip() if isinstance(reason, str) else '')[:500] or 'handoff failed'
    coordination = authority_persistence(connection.authority).coordination

    def write():
        failed = coordination.fail_handoff(session_id=ref.session_id, error=reason, only_states=(HandoffStatus.PENDING,))
        record = coordination.get_handoff(session_id=ref.session_id)
        return failed, str(record.state) if record is not None else ''
    failed, current = await asyncio.to_thread(write)
    return {'failed': bool(failed), 'state': 'failed' if failed else current}
