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
from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {'handoff.request': partial(request, connection), 'handoff.state': partial(state, connection),
            'handoff.fail': partial(fail, connection)}


def _session(connection, ref, params, fields, capability):
    if set(params) - fields or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, capability)


def _destination(platform_name):
    """The home channel the watcher will deliver to, by the CLI's own rule (relay-fronted platforms
    included): an unconfigured platform or a missing home channel would otherwise sit ``pending``
    until the client gives up."""
    from gateway.handoff_target import HandoffTargetError, resolve_handoff_home
    try:
        return resolve_handoff_home(platform_name)
    except HandoffTargetError as exc:
        raise RuntimeStoreError('invalid_params', field='platform', detail=_REFUSALS[exc.reason].format(
            platform=platform_name)) from None


_REFUSALS = {'unknown_platform': "unknown platform '{platform}'",
             'platform_not_enabled': "platform '{platform}' is not configured/enabled in the gateway",
             'no_home_channel': 'no home channel configured for {platform}; set one with /sethome on the '
                                'destination chat first'}


async def request(connection, ref, params):
    _session(connection, ref, params, {'session_id', 'profile', 'platform'}, 'session:control')
    platform_name = params.get('platform')
    if not isinstance(platform_name, str) or not platform_name.strip():
        raise RuntimeStoreError('invalid_params', field='platform', detail='platform required')
    platform_name = platform_name.strip().lower()
    live = connection.authority.sessions[ref.session_id]
    if live.task is not None and not live.task.done():
        # The watcher's repoint and synthetic turn would race the running one.
        raise RuntimeStoreError('session_busy', detail='wait for the current turn to finish, then retry the handoff')
    db = connection.authority.db

    def write():
        home = _destination(platform_name)
        if not db.request_handoff(ref.session_id, platform_name):
            raise RuntimeStoreError('session_busy', field='handoff',
                                    detail='a handoff of this session is already in flight; wait for it to settle')
        return home
    home = await asyncio.to_thread(write)
    return {'queued': True, 'session_key': ref.session_id, 'platform': platform_name, 'home_name': home.name}


async def state(connection, ref, params):
    _session(connection, ref, params, {'session_id', 'profile'}, 'session:read')
    record = await asyncio.to_thread(connection.authority.db.get_handoff_state, ref.session_id) or {}
    return {field: record.get(field) or '' for field in ('state', 'platform', 'error')}


async def fail(connection, ref, params):
    """A client's poll timeout: fails a handoff no watcher claimed yet. A claimed (``running``) row is
    the watcher's to settle, so the write is a compare-and-swap on ``pending``."""
    _session(connection, ref, params, {'session_id', 'profile', 'error'}, 'session:control')
    reason = params.get('error')
    reason = (reason.strip() if isinstance(reason, str) else '')[:500] or 'handoff failed'
    db = connection.authority.db

    def write():
        failed = db.fail_handoff(ref.session_id, reason, only_states=('pending',))
        return failed, (db.get_handoff_state(ref.session_id) or {}).get('state') or ''
    failed, current = await asyncio.to_thread(write)
    return {'failed': bool(failed), 'state': 'failed' if failed else current}
