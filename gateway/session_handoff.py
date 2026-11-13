"""Shared-owner handoff verbs (Desktop /handoff, CLI /handoff).

Port of tui_gateway/methods_session.py's handoff.request/state/fail onto the
owner's session surface, so the same verbs work on a shared gateway instead
of being refused by name.
"""
from __future__ import annotations

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {
        'handoff.request': lambda ref, params: request(connection, ref, params),
        'handoff.state': lambda ref, params: state(connection, ref, params),
        'handoff.fail': lambda ref, params: fail(connection, ref, params),
    }


async def request(connection, ref, params):
    authority = connection.authority
    if not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    platform_name = str(params.get('platform') or '').strip().lower()
    if not platform_name:
        raise RuntimeStoreError('invalid_params', field='platform', detail='platform required')

    live = authority.sessions.get(ref.session_id)
    if live is not None and live.task is not None and not live.task.done():
        raise RuntimeStoreError('session_busy', detail='session busy — wait for the current turn to finish, then retry the handoff')

    from gateway.config import Platform, load_gateway_config
    from gateway.session_authorities import owner_scope
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        raise RuntimeStoreError('invalid_params', field='platform', detail=f"unknown platform '{platform_name}'")
    try:
        with owner_scope(authority):
            gw_config = load_gateway_config()
    except Exception as exc:
        raise RuntimeStoreError('storage_unavailable', detail=f'could not load gateway config: {exc}')
    pcfg = gw_config.platforms.get(platform)
    if not getattr(pcfg, 'enabled', False):
        raise RuntimeStoreError('invalid_params', field='platform', detail=f"platform '{platform_name}' is not configured/enabled in the gateway")
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        raise RuntimeStoreError('invalid_params', field='platform',
                               detail=f"no home channel configured for {platform_name} — set one with /sethome on the destination chat first")

    db = authority.db
    if db is None:
        raise RuntimeStoreError('storage_unavailable')
    row = db.get_session(ref.session_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    try:
        if not db.get_session(ref.session_id):
            db.set_session_title(ref.session_id, f'handoff-{ref.session_id[:8]}')
        if not db.request_handoff(ref.session_id, platform_name):
            raise RuntimeStoreError('already_in_flight', detail='session is already in flight for handoff — wait for it to settle, then retry')
    except RuntimeStoreError:
        raise
    except Exception as exc:
        raise RuntimeStoreError('storage_unavailable', detail=str(exc))
    return {'queued': True, 'session_key': ref.session_id, 'platform': platform_name, 'home_name': home.name}


async def state(connection, ref, params):
    db = connection.authority.db
    if db is None:
        raise RuntimeStoreError('storage_unavailable')
    record = db.get_handoff_state(ref.session_id) or {}
    return {field: (record.get(field) or '') for field in ('state', 'platform', 'error')}


async def fail(connection, ref, params):
    db = connection.authority.db
    if db is None:
        raise RuntimeStoreError('storage_unavailable')
    reason = str(params.get('error') or 'handoff failed').strip()[:500]
    try:
        failed = db.fail_handoff(ref.session_id, reason, only_states=('pending',))
    except TypeError:
        failed = ((db.get_handoff_state(ref.session_id) or {}).get('state') or '') == 'pending'
        if failed:
            db.fail_handoff(ref.session_id, reason)
    state_val = 'failed' if failed else ((db.get_handoff_state(ref.session_id) or {}).get('state') or '')
    return {'failed': bool(failed), 'state': state_val}
