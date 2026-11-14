"""Sessionless Desktop setup; legacy chat metadata is retired without touching history."""
import asyncio
from functools import partial

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {'onboarding.state': partial(state, connection),
            'onboarding.set_run': partial(set_run, connection)}


def _authorize(connection):
    if 'session:operator' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')


async def _retire(connection):
    from hermes_cli.onboarding_run import retire_conversational_runs
    from gateway.session_runtime_workers import tracked_write
    from gateway.session_server_requests import withdraw
    target = connection._sibling_for('default') or connection
    # None means default IS this authority; an unserved/foreign home raises, never falls back.
    authority = target.authority
    lock = getattr(authority, '_onboarding_start_lock', None)
    if lock is None:
        lock = authority._onboarding_start_lock = asyncio.Lock()
    async with lock:
        if getattr(authority, '_sessionless_setup_migrated', False):
            return
        retired = await tracked_write(authority, partial(retire_conversational_runs, authority.db))
        for sid in retired:
            live = authority.sessions.get(sid)
            if live is not None:
                withdraw(live.event_stream, sid, 'setup_beat')
        authority._sessionless_setup_migrated = True


def _state(connection, assume_fresh=False):
    from hermes_cli.onboarding_run import guide_progress, should_run
    return {'run': should_run(assume_fresh=assume_fresh), **guide_progress(),
            'eligible': bool(getattr(connection.transport, 'desktop_local', False))}


async def state(connection, ref, params):
    _authorize(connection)
    if not params.get('sessionless'):
        from tui_gateway.server import _run_state
        return await asyncio.to_thread(_run_state, assume_fresh=bool(params.get('assume_fresh')))
    await _retire(connection)
    return await asyncio.to_thread(_state, connection, bool(params.get('assume_fresh')))


async def set_run(connection, ref, params):
    _authorize(connection)
    from hermes_cli.onboarding_run import set_run as persist
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import OnboardingSetRunParams
    from pydantic import ValidationError
    try:
        request = OnboardingSetRunParams.model_validate(params)
    except ValidationError as exc:
        raise RuntimeStoreError('invalid_params', detail=str(exc)) from exc
    if not request.sessionless:
        from tui_gateway.server import _run_state
        await asyncio.to_thread(persist, request.run, mark_profile_offered=request.mark_profile_offered)
        return await asyncio.to_thread(_run_state)
    await asyncio.to_thread(persist, request.run, step=request.step, restart=request.restart,
                            mark_profile_offered=request.mark_profile_offered, sessionless=True)
    return await asyncio.to_thread(_state, connection)
