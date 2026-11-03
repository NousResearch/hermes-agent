"""``session.detach {end_reason}``: the last viewer leaving an idle local session ends its row with the
reason main's classic CLI wrote itself (``cli_close`` on quit, ``new_session`` / ``resumed_other`` /
``branched`` on a switch). Real authority + SQLite."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_detach_with_end_reason_ends_the_idle_local_row(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    db = authority.db

    def viewer():
        return AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})

    try:
        first, second = viewer(), viewer()
        refs = {}
        for name in ('quit', 'shared', 'plain'):
            ref = create_local_session(authority, first.actor, {'request_id': name, 'source': 'cli',
                                                                'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
            refs[name] = ref, local_receipt(db, ref.session_id)['entry']['session_id']
        subs = {}
        for name, (ref, _) in refs.items():
            reply = await first.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': ref.session_id}})
            subs[name] = reply['result']['subscription_id']
        await second.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': refs['shared'][0].session_id}})

        async def detach(name, **extra):
            return await first.dispatch({'id': 2, 'method': 'session.detach', 'params': {
                'session_id': refs[name][0].session_id, 'subscription_id': subs[name], **extra}})

        def ended(name):
            row = db.get_session(refs[name][1])
            return row['ended_at'] is not None, row['end_reason']

        reply = await detach('quit', end_reason='cli_close')
        assert reply['result']['detached'] is True, reply
        assert ended('quit') == (True, 'cli_close')
        await detach('shared', end_reason='new_session')
        assert ended('shared') == (False, None), 'another viewer still holds the session'
        await detach('plain')
        assert ended('plain') == (False, None), 'a plain detach is not an end'
        bad = await first.dispatch({'id': 3, 'method': 'session.detach', 'params': {
            'session_id': refs['plain'][0].session_id, 'subscription_id': 'x', 'end_reason': 7}})
        assert bad['error']['message'] == 'invalid_params'
    finally:
        store.close_all_db_handles()
