"""ACP v0.9 has no per-session destroy: the last viewer leaving an idle ACP session is its end (#118216).

Real authority + SQLite. Without an ``ended_at`` writer on the gateway path, source='acp' rows stay
open forever and the ended-session guard keeps prune/archive away from them.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_last_viewer_or_shutdown_ends_only_idle_acp_sessions(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run_runtime import settle_gateway_runtime
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': [], 'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    db = authority.db

    def viewer():
        return AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})

    owner = viewer().actor

    def create(source, request_id):
        ref = create_local_session(authority, owner, {'request_id': request_id, 'source': source,
                                                      'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        return ref, local_receipt(db, ref.session_id)['entry']['session_id']

    def ended(target):
        row = db.get_session(target)
        return row['ended_at'] is not None, row['end_reason']

    async def resume(connection, ref):
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})
        assert 'result' in reply, reply

    try:
        acp, acp_row = create('acp', 'editor')
        gui, gui_row = create('gui', 'desktop')
        busy, busy_row = create('acp', 'busy-editor')
        unseen, unseen_row = create('acp', 'unseen-editor')
        first, second = viewer(), viewer()
        for connection in (first, second):
            await resume(connection, acp)
            await resume(connection, gui)
        await resume(first, busy)
        authority._schedule = lambda ref: None
        await authority.submit(first.actor, Submission('queued', busy, {'text': 'later'}, 'queue'))

        await first.close()
        assert ended(acp_row) == (False, None), 'another viewer still holds the ACP session'
        assert ended(busy_row) == (False, None), 'a queued admission keeps the ACP session open'
        await second.close()
        assert ended(acp_row) == (True, 'acp_disconnect')
        assert ended(gui_row) == (False, None), 'only ACP sessions end on disconnect'
        assert ended(busy_row) == (False, None)

        # Gateway shutdown ends the ACP sessions nobody is viewing (never attached this run included).
        await settle_gateway_runtime(runner)
        assert ended(unseen_row) == (True, 'acp_disconnect')
        assert ended(busy_row) == (False, None) and ended(gui_row) == (False, None)
    finally:
        store.close_all_db_handles()
