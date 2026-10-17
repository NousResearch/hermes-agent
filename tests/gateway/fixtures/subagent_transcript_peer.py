"""Authenticated owner fixture: a viewer reads a delegated child's stored transcript.

Real gateway runner, session authority, ticketed WebSocket and the profile's own ``state.db``
under a temp HERMES_HOME; no inference. Child rows are written the way a delegated child writes
them (``source='subagent'``, ``_delegate_from`` + ``_subagent_id`` markers, messages as it runs).
Every reply lands in ``receipt.json`` for the test to judge.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback
import weakref


def seed(db, target):
    """The conversation's children as the store holds them; returns the row ids per child."""
    def child(sid, parent, subagent_id, *, source='subagent', **cfg):
        db.create_session(sid, source=source, parent_session_id=parent,
                          model_config={'_delegate_from': parent, '_subagent_id': subagent_id, **cfg})

    # A foreign conversation's child that happens to draw the same subagent_id, stored FIRST.
    db.create_session('foreign-parent', source='cli')
    child('foreign-done', 'foreign-parent', 'sa-1-done')
    child('foreign-child', 'foreign-parent', 'sa-0-foreign')
    child('live-child', target, 'sa-0-live')
    child('done-child', target, 'sa-1-done')
    # The finished child compressed once: the continuation inherits its model_config.
    db.end_session('done-child', 'compression')
    db.create_session('done-child-2', source='subagent', parent_session_id='done-child',
                      model_config={'_delegate_from': target, '_subagent_id': 'sa-1-done'})
    # An orchestrator child's own worker, delegated from the child's first segment.
    child('nested-child', 'done-child', 'sa-0-nested')
    # A branch of this conversation is another conversation: its children are not ours.
    db.create_session('branch', source='gui', parent_session_id=target, model_config={'_branched_from': target})
    child('branch-child', 'branch', 'sa-0-branch')
    rows = {}
    for sid, texts in [('live-child', ['live goal', 'live step one']), ('done-child-2', ['done summary', 'done final']),
                       ('nested-child', ['nested goal', 'nested final']), ('foreign-done', ['FOREIGN_PRIVATE']),
                       ('foreign-child', ['FOREIGN_PRIVATE']), ('branch-child', ['BRANCH_PRIVATE'])]:
        rows[sid] = [db.append_message(sid, role='user' if i == 0 else 'assistant', content=text)
                     for i, text in enumerate(texts)]
    return rows


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_ancillary import physical_target
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_contract import SessionRef
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _unregister_subagent
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='subagent-transcript')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {}

    class Agent:
        pass

    def socket(actor):
        ticket = mint_ticket(user_id=actor, provider='fixture')
        return connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket)

    async def transcript(ws, sid, subagent_id, **extra):
        reply = await rpc(ws, 'subagent.transcript', session_id=sid, subagent_id=subagent_id, **extra)
        return reply.get('result', reply.get('error'))

    try:
        async with socket('owner') as ws, socket('peer_operator') as peer:
            created = await rpc(ws, 'session.create', request_id='transcript-owner', source='gui', toolsets=[])
            sid = created['result']['session_id']
            live = authority.sessions[sid]
            target = physical_target(authority, SessionRef(str(home), sid))
            receipt['rows'] = seed(authority.db, target)
            parent = Agent()
            parent.session_id = target
            with runner._agent_cache_lock:
                runner._agent_cache[live.route] = (parent, 'fixture')
            running = Agent()
            running._subagent_id, running.session_id = 'sa-0-live', 'live-child'
            running._delegate_parent_ref = weakref.ref(parent)
            running._parent_session_id = target
            _register_child(running, parent, 'live goal', owner_session_id=None,
                            owner_transport=None, owner_session_record=None)
            try:
                roster = await rpc(ws, 'subagent.list', session_id=sid)
                receipt['list'] = roster.get('result', roster.get('error'))
                first = receipt['live'] = await transcript(ws, sid, 'sa-0-live')
                receipt['peer_live'] = await transcript(peer, sid, 'sa-0-live')
                # The child writes on while the viewer polls from the cursor it was handed.
                receipt['rows']['live-child'].append(authority.db.append_message(
                    'live-child', role='assistant', content='live step two'))
                receipt['live_after'] = await transcript(ws, sid, 'sa-0-live', after=first['cursor'])
                receipt['live_caught_up'] = await transcript(ws, sid, 'sa-0-live',
                                                             after=receipt['live_after']['cursor'])
            finally:
                _unregister_subagent('sa-0-live', agent=running)
            # Ended: gone from the registry, still readable from the store.
            receipt['live_ended'] = await transcript(ws, sid, 'sa-0-live')
            for name in ('sa-1-done', 'sa-0-nested', 'sa-0-foreign', 'sa-0-branch', 'sa-9-unknown'):
                receipt[name] = await transcript(ws, sid, name)
            receipt['unknown_session'] = await transcript(ws, 'no-such-session', 'sa-1-done')
            receipt['foreign_profile'] = await transcript(ws, sid, 'sa-1-done', profile='foreign')
            receipt['bad_cursor'] = await transcript(ws, sid, 'sa-1-done', after=-1)
            described = await rpc(ws, 'runtime.describe')
            receipt['capabilities'] = described['result']['capabilities']
    finally:
        await stop_gateway_api(api)
    (home / 'receipt.json').write_text(json.dumps(receipt))


if __name__ == '__main__':
    status = 0
    try:
        asyncio.run(probe())
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
