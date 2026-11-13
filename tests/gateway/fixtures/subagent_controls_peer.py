"""Authenticated owner fixture: Desktop's per-child Steer/Stop reach only this session's children."""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback
import weakref


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _close_subagent_steering, _unregister_subagent
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='subagent-controls')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {}

    class Agent:
        def __init__(self):
            self.steered, self.stopped = [], []

        def steer(self, text):
            self.steered.append(text)
            return True

        def hard_interrupt(self, message=None):
            self.stopped.append(message)
            return True

    ticket = mint_ticket(user_id='owner', provider='fixture')
    async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
        created = await rpc(ws, 'session.create', request_id='controls-owner', source='gui', toolsets=[])
        sid = created['result']['session_id']
        live = authority.sessions[sid]
        parent, other_parent = Agent(), Agent()
        parent.session_id, other_parent.session_id = sid, 'foreign-session'
        with runner._agent_cache_lock:
            runner._agent_cache[live.route] = (parent, 'fixture')
        children = {}
        try:
            for name, ancestor in [('owned', parent), ('sibling', other_parent)]:
                child = children[name] = Agent()
                child._subagent_id = name
                child._delegate_parent_ref = weakref.ref(ancestor)
                child._parent_session_id = ancestor.session_id
                _register_child(child, ancestor, name + '-goal', owner_session_id=None,
                                owner_transport=None, owner_session_record=None)
            # The exact Desktop producer frames (subagent-controls.tsx).
            for name in ('owned', 'sibling'):
                steer = await rpc(ws, 'subagent.steer', session_id=sid, subagent_id=name, text='check ' + name)
                stop = await rpc(ws, 'subagent.interrupt', session_id=sid, subagent_id=name)
                receipt[name] = {'steer': steer.get('result', steer.get('error')),
                                 'stop': stop.get('result', stop.get('error'))}
            receipt['delivered'] = {name: [child.steered, len(child.stopped)] for name, child in children.items()}
            # A child past its final steer drain refuses instead of dropping the text.
            _close_subagent_steering('owned', children['owned'])
            late = await rpc(ws, 'subagent.steer', session_id=sid, subagent_id='owned', text='too late')
            receipt['late'] = late.get('result', late.get('error'))
        finally:
            for name, child in children.items():
                _unregister_subagent(name, agent=child)
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
