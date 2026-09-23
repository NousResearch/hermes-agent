"""Authenticated TCP fixture with owner-seeded state; no external inference."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
import weakref


async def probe(mode):
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_contract import SessionRef
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='ancillary')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True

    def socket(actor='owner'):
        ticket = mint_ticket(user_id=actor, provider='fixture')
        return connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket)

    try:
        async with socket() as ws, socket('peer_operator') as peer_operator:
            created = await rpc(ws, 'session.create', request_id='read-owner', source='gui', toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            ref = SessionRef(str(home), sid)
            live = authority.sessions[sid]
            methods = ['session.control.read', 'process.list', 'subagent.list', 'subagent.tail']
            for method in methods:
                args = {'subagent_id': 'unowned'} if method == 'subagent.tail' else {}
                shared = await rpc(peer_operator, method, session_id=sid, **args)
                original = await rpc(ws, method, session_id=sid, **args)
                # Both tickets are verified dashboard operators; underlying child
                # membership checks still apply, including the absent tail target.
                assert shared.get('result') == original.get('result'), (shared, original)
                assert shared.get('error') == original.get('error'), (shared, original)
                assert shared.get('error', {}).get('message') != 'permission_denied', shared
                denied = await rpc(ws, method, session_id=sid, profile='foreign', **args)
                assert denied.get('error', {}).get('message') == 'profile_mismatch', denied

            if mode == 'control':
                from hermes_cli.goals import GoalState, GoalGate
                from hermes_cli.loops import LoopState
                from hermes_cli.heartbeat import HeartbeatState
                # Owner writes fixture rows; reads must not call managers that clear barriers.
                goal = GoalState('OWNER_GOAL', status='paused', created_at=10, waiting_on_pid=99999999,
                                 gates=[GoalGate(command='owned command', last_output_tail='PRIVATE_GATE_OUTPUT')])
                loop = LoopState('OWNER_LOOP', status='paused', created_at=20, route={'token': 'PRIVATE_ROUTE'})
                heartbeat = HeartbeatState('OWNER_HEARTBEAT', 600, status='paused', created_at=30)
                for key, state in [('goal', goal), ('loop', loop), ('heartbeat', heartbeat)]:
                    authority.db.set_meta(key + ':' + sid, state.to_json())
                    authority.db.set_meta(key + ':' + live.route, state.to_json().replace('OWNER_', 'WRONG_ROUTE_'))
                before = authority.db._conn.total_changes
                responses = [await rpc(ws, 'session.control.read', session_id=sid) for _ in range(2)]
                assert 'result' in responses[0], responses
                snap = responses[0]['result']['control']
                assert responses[0] == responses[1]
                assert snap['goal']['title'] == 'OWNER_GOAL' and snap['loop']['prompt'] == 'OWNER_LOOP'
                assert snap['heartbeat']['prompt'] == 'OWNER_HEARTBEAT' and snap['updated_at'] == 30
                assert snap['goal']['wait_barrier']['target'] == 99999999
                assert 'PRIVATE_' not in json.dumps(snap) and len(snap['revision']) == 64
                assert authority.db._conn.total_changes == before
                for method, expected in [('process.list', {'processes': []}),
                                         ('subagent.list', {'subagents': [], 'delegations': []})]:
                    result = await rpc(ws, method, session_id=sid)
                    assert result.get('result') == expected, result
                assert authority.agent(ref) is None
                assert not authority.db.get_messages(sid)
                authority.sessions.pop(sid)
                result = await rpc(ws, 'session.control.read', session_id=sid)
                assert result.get('error', {}).get('message') == 'not_found', result
                assert sid not in authority.sessions, 'read restored a cold session'
                authority.sessions[sid] = live
            else:
                from tools.process_registry import process_registry
                from tools.delegate_tool_child_run import _register_child
                from tools.delegate_tool_registry import _unregister_subagent
                class Agent:
                    pass
                parent, old_parent, other_parent = Agent(), Agent(), Agent()
                parent.session_id = old_parent.session_id = sid
                other_parent.session_id = 'foreign-session'
                with runner._agent_cache_lock:
                    runner._agent_cache[live.route] = (parent, 'fixture')
                children = []
                processes = []
                gate = home / 'release-process'
                try:
                    for name, ancestor in [('owned', parent), ('stale', old_parent), ('sibling', other_parent)]:
                        child = Agent()
                        child._subagent_id = name
                        child._delegate_parent_ref = weakref.ref(ancestor)
                        child._parent_session_id = ancestor.session_id
                        child._live_transcript_path = home / (name + '.txt')
                        child._live_transcript_path.write_text(name.upper() + '_PRIVATE\n' + 'x' * 18000 + '\n' + name.upper() + '_TAIL')
                        _register_child(child, ancestor, name + '-goal', owner_session_id=None,
                                        owner_transport=None, owner_session_record=None)
                        children.append(child)
                    command = (shlex.quote(sys.executable) + ' -c ' + shlex.quote(
                        "import pathlib,time;print('PROCESS_OWNED',flush=True);p=pathlib.Path(" + repr(str(gate)) + ");\nwhile not p.exists(): time.sleep(.02)"))
                    for task, route in [(sid, live.route), ('foreign', live.route), (sid, 'wrong-route')]:
                        processes.append(process_registry.spawn_local(command, cwd=str(home), task_id=task,
                                                                      owner_task_id=task, session_key=route))
                    async with asyncio.timeout(10):
                        while any('PROCESS_OWNED' not in p.output_buffer for p in processes):
                            await asyncio.sleep(.02)
                    process_result = await rpc(ws, 'process.list', session_id=sid)
                    rows = process_result.get('result', {}).get('processes')
                    assert rows and [p['session_id'] for p in rows] == [processes[0].id], process_result
                    assert rows[0]['status'] == 'running' and 'PROCESS_OWNED' in rows[0]['output_tail']
                    # Real process handoff stores the parent physical ID as its routing key.
                    transferred = process_registry.transfer_ownership(
                        processes[1].id, from_owner='foreign', to_owner=sid, to_task_id=sid, to_session_key=sid)
                    assert transferred is processes[1]
                    handed = await rpc(ws, 'process.list', session_id=sid)
                    assert {p['session_id'] for p in handed['result']['processes']} == {p.id for p in processes[:2]}, handed
                    roster = await rpc(ws, 'subagent.list', session_id=sid)
                    assert [r['subagent_id'] for r in roster.get('result', {}).get('subagents', [])] == ['owned'], roster
                    assert 'agent' not in roster['result']['subagents'][0]
                    for child in children:
                        tail = await rpc(ws, 'subagent.tail', session_id=sid, subagent_id=child._subagent_id)
                        data = tail['result']
                        assert data['available'] == (child._subagent_id == 'owned'), tail
                        if data['available']:
                            assert data['text'].endswith('OWNED_TAIL') and data['truncated']
                            assert len(data['text'].encode()) <= 16384
                        else:
                            assert data['text'] == ''
                    # A route can be reused while an old LiveSession remains attached.
                    with runner.session_store._lock:
                        entry = runner.session_store._entries[live.route]
                        entry.session_id = 'replacement-conversation'
                    rejected = await rpc(ws, 'subagent.list', session_id=sid)
                    assert rejected.get('error', {}).get('message') == 'stale_generation', rejected
                    with runner.session_store._lock:
                        entry.session_id = sid
                    # Identical durable session IDs cannot transfer old-object read authority.
                    with runner._agent_cache_lock:
                        runner._agent_cache[live.route] = (Agent(), 'new-generation')
                    roster = await rpc(ws, 'subagent.list', session_id=sid)
                    assert roster['result']['subagents'] == [], roster
                    tail = await rpc(ws, 'subagent.tail', session_id=sid, subagent_id='owned')
                    assert not tail['result']['available']
                    from dataclasses import replace
                    from gateway.config import Platform
                    adapter = runner.adapters[Platform.LOCAL]
                    policy = adapter.policies[live.source.chat_id]
                    adapter.policies[live.source.chat_id] = replace(policy, ignore_user_config=True)
                    for method in ('process.list', 'subagent.list', 'subagent.tail'):
                        args = {'subagent_id': 'owned'} if method == 'subagent.tail' else {}
                        result = await rpc(ws, method, session_id=sid, **args)
                        assert result.get('error', {}).get('message') == 'unsupported_projection', result
                finally:
                    gate.touch()
                    for proc in processes:
                        await asyncio.to_thread(proc._completion_event.wait, 10)
                        assert proc.exited and proc.exit_code == 0
                    for child in children:
                        _unregister_subagent(child._subagent_id, agent=child)
            (home / 'receipt.json').write_text(json.dumps({'mode': mode, 'authenticated_ws': True,
                'foreign_actor_denied': True, 'foreign_profile_denied': True, 'no_inference': True}))
    finally:
        await stop_gateway_api(api)


if __name__ == '__main__':
    status = 0
    try:
        asyncio.run(probe(sys.argv[1]))
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
