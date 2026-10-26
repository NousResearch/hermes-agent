"""Authenticated TCP fixture for the Ink/Desktop verbs the owner serves itself; no inference."""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback

_CONFIG = '''gateway:
  multiplex_profiles: false
model:
  provider: custom
  default: no-inference
  base_url: http://127.0.0.1:1/v1
auxiliary:
  title_generation:
    enabled: false
quick_commands:
  probe-exec:
    type: exec
    command: "touch {marker} && touch relative-exec-ran && echo QUICK_EXEC_OK"
'''


async def probe(mode):
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    marker = home / 'quick-exec-ran'
    (home / 'config.yaml').write_text(_CONFIG.format(marker=marker), encoding='utf-8')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='routes')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True

    ticket = mint_ticket(user_id='owner', provider='fixture')
    # The same operator with a read-only grant: control, submit and exec verbs must refuse it.
    reader = AuthorityConnection(authority, object(), {'user_id': 'owner', 'provider': 'fixture',
                                                       'capabilities': ['session:read']}, operator=True)
    try:
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            # The session's own folder, distinct from the gateway process's cwd.
            work = home / 'session-work'
            work.mkdir(exist_ok=True)
            created = await rpc(ws, 'session.create', request_id='routes-owner', source='gui', toolsets=[],
                                cwd=str(work))
            assert 'result' in created, created
            sid = created['result']['session_id']
            receipt = {'mode': mode, 'authenticated_ws': True}
            await MODES[mode](authority, ws, reader, sid, marker, receipt)
            (home / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    finally:
        await reader.close()
        await stop_gateway_api(api)


def _error(reply):
    return reply.get('error', {}).get('message')


async def _call(reader, method, **params):
    return await reader.dispatch({'id': method, 'method': method, 'params': params})


async def _react(authority, ws, reader, sid, marker, receipt):
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from gateway.session_ancillary import physical_target
    from gateway.session_contract import SessionRef
    target = physical_target(authority, SessionRef(authority.profile_id, sid))
    user_row = authority.db.append_message(target, 'user', 'REACT_USER_TEXT')
    authority.db.append_message(target, 'assistant', 'REACT_ASSISTANT_TEXT')
    added = await rpc(ws, 'message.react', session_id=sid, row_id=user_row, emoji='👍')
    assert added.get('result', {}).get('row_id') == user_row, added
    assert [r['emoji'] for r in added['result']['reactions']] == ['👍'], added
    assert [r['emoji'] for r in authority.db.get_message_reactions(target, user_row)] == ['👍']
    newest = await rpc(ws, 'message.react', session_id=sid, newest_role='assistant', emoji='❤️')
    assert newest.get('result', {}).get('row_id') == authority.db.latest_message_row_id(target, role='assistant'), newest
    cleared = await rpc(ws, 'message.react', session_id=sid, row_id=user_row, emoji=None)
    assert cleared.get('result', {}).get('reactions') == [], cleared
    for params, reason in [({'row_id': 999999, 'emoji': '👍'}, 'not_found'),
                           ({'emoji': '👍'}, 'invalid_params'),
                           ({'row_id': user_row, 'emoji': '👍', 'profile': 'foreign'}, 'profile_mismatch')]:
        denied = await rpc(ws, 'message.react', session_id=sid, **params)
        assert _error(denied) == reason, (params, denied)
    denied = await _call(reader, 'message.react', session_id=sid, row_id=user_row, emoji='👎')
    assert _error(denied) == 'permission_denied', denied
    assert authority.db.get_message_reactions(target, user_row) == []
    receipt['reacted'] = True


async def _control(authority, ws, reader, sid, marker, receipt):
    from tests.gateway.fixtures.local_recovery_probe import rpc

    async def control(action, **args):
        params = {'action': action, **({'args': args} if args else {})}
        return await rpc(ws, 'session.control', session_id=sid, **params)

    updates = []
    authority.sessions[sid].event_stream.observers.add(
        lambda frame: frame['params']['type'] == 'session.control.update' and updates.append(frame['params']))
    for command, kind in (('goal ship the owned fixture', 'goal'), ('loop every 10m CHECK_LOOP', 'loop'),
                          ('heartbeat every 10m PING_HEARTBEAT', 'heartbeat')):
        typed = await rpc(ws, 'slash.exec', session_id=sid, command=command)
        assert 'result' in typed, (command, typed)
        # A typed command repaints every viewer's cards too (Ink has no other refresh path).
        assert updates and updates.pop()['payload']['control'][kind]['status'] == 'active', (command, updates)
    denied = await _call(reader, 'session.control', session_id=sid, action='goal.pause')
    assert _error(denied) == 'permission_denied', denied
    # Every other viewer of the session: what its card repaints from.
    updates.clear()
    for action, check in [
            ('goal.pause', lambda c: c['goal']['status'] == 'paused'),
            ('goal.resume', lambda c: c['goal']['status'] == 'active'),
            ('loop.pause', lambda c: c['loop']['status'] == 'paused'),
            ('loop.resume', lambda c: c['loop']['status'] == 'active'),
            ('heartbeat.pause', lambda c: c['heartbeat']['status'] == 'paused'),
            ('heartbeat.resume', lambda c: c['heartbeat']['status'] == 'active')]:
        reply = await control(action)
        assert 'result' in reply, (action, reply)
        assert check(reply['result']['control']), (action, reply)
        assert set(reply['result']['dispatch']) == {'type', 'output', 'notice', 'message', 'display'}, reply
        read = await rpc(ws, 'session.control.read', session_id=sid)
        assert read['result']['control'] == reply['result']['control'], (action, read, reply)
        assert updates and updates[-1]['session_id'] == sid, (action, updates)
        assert updates.pop()['payload'] == {'control': reply['result']['control']}, (action, reply)
    # Free subgoal text is a criterion, never a /subgoal verb.
    added = await control('subgoal.add', text='clear the cache first')
    assert added['result']['control']['goal']['subgoals'] == ['clear the cache first'], added
    removed = await control('subgoal.remove', index=1)
    assert removed['result']['control']['goal']['subgoals'] == [], removed
    for action, args in [('goal.gate.add', {}), ('nonsense', {}), ('subgoal.remove', {'index': 0}),
                         ('subgoal.add', {'text': '  '})]:
        rejected = await control(action, **args)
        assert _error(rejected) == 'invalid_params', (action, rejected)
    for action, field in [('heartbeat.clear', 'heartbeat'), ('loop.stop', None), ('goal.clear', 'goal')]:
        reply = await control(action)
        assert 'result' in reply, (action, reply)
        if field:
            assert reply['result']['control'][field] is None, (action, reply)
    foreign = await rpc(ws, 'session.control', session_id=sid, action='goal.pause', profile='foreign')
    assert _error(foreign) == 'profile_mismatch', foreign
    receipt['controlled'] = True


async def _commands(authority, ws, reader, sid, marker, receipt):
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from hermes_cli.plugins import _ensure_plugins_discovered
    calls = []

    def plugin(args):
        from gateway.session_context import get_session_env
        calls.append(args)
        return f'PLUGIN_OK {args} platform={get_session_env("HERMES_SESSION_PLATFORM")}'
    _ensure_plugins_discovered()._plugin_commands['probe-plugin'] = {
        'handler': plugin, 'description': 'fixture command', 'plugin': 'fixture'}

    for method, params in [('slash.exec', {'command': 'probe-exec'}),
                           ('command.dispatch', {'name': 'probe-plugin', 'arg': 'X'})]:
        denied = await _call(reader, method, session_id=sid, **params)
        assert _error(denied) == 'permission_denied', (method, denied)
    assert not marker.exists() and not calls, 'a read-only socket ran an exec/plugin command'
    (Path.cwd() / 'relative-exec-ran').unlink(missing_ok=True)
    ran = await rpc(ws, 'slash.exec', session_id=sid, command='probe-exec')
    assert ran.get('result') == {'type': 'exec', 'output': 'QUICK_EXEC_OK'}, ran
    assert marker.exists()
    # A relative path acts in the session's folder, never the gateway process's.
    work = Path(os.environ['HERMES_HOME']) / 'session-work'
    assert (work / 'relative-exec-ran').exists(), 'exec quick command ran outside the session folder'
    assert not (Path.cwd() / 'relative-exec-ran').exists()
    dispatched = await rpc(ws, 'command.dispatch', session_id=sid, name='probe-plugin', arg='ARG_ONE two')
    assert dispatched.get('result', {}).get('type') == 'exec', dispatched
    assert dispatched['result']['output'].startswith('PLUGIN_OK ARG_ONE two platform=local'), dispatched
    typed = await rpc(ws, 'slash.exec', session_id=sid, command='probe_plugin THREE')
    assert typed.get('result', {}).get('output', '').startswith('PLUGIN_OK THREE'), typed
    assert calls == ['ARG_ONE two', 'THREE']
    # Under multiplex the runner's own quick commands are the LAUNCH profile's; one sharing the plugin's
    # name must neither shadow nor refuse this session's plugin command.
    authority.runner.config.quick_commands['probe-plugin'] = {'type': 'exec', 'command': 'echo LAUNCH_QUICK'}
    shadowed = await rpc(ws, 'command.dispatch', session_id=sid, name='probe-plugin', arg='FOUR')
    assert shadowed.get('result', {}).get('output', '').startswith('PLUGIN_OK FOUR'), shadowed
    assert calls[-1] == 'FOUR'
    unknown = await rpc(ws, 'slash.exec', session_id=sid, command='no-such-command')
    assert _error(unknown) == 'unsupported_command', unknown
    assert not authority.db.get_messages(sid), 'a command started a turn'
    receipt['commands'] = True


async def _control_refusals(authority, ws, reader, sid, marker, receipt):
    """RT-8: a criterion button that cannot act is a refusal the card shows, never a success envelope."""
    from tests.gateway.fixtures.local_recovery_probe import rpc

    async def control(action, **args):
        return await rpc(ws, 'session.control', session_id=sid, action=action, args=args)

    def refused(reply, reason):
        assert 'result' not in reply, reply
        assert _error(reply) == reason and reply['error']['data'].get('detail'), reply

    refused(await control('subgoal.add', text='write the release notes'), 'goal_not_active')
    refused(await control('subgoal.clear'), 'goal_not_active')
    assert 'result' in await rpc(ws, 'slash.exec', session_id=sid, command='goal ship the owned fixture')
    added = await control('subgoal.add', text='cite the changelog')
    assert added['result']['control']['goal']['subgoals'] == ['cite the changelog'], added
    refused(await control('subgoal.remove', index=2), 'invalid_params')
    read = await rpc(ws, 'session.control.read', session_id=sid)
    assert read['result']['control']['goal']['subgoals'] == ['cite the changelog'], read
    receipt['refused'] = True


async def _connection_card(authority, ws, reader, sid, marker, receipt):
    """RT-2: the manage_connections card's calls reach the session's own open operation, with the legacy handlers' policy and error shape."""
    from pathlib import Path
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from gateway.run import _profile_runtime_scope
    from tools.connectors import live, run
    from tools.connectors.contract import Actor, TargetState
    from tools import connectors
    from tools.connectors.gateway.errors import GatewayAuthError
    from tools.connectors.operation import ConnectionOperation, Target

    created = await rpc(ws, 'session.create', request_id='routes-card', source='gui', toolsets=['connections'],
                        cwd=str(Path(authority.profile_id) / 'session-work'))
    sid = created['result']['session_id']
    owner = {'type': 'session', 'session_id': sid}
    connectors.connectors_available = lambda *a, **k: True
    with _profile_runtime_scope(Path(authority.profile_id)):
        operation = ConnectionOperation([Target('linear', 'mcp', 'install')], session_key=authority.sessions[sid].route)
        live.open(operation)
    # A resume restores the open card.
    resumed = await rpc(ws, 'session.resume', session_id=sid)
    assert resumed['result']['pending_connection']['op_id'] == operation.op_id, resumed
    status = await rpc(ws, 'connectors.operation.status', owner=owner, op_id=operation.op_id)
    assert status.get('result', {}).get('op_id') == operation.op_id, status
    assert (await rpc(ws, 'connectors.operation.wake', owner=owner, op_id=operation.op_id)).get('result') == {'status': 'ok'}

    # Try again on a failed target re-runs it under the session's own context and returns the operation.
    operation.transition('linear', TargetState.failed, Actor.backend_watcher)
    seen = {}

    def rerun(op, names):
        from gateway.session_context import get_session_env
        seen.update(names=names, key=get_session_env('HERMES_SESSION_KEY'), platform=get_session_env('HERMES_SESSION_PLATFORM'))
        op.transition('linear', TargetState.initiated, Actor.user)

    run.reissue = rerun
    retried = await rpc(ws, 'connectors.connect', owner=owner, connectors=['linear'], reconnect=True)
    assert retried.get('result', {}).get('targets', [{}])[0].get('state') == 'initiated', retried
    assert seen == {'names': ['linear'], 'key': authority.sessions[sid].route, 'platform': 'local'}, seen

    def auth_failure(op, names):
        raise GatewayAuthError('expired', code='invalid_token', status=401)

    run.reissue = auth_failure
    refused = await rpc(ws, 'connectors.connect', owner=owner, connectors=['linear'], reconnect=True)
    assert refused['error']['code'] == 4032 and refused['error']['data']['reason'] == 'NEEDS_NOUS_AUTH', refused

    def broken(op, names):
        raise OSError('transport down')

    run.reissue = broken
    failed = await rpc(ws, 'connectors.connect', owner=owner, connectors=['linear'], reconnect=True)
    assert failed['error']['code'] == 5034 and failed['error']['data']['reason'] == 'CONNECTOR_REQUEST_FAILED', failed
    connectors.connectors_available = lambda *a, **k: False
    gated = await rpc(ws, 'connectors.connect', owner=owner, connectors=['linear'], reconnect=True)
    assert gated['error']['code'] == 4031 and gated['error']['data']['reason'] == 'CONNECTORS_UNAVAILABLE', gated
    connectors.connectors_available = lambda *a, **k: True
    # A session whose tools leave out manage_connections cannot re-run a target, even with connectors on.
    toolless = (await rpc(ws, 'session.create', request_id='routes-toolless', source='gui', toolsets=[],
                          cwd=str(Path(authority.profile_id) / 'session-work')))['result']['session_id']
    with _profile_runtime_scope(Path(authority.profile_id)):
        foreign = ConnectionOperation([Target('linear', 'mcp', 'install')], session_key=authority.sessions[toolless].route)
        live.open(foreign)
    foreign.transition('linear', TargetState.failed, Actor.backend_watcher)
    seen.clear()
    run.reissue = rerun
    untooled = await rpc(ws, 'connectors.connect', owner={'type': 'session', 'session_id': toolless},
                         connectors=['linear'], reconnect=True)
    assert untooled['error']['code'] == 4031 and seen == {}, untooled
    # A well-formed envelope with a bad answer is an invalid answer, not invalid parameters.
    for result in ({'targets': [{'name': 'linear', 'status': 'connected'}]}, None):
        bad = await rpc(ws, 'connection.respond', owner=owner, op_id=operation.op_id,
                        **({'result': result} if result is not None else {}))
        assert bad['error']['code'] == 4002 and bad['error']['data']['reason'] == 'INVALID_ANSWER', bad

    answered = await rpc(ws, 'connection.respond', owner=owner, op_id=operation.op_id,
                         result={'targets': [{'name': 'linear', 'status': 'skipped'}]})
    assert answered.get('result') == {'status': 'ok', 'settled': True}, answered
    assert operation.settled and operation.target('linear').state.value == 'skipped'
    closed = await rpc(ws, 'connectors.connect', owner=owner, connectors=['linear'], reconnect=True)
    assert closed['error']['code'] == 4004 and closed['error']['data']['reason'] == 'UNKNOWN_OPERATION', closed
    assert (await rpc(ws, 'session.resume', session_id=sid))['result'].get('pending_connection') is None
    missing = await rpc(ws, 'connection.respond', owner=owner, op_id='no-such-op', result={'targets': []})
    assert missing['error']['code'] == 4004, missing
    # A read-only caller is refused before the session id is looked up, so it cannot probe for sessions.
    for target in (sid, 'no-such-session'):
        denied = await _call(reader, 'connection.respond', owner={'type': 'session', 'session_id': target},
                             op_id='no-such-op', result={'targets': []})
        assert _error(denied) == 'permission_denied', denied
    receipt['card'] = True


MODES = {'react': _react, 'control': _control, 'control_refusals': _control_refusals, 'commands': _commands,
         'connection_card': _connection_card}


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
