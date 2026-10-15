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
    (home / 'config.yaml').write_text(_CONFIG.format(marker=marker))
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
            (home / 'receipt.json').write_text(json.dumps(receipt))
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

    for command in ('goal ship the owned fixture', 'loop every 10m CHECK_LOOP', 'heartbeat every 10m PING_HEARTBEAT'):
        typed = await rpc(ws, 'slash.exec', session_id=sid, command=command)
        assert 'result' in typed, (command, typed)
    denied = await _call(reader, 'session.control', session_id=sid, action='goal.pause')
    assert _error(denied) == 'permission_denied', denied
    # Every other viewer of the session: what its card repaints from.
    updates = []
    authority.sessions[sid].event_stream.observers.add(
        lambda frame: frame['params']['type'] == 'session.control.update' and updates.append(frame['params']))
    created = await control('goal.set', title='Ship the structured controls', max_turns=7,
                            contract={'outcome': 'Setters work', 'verification': 'Run focused tests'})
    assert created['result']['control']['goal']['title'] == 'Ship the structured controls', created
    timed = await control('goal.wait_time', seconds=30, reason='Await CI')
    assert timed['result']['control']['goal']['wait_barrier']['type'] == 'until', timed
    loop_set = await control('loop.set', prompt='CHECK_LOOP', interval_seconds=None, times=3)
    assert loop_set['result']['control']['loop']['mode'] == 'self_paced', loop_set
    heartbeat_set = await control('heartbeat.set', prompt='PING_HEARTBEAT', interval_seconds=60)
    assert heartbeat_set['result']['control']['heartbeat']['interval_seconds'] == 60, heartbeat_set
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


MODES = {'react': _react, 'control': _control, 'commands': _commands}


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
