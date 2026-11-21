"""A plugin tool reads the accepted request id of the turn calling it (``plugin_session.session_submission_id``).

The id is bound to the admission the tool call executes under, in the gateway's own turn and in a
managed worker alike: a second turn queued on the same session, admitted while the first one's tool
is still running, never shows through to the first one's call.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until

ROOT = Path(__file__).resolve().parents[2]

_PLUGIN = '''import json
import os
import time
from pathlib import Path

from hermes_cli.plugin_session import session_submission_id


def register(ctx):
    def probe(args, **kw):
        log, release = Path(%r), Path(%r)
        if args.get("hold"):
            with log.open("a") as out:
                out.write(json.dumps({"label": args["label"], "held": True}) + "\\n")
            deadline = time.monotonic() + 60
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(.05)
        with log.open("a") as out:
            out.write(json.dumps({"label": args["label"], "id": session_submission_id(), "pid": os.getpid()}) + "\\n")
        return json.dumps({"ok": True})
    ctx.register_tool(name="sid_probe", toolset="sid_probe", schema={"name": "sid_probe", "description": "probe",
        "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "hold": {"type": "boolean"}}}},
        handler=probe)
'''


class Model(BaseHTTPRequestHandler):
    """A user turn ``HOLD <label>`` / ``PROBE <label>`` calls ``sid_probe`` once; its result ends the turn."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or [{'role': 'system'}]
        last = messages[-1]
        text = last.get('content') if last['role'] == 'user' and isinstance(last.get('content'), str) else ''
        verb, _, label = text.partition(' ')
        if verb in ('HOLD', 'PROBE'):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'c{len(messages)}', 'index': 0,
                       'type': 'function', 'function': {'name': 'sid_probe',
                       'arguments': json.dumps({'label': label, 'hold': verb == 'HOLD'})}}]}
        else:
            message = {'role': 'assistant', 'content': 'DONE'}
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}
        frame = {'id': 'sid', 'model': 'm', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _lines(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('managed', [False, True], ids=['in-process', 'managed-worker'])
def test_each_tool_call_reads_its_own_turns_submission_id(tmp_path, managed):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    log, release = tmp_path / 'calls.jsonl', tmp_path / 'release'
    plugin = home / 'plugins' / 'sid-probe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text("name: sid-probe\nversion: '1.0'\n")
    (plugin / '__init__.py').write_text(_PLUGIN % (str(log), str(release)))
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': managed},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'approvals': {'mode': 'off'},
        'plugins': {'enabled': ['sid-probe']}, 'platform_toolsets': {'cli': ['sid_probe']}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def statuses():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return dict(db.execute('SELECT request_id, status FROM session_admissions').fetchall())

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='sid', source='cli', cwd=str(home), model='m',
                                provider='custom', base_url=url, api_key='loopback-only', toolsets=['sid_probe'],
                                ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            first = await rpc(ws, 'prompt.submit', session_id=sid, submission_id='turn-a', text='HOLD a')
            assert first.get('result', {}).get('status') == 'queued', first
            await async_wait_until(lambda: any(line.get('held') for line in _lines(log)), timeout=90, interval=.1,
                                   what="turn A's tool call to start")
            # Turn B is admitted while A's tool call is still running.
            second = await rpc(ws, 'prompt.submit', session_id=sid, submission_id='turn-b', text='PROBE b')
            assert second.get('result', {}).get('status') == 'queued', second
            await async_wait_until(lambda: 'turn-b' in statuses(), timeout=10, interval=.05, what='turn B admitted')
            release.touch()
            await async_wait_until(lambda: statuses().get('turn-b') == 'terminal', timeout=90, interval=.1,
                                   what='both turns to settle')
    try:
        with daemon(ROOT, home, env, barrier=False) as (owner, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()

    calls = [line for line in _lines(log) if not line.get('held')]
    assert [(c['label'], c['id']) for c in calls] == [('a', 'turn-a'), ('b', 'turn-b')], calls
    # The managed path really ran the tools in worker children, one per admitted turn.
    pids = {c['pid'] for c in calls}
    assert (owner.pid not in pids and len(pids) == 2) if managed else pids == {owner.pid}, (owner.pid, calls)


def _row(request_id, **payload):
    return {'admission_id': 'adm-' + request_id, 'request_id': request_id, 'payload': {'text': 'x', **payload}}


def _worker_frame(row, tmp_path, monkeypatch):
    """The bootstrap frame the owner hands the managed child for ``row`` (``_bootstrap``), through JSON."""
    from gateway.session_managed_worker import _bootstrap
    from tests.gateway.test_managed_turn_contract import assignment
    _frame, (authority, ref, _base, policy, scope) = assignment(tmp_path, monkeypatch)
    return json.loads(json.dumps(_bootstrap(authority, ref, row, policy, scope)))


@pytest.mark.parametrize('row, expected', [
    (_row('turn-a'), 'turn-a'),
    (_row('native-1', native_text_v1={}), None),  # a chat message: its event is the platform's
    (_row('tick-1', local_automation_v1={}), None),  # plugin injection, completions, /loop
], ids=['submitted', 'native', 'automation'])
def test_the_managed_child_reads_what_the_owner_would_have_bound(tmp_path, monkeypatch, row, expected):
    from gateway.session_kanban import run_worker_turns
    from hermes_cli.plugin_session import session_submission_id
    frame = _worker_frame(row, tmp_path, monkeypatch)
    seen = []
    agent = SimpleNamespace(valid_tool_names=set(), run_conversation=lambda text, **kw: seen.append(
        session_submission_id()) or {'final_response': 'ok'})
    run_worker_turns(agent, frame, [])
    assert seen == [expected] and session_submission_id() is None


def test_a_forged_submission_id_in_the_bootstrap_is_refused(tmp_path, monkeypatch):
    from gateway.session_managed_worker import worker_turn_scope
    frame = _worker_frame(_row('turn-a'), tmp_path, monkeypatch)
    request = json.loads(frame['policy']['request_json'])
    for forged in ('', 7, None):
        request['turn_v1']['submission_id'] = forged
        bad = {**frame, 'policy': {**frame['policy'], 'request_json': json.dumps(request)}}
        with pytest.raises(ValueError, match='invalid_managed_worker_bootstrap'), worker_turn_scope(bad):
            pass


def test_in_process_only_the_admissions_own_event_reads_its_id():
    """None outside a turn (the classic CLI, cron); inside the admission's turn its id, and not in a
    queued follow-up the same turn drains as its own input (another sender's message)."""
    from gateway.session_context import clear_session_vars, scoped_turn_actor, set_session_vars
    from gateway.session_display import submission_turn_scope
    from gateway.session_state import TurnState
    from hermes_cli.plugin_session import session_submission_id

    admitted = SimpleNamespace(message_id='adm-a', user_id='u')
    follow_up = SimpleNamespace(message_id='native-2', user_id='v')
    assert session_submission_id() is None
    with submission_turn_scope('adm-a', 'turn-a'):
        tokens = set_session_vars(session_key='k', turn_event=admitted)
        try:
            outer = session_submission_id()
            with scoped_turn_actor(follow_up, TurnState()):
                inner = session_submission_id()
            after = session_submission_id()
        finally:
            clear_session_vars(tokens)
        assert session_submission_id() is None  # no turn event bound: not inside this admission's turn
    with submission_turn_scope('adm-b', None):  # an admission with no request id
        tokens = set_session_vars(session_key='k', turn_event=SimpleNamespace(message_id='adm-b'))
        try:
            unnamed = session_submission_id()
        finally:
            clear_session_vars(tokens)
    assert (outer, inner, after, unnamed) == ('turn-a', None, 'turn-a', None)
