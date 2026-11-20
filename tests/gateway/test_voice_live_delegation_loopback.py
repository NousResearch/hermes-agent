"""Recorded GPT-Live delegation through a real gateway and a loopback model."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from typing import cast

import pytest

from gateway.session_voice_capture import record_voice_events
from hermes_cli.config_defaults import DEFAULT_CONFIG
from persistence import ProfileScope, SessionId
from persistence.conversation.live_voice import VoiceFragment
from persistence.core.backend import PersistenceStore
from persistence.sqlite import SQLiteBackend
from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until
from tools.gpt_live_sideband import ClientDelegation
from tools.voice_live import VOICE_LIVE_TURN_NOTE


CALL = 'recorded-call'
DELEGATION = 'delegate-once'
EVENTS = ('speech-a', 'speech-b', 'speech-c')
RAW_SPEECH = 'Bring the\tnotes. Today.'
SUBMIT_TEXT = 'Bring the notes. Today.'
QUESTION = 'Can we work on the debrief?'
ANSWER = 'I can help with the debrief.'
FOLLOWUP = 'And check the date.'


class _ModelServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(('127.0.0.1', 0), Model)
        self.requests = []


@asynccontextmanager
async def _open_capture(home, sid):
    """Drive the actual recorder with scripted provider events, not prewritten rows."""
    scope = ProfileScope(profile_dir=home, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: home / 'state.db'})
    marked, close = asyncio.Event(), asyncio.Event()

    async def events():
        yield VoiceFragment('question', 'user', QUESTION, 0, 200)
        yield VoiceFragment('answer', 'assistant', ANSWER, 300, 500)
        yield VoiceFragment(EVENTS[0], 'user', 'Bring the\t', 900, 1200)
        yield VoiceFragment(EVENTS[1], 'user', 'notes.', 800, 1050)
        # Real provider deltas leave ordinary gaps between words.
        yield VoiceFragment(EVENTS[2], 'user', ' Today.', 1500, 1700)
        yield ClientDelegation(call_id=CALL, event_id='passive-marker', delegation_id=DELEGATION, offset_ms=1650)
        marked.set()
        await close.wait()
        yield None

    try:
        messages = cast(PersistenceStore, backend.open(scope)).messages
        call = messages.start_voice_capture(SessionId(sid), CALL)
        async with asyncio.TaskGroup() as tasks:
            recorded = tasks.create_task(record_voice_events(messages, call, events()))
            try:
                await asyncio.wait_for(marked.wait(), timeout=5)
                yield
            finally:
                close.set()
        assert recorded.result()
    finally:
        backend.close()


def _query(home, sql, *args):
    with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
        return db.execute(sql, args).fetchall()


def _read_transcript(home, sid):
    return _query(home, 'SELECT id, message_uid, role, content, api_content, display_metadata '
                  'FROM messages WHERE session_id=? ORDER BY id', sid)


async def _settled(home, sid, request_id):
    def admission() -> tuple[str, str, str | None, int | None] | None:
        rows = _query(home, 'SELECT admission_id, status, outcome, generation FROM session_admissions '
                      'WHERE target_session_id=? AND request_id=?', sid, request_id)
        log = (home / 'logs' / 'gateway.log').read_text(encoding='utf-8')
        paused = next((line for line in log.splitlines() if f'Session {sid} paused:' in line), None)
        assert paused is None, f'{request_id}: {paused}; admission={rows!r}'
        return rows[0] if rows and rows[0][1] in ('terminal', 'unknown') else None

    try:
        settled = await async_wait_until(admission, timeout=35, interval=.05, what=f'admission {request_id} to settle')
        assert settled is not None
        assert settled[1] != 'unknown', f'{request_id}: owner lost the execution; admission={settled!r}'
        return settled
    except AssertionError as exc:
        stderr = home / 'logs' / 'managed-worker.log'
        if stderr.exists():
            exc.add_note(stderr.read_text(encoding='utf-8')[-6000:])
        raise


def _user_contents(request):
    return [m['content'] if isinstance(m['content'], str) else ''.join(p.get('text', '') for p in m['content'])
            for m in request['messages'] if m['role'] == 'user']


async def _exercise(home, desc, base, model, managed):
    async with AsyncExitStack() as resources:
        ws = await resources.enter_async_context(websocket(home, desc))
        # The opt-in policy admits a local CLI custom route to a managed worker.
        created = await rpc(ws, 'session.create', request_id='recorded-chat', source='cli' if managed else 'gui',
                            cwd=str(home), model='loopback-voice-delegation', provider='custom', base_url=base,
                            api_key='loopback-only', toolsets=[], ignore_rules=True)
        assert 'result' in created, created
        sid = created['result']['session_id']
        await resources.enter_async_context(_open_capture(home, sid))
        assert [(r[2], r[3]) for r in _read_transcript(home, sid)] == [('user', QUESTION), ('assistant', ANSWER)]
        assert not model.requests
        request_id = 'voice-live:' + json.dumps([CALL, DELEGATION], separators=(',', ':'))
        params = {'session_id': sid, 'submission_id': request_id, 'surface': 'voice-live',
                  'text': SUBMIT_TEXT,
                  'voice_claim': {'call_id': CALL, 'delegation_id': DELEGATION, 'event_ids': list(EVENTS)}}
        reply = await rpc(ws, 'prompt.submit', **params)
        assert reply.get('result', {}).get('status') == 'queued', reply
        admission_id, status, outcome, generation = await _settled(home, sid, request_id)
        assert (status, outcome) == ('terminal', 'completed'), (status, outcome)
        worker = _query(home, 'SELECT kind, status FROM worker_executions WHERE session_id=? AND generation=?',
                        sid, generation)
        assert worker == ([('compute', 'terminal')] if managed else []), worker
        assert len(model.requests) == 1, model.requests
        prior, sent = _user_contents(model.requests[0])
        assert prior == QUESTION and sent.count(VOICE_LIVE_TURN_NOTE) == 1
        assert any(m['role'] == 'assistant' and m['content'] == ANSWER for m in model.requests[0]['messages'])
        assert sent.count(RAW_SPEECH) == 1, repr(sent)
        rows = _read_transcript(home, sid)
        assert [(r[2], r[3]) for r in rows] == [
            ('user', QUESTION), ('assistant', ANSWER), ('user', RAW_SPEECH), ('assistant', 'RECOVERY_ACK_' + sent),
        ]
        assert rows[2][1] and rows[2][4] == sent
        voice = json.loads(rows[2][5])['voice_live']
        assert voice.pop('call_key') and voice == {'partial': False, 'start_ms': 800, 'end_ms': 1700,
                                                         'first_event_id': EVENTS[0]}
        assert _query(home, 'SELECT row_id FROM live_voice_projections WHERE call_id=? AND admission_id=?',
                      CALL, admission_id) == [(rows[2][0],)]

        replay = await rpc(ws, 'prompt.submit', **params)
        assert replay.get('result', {}).get('admission_id') == admission_id, replay
        assert replay['result']['status'] == 'terminal', replay

        assert _read_transcript(home, sid) == rows
        typed = await rpc(ws, 'prompt.submit', session_id=sid, submission_id='typed-followup', text=FOLLOWUP)
        assert typed.get('result', {}).get('status') == 'queued', typed
        assert (await _settled(home, sid, 'typed-followup'))[1:3] == ('terminal', 'completed')
        assert len(model.requests) == 2, model.requests
        assert _user_contents(model.requests[1]) == [QUESTION, sent, FOLLOWUP]
        assert any(m['role'] == 'assistant' and m['content'] == ANSWER for m in model.requests[1]['messages'])

        return sid, rows


@pytest.mark.parametrize('managed', [False, True], ids=['local', 'managed'])
def test_recorded_delegation_reuses_one_durable_user_row(tmp_path, managed):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    # The daemon and its managed child don't inherit pytest's network fence.
    # Fail closed on non-loopback Python socket operations in both processes.
    (user / 'sitecustomize.py').write_text(
        'import socket, sys\n'
        'def loopback_only(event, args):\n'
        '    if event == "socket.getaddrinfo":\n'
        '        host = args[0]\n'
        '    elif event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):\n'
        '        host = args[1][0]\n'
        '    else:\n'
        '        return\n'
        '    if host not in (None, "127.0.0.1", "::1", "localhost"):\n'
        '        raise OSError(f"GPT-Live test blocked external socket: {host}")\n'
        'sys.addaudithook(loopback_only)\n', encoding='utf-8')
    with _ModelServer() as model:
        thread = threading.Thread(target=model.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{model.server_port}/v1'
            (home / 'config.yaml').write_text(json.dumps({
                '_config_version': DEFAULT_CONFIG['_config_version'],
                'gateway': {'multiplex_profiles': False, 'managed_workers': managed},
                'model': {'provider': 'custom', 'default': 'loopback-voice-delegation', 'base_url': base},
                'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'gui': [], 'cli': []},
                'terminal': {'cwd': str(home)},
            }), encoding='utf-8')
            env = child_env() | {'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
                                 'PYTHONPATH': str(user), 'OPENAI_API_KEY': 'loopback-only',
                                 'OPENAI_BASE_URL': base, 'PYTHONUNBUFFERED': '1'}
            with daemon(root, home, env, barrier=False) as (_proc, desc):
                sid, original = asyncio.run(_exercise(home, desc, base, model, managed))
            # Fresh read-only SQLite connection after owner exit: not a cached view of the turn.
            final = _read_transcript(home, sid)
            assert final[:4] == original
            assert [(r[2], r[3]) for r in final[4:]] == [('user', FOLLOWUP), ('assistant', 'RECOVERY_ACK_' + FOLLOWUP)]
        finally:
            model.shutdown()
            thread.join(timeout=5)
