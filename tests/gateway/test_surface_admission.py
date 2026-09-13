"""HUD / live-voice submit fields survive canonical admission and reach the model as a per-turn note."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


async def _authority(tmp_path, monkeypatch, answer):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    store._db.create_session('s', source='telegram')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    return authority


@pytest.mark.asyncio
async def test_surface_fields_are_admitted_and_scoped_to_their_own_turn(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_surface import surface_turn_note
    from hermes_state_runtime import list_session_admissions
    from tools.voice_live import VOICE_LIVE_TURN_NOTE

    seen = []
    async def answer(event):
        # The note is rendered against the executing agent's tools; a live-voice turn needs none.
        seen.append((event.text, surface_turn_note(SimpleNamespace(valid_tool_names=set()))))
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
            'session_id': 's', 'submission_id': 'spoken', 'text': "what's the weather",
            'surface': 'voice-live', 'voice_context': "Voice assistant: Hi\nUser: what's the weather"}})
        assert reply['result']['status'] == 'queued', reply
        await authority.sessions['s'].task
        plain = await connection.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
            'session_id': 's', 'submission_id': 'typed', 'text': 'and tomorrow?'}})
        assert plain['result']['status'] == 'queued', plain
        await authority.sessions['s'].task
        rejected = await connection.dispatch({'id': 4, 'method': 'prompt.submit', 'params': {
            'session_id': 's', 'submission_id': 'smuggled', 'text': 'x', 'voice_context': 'User: smuggled'}})
        assert rejected['error']['message'] == 'invalid_params', rejected
    finally:
        await connection.close()

    rows = {row['request_id']: row for row in list_session_admissions(authority.db, session_id='s', pending_only=False)}
    assert rows['spoken']['payload']['surface_v1'] == {
        'surface': 'voice-live', 'voice_context': "Voice assistant: Hi\nUser: what's the weather"}
    assert 'surface_v1' not in rows['typed']['payload'] and 'smuggled' not in rows
    (spoken_text, spoken_note), (typed_text, typed_note) = seen
    assert spoken_text == "what's the weather", 'the persisted user row stays the words the user said'
    assert spoken_note.startswith(VOICE_LIVE_TURN_NOTE) and "User: what's the weather" in spoken_note
    assert (typed_text, typed_note) == ('and tomorrow?', '')


def _admissions(home):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        return dict(db.execute('SELECT request_id, status FROM session_admissions'))


def _wait_for(predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail('Timed out waiting for the admitted turn')


def test_real_daemon_accepts_hud_and_voice_live_submits_and_notes_the_model(tmp_path, monkeypatch):
    from tools.voice_live import VOICE_LIVE_TURN_NOTE

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(user))
    monkeypatch.setenv('USERPROFILE', str(user))
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=model.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'gui': []}, 'terminal': {'cwd': str(home)},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')

    async def drive(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='surface-target', source='gui', cwd=str(home), toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            replies = {}
            for name, extra in (('window', {}), ('hud', {'surface': 'hud'}),
                                ('voice', {'surface': 'voice-live', 'voice_context': 'User: earlier words'})):
                replies[name] = await rpc(ws, 'prompt.submit', session_id=sid, submission_id=name, text=name.upper() + '_TURN', **extra)
                await asyncio.to_thread(_wait_for, lambda: _admissions(home).get(name) == 'terminal')
            return replies

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            replies = asyncio.run(drive(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)

    assert all(reply.get('result', {}).get('status') == 'queued' for reply in replies.values()), replies
    users = {}
    for request in model.requests:
        content = next(m['content'] for m in reversed(request['messages']) if m['role'] == 'user')
        text = content if isinstance(content, str) else ''.join(p.get('text', '') for p in content)
        for name in ('window', 'hud', 'voice'):
            if name.upper() + '_TURN' in text:
                users[name] = text
    assert set(users) == {'window', 'hud', 'voice'}, users
    # Live voice: the delegation contract + the spoken context ride the model input, never the row.
    assert VOICE_LIVE_TURN_NOTE in users['voice'] and 'User: earlier words' in users['voice']
    assert VOICE_LIVE_TURN_NOTE not in users['window'] and VOICE_LIVE_TURN_NOTE not in users['hud']
    # HUD without read_window_below (toolsets=[]) is accepted and withholds the tool-gated note.
    assert 'HUD mode' not in users['hud'] and 'HUD mode' not in users['window']
    assert all(request['messages'][0]['role'] == 'system' for request in model.requests)
    assert len({request['messages'][0]['content'] for request in model.requests}) == 1, 'frozen system prefix must not vary per surface'
