"""Desktop large-paste ``title_preview`` and hidden ``display_kind`` sends survive canonical admission (N3)."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.gateway.test_surface_admission import _admissions, _authority, _wait_for


@pytest.mark.asyncio
async def test_display_fields_are_admitted_and_bound_to_their_own_turn(tmp_path, monkeypatch):
    from gateway.response_filters import display_kind_for_event, display_metadata_for_event
    from gateway.session_controls import AuthorityConnection
    from hermes_state_runtime import list_session_admissions

    seen = []
    async def answer(event):
        # A follow-up event on the same task (a /queue chain) must not inherit the admission's presentation.
        stray = SimpleNamespace(message_id='other', internal=False, metadata={})
        seen.append((event.text, display_kind_for_event(event), display_metadata_for_event(event),
                     display_kind_for_event(stray), display_metadata_for_event(stray)))
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    preview = 'P' * 1500
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        for rid, extra in (('paste', {'title_preview': preview}), ('hidden', {'display_kind': 'hidden'}), ('plain', {})):
            reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': rid, 'text': rid, **extra}})
            assert reply.get('result', {}).get('status') == 'queued', reply
            await authority.sessions['s'].task
        # Only `hidden` is client-authorable; any other kind is refused, never rendered as a user row.
        refused = await connection.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
            'session_id': 's', 'submission_id': 'minted', 'text': 'x', 'display_kind': 'internal_notification'}})
        assert refused['error']['message'] == 'invalid_params', refused
    finally:
        await connection.close()

    rows = {row['request_id']: row for row in list_session_admissions(authority.db, session_id='s', pending_only=False)}
    assert rows['paste']['payload']['display_v1'] == {'title_preview': 'P' * 1000}
    assert rows['hidden']['payload']['display_v1'] == {'kind': 'hidden'}
    assert 'display_v1' not in rows['plain']['payload'] and 'minted' not in rows
    assert seen == [
        ('paste', None, {'title_preview': 'P' * 1000}, None, {}),
        ('hidden', 'hidden', {}, None, {}),
        ('plain', None, {}, None, {}),
    ]


def test_real_daemon_persists_hidden_rows_and_the_paste_title_preview(tmp_path, monkeypatch):
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
        # Instant (derived) title only: no auxiliary model call.
        'auxiliary': {'title_generation': {'enabled': True, 'model_upgrade_enabled': False}},
        'platform_toolsets': {'gui': []}, 'terminal': {'cwd': str(home)},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')

    paste = home / 'paste.txt'
    paste.write_text('Quarterly revenue reconciliation notes\nrow 1\n' * 200)

    async def drive(desc):
        async with websocket(home, desc) as ws:
            sids = {}
            for name, extra in (('paste', {'title_preview': 'Quarterly revenue notes PREVIEW_ONLY'}),
                                ('hidden', {'display_kind': 'hidden'})):
                created = await rpc(ws, 'session.create', request_id='display-' + name, source='gui', cwd=str(home), toolsets=[])
                assert 'result' in created, created
                sids[name] = created['result']['session_id']
                reply = await rpc(ws, 'prompt.submit', session_id=sids[name], submission_id=name,
                                  text=f'@file:{paste}' if name == 'paste' else 'HIDDEN_WIDGET_INTENT', **extra)
                assert reply.get('result', {}).get('status') == 'queued', reply
                await asyncio.to_thread(_wait_for, lambda: _admissions(home).get(name) == 'terminal')
            return sids

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            sids = asyncio.run(drive(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)

    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        rows = {sid: db.execute("SELECT content, display_kind, display_metadata FROM messages WHERE session_id=? AND role='user'",
                                (sid,)).fetchall() for sid in sids.values()}
    [(hidden_text, hidden_kind, _)] = rows[sids['hidden']]
    assert (hidden_text, hidden_kind) == ('HIDDEN_WIDGET_INTENT', 'hidden'), 'a hidden widget intent never persists as a user bubble'
    [(_, paste_kind, paste_meta)] = rows[sids['paste']]
    # The session titler reads the user row's display_metadata.title_preview (agent/turn_context.py).
    assert paste_kind is None and json.loads(paste_meta)['title_preview'] == 'Quarterly revenue notes PREVIEW_ONLY'
    # The preview is titler input only: it never reaches the model turn.
    assert model.requests and not any('PREVIEW_ONLY' in json.dumps(r['messages']) for r in model.requests)
