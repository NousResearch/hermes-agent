"""The Ink TUI's ``complete.path`` over the canonical connection lists the session's own folder.

The local TUI sends every request on its canonical connection; with no handler there, each path
or ``@`` completion answered ``-32601``. The shared gateway's own cwd is its daemon folder, so the
listing must be rooted at the session's policy cwd.
"""
import json
from types import SimpleNamespace

import pytest

from gateway import session_policy
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def rpc(tmp_path, monkeypatch):
    project, daemon, home = tmp_path / 'project', tmp_path / 'daemon', tmp_path / 'home'
    for folder, name in ((project, 'project-notes.md'), (daemon, 'daemon-notes.md'), (home, None)):
        folder.mkdir()
        if name:
            (folder / name).write_text('x', encoding='utf-8')
    monkeypatch.chdir(daemon)
    monkeypatch.setenv('TERMINAL_CWD', str(daemon))
    backend = {'TERMINAL_ENV': 'local'}
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: SimpleNamespace(
        cwd=str(project), terminal_json=json.dumps({**backend, 'TERMINAL_CWD': str(project)})))
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(home),
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})

        async def call(params):
            return await connection.dispatch({'id': 1, 'method': 'complete.path', 'params': params})

        yield call, backend


@pytest.mark.asyncio
async def test_complete_path_lists_the_session_folder(rpc):
    call, _ = rpc
    reply = await call({'word': '@file:', 'session_id': 's', 'cwd': '/client/claim'})
    assert 'error' not in reply, reply
    texts = [item['text'] for item in reply['result']['items']]
    assert '@file:project-notes.md' in texts
    assert '@file:daemon-notes.md' not in texts


@pytest.mark.asyncio
async def test_complete_path_lists_nothing_for_a_remote_backend_or_without_a_session(rpc):
    call, backend = rpc
    assert (await call({'word': '@file:'}))['result'] == {'items': []}
    backend['TERMINAL_ENV'] = 'ssh'
    assert (await call({'word': '@file:', 'session_id': 's'}))['result'] == {'items': []}


@pytest.mark.asyncio
async def test_complete_path_refuses_a_session_it_does_not_serve(rpc):
    call, _ = rpc
    reply = await call({'word': '@file:', 'session_id': 'other'})
    assert reply['error']['data']['reason'] == 'not_found'
