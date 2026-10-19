"""Read actual session rows through the authenticated dashboard app across two profiles."""
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(scope='module')
def session_api(tmp_path_factory):
    home = tmp_path_factory.mktemp('session-api')
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(home / 'config.yaml', {'lsp': {'enabled': False}})
    from hermes_cli.profiles import _get_default_hermes_home, _get_profiles_root, create_profile

    assert _get_default_hermes_home().resolve() == home.resolve()
    assert _get_profiles_root().resolve() == (home / 'profiles').resolve()
    worker = create_profile('api-acceptance', no_alias=True, no_skills=True)
    assert worker.resolve() == (home / 'profiles/api-acceptance').resolve()
    from hermes_state import SessionDB

    root = Path(__file__).resolve().parents[3]
    content = (root / 'README.md').read_text(encoding='utf-8')
    sessions = {}
    for name, owner in [('default', home), ('api-acceptance', worker)]:
        sid = uuid4().hex
        db = SessionDB(db_path=owner / 'state.db')
        try:
            db.create_session(sid, source='cli', model='gpt-4.1', system_prompt=content,
                              model_config={'provider': 'openai', 'model': 'gpt-4.1'}, user_id=os.getlogin())
            db.append_message(sid, 'user', content.splitlines()[0])
            stored = db.get_session(sid)
            assert stored['system_prompt'] == content
            assert stored['model_config'] and stored['user_id'] == os.getlogin()
        finally:
            db.close()
        sessions[name] = sid
    from starlette.testclient import TestClient
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    assert _SESSION_TOKEN
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    try:
        yield client, sessions
    finally:
        client.close()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


@pytest.mark.parametrize('surface,full', [('sessions', False), ('sessions', True),
                                        ('profiles', True), ('sidebar', False), ('detail', False)])
def test_real_session_rows_keep_their_projection_across_profiles(session_api, surface, full):
    client, sessions = session_api
    for profile in ('default', 'api-acceptance', 'default'):
        if surface == 'sidebar':
            response = client.get('/api/profiles/sessions/sidebar', params={'recents_profile': profile})
            assert response.status_code == 200, response.text
            rows = response.json()['recents']['sessions']
        elif surface == 'detail':
            response = client.get('/api/sessions/' + sessions[profile], params={'profile': profile})
            assert response.status_code == 200, response.text
            rows = [response.json()]
        else:
            endpoint = '/api/sessions' if surface == 'sessions' else '/api/profiles/sessions'
            response = client.get(endpoint, params={'profile': profile, 'full': int(full)})
            assert response.status_code == 200, response.text
            rows = response.json()['sessions']
        assert {row['id'] for row in rows} == {sessions[profile]}
        for row in rows:
            assert not {'system_prompt', 'system_prompt_hash', 'model_config', 'user_id'} & row.keys()
            assert row['profile'] == profile and row['model'] == 'gpt-4.1'
            assert isinstance(row['archived'], bool) and isinstance(row['pinned'], bool)
