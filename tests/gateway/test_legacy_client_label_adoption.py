"""A phone app's self-labelled chat continues in Desktop; a plugin platform sharing the label is untouched.

The pre-authority ``hermes serve`` stored whatever ``source`` a client sent, so phone apps signed in
through the dashboard login left routeless rows labelled ``mobile``, ``ios`` and the like (#146814).
They are the owner's chats: Desktop's resume adopts them as writable conversations, while a third-party
WebUI's rows (its own agent writes them) still open read-only. The catalog plugin hermes-mobile
registers ``Platform('mobile')``; its routed chats keep their messaging route.
"""
from types import SimpleNamespace

import pytest

_CAPS = ['session:create', 'session:read', 'session:submit', 'session:control', 'session:approve',
         'session:respond']


async def _owner(tmp_path, monkeypatch, toolsets):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.owner_identity import host_owner_subject
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': toolsets})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _restored_source=lambda entry: entry.origin,
                             _delivery_adapter_for=lambda source: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    desktop = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': host_owner_subject(),
        'profile_id': str(tmp_path), 'instance_id': 'first', 'capabilities': _CAPS})
    return store, desktop


@pytest.mark.asyncio
async def test_a_phone_apps_own_label_resumes_writable_and_a_webui_row_stays_read_only(tmp_path, monkeypatch):
    store, desktop = await _owner(tmp_path, monkeypatch, {'gui': []})
    db = store._db
    for sid, source, user_id in (('phone-mobile', 'mobile', None), ('phone-ios', 'ios', 'basic:admin'),
                                 ('webui-row', 'webui', None), ('teams-row', 'teams', None),
                                 ('curator-row', 'curator', None)):
        db.create_session(sid, source=source, model='m', cwd=str(tmp_path), user_id=user_id)
        db.append_message(sid, role='user', content='hi from ' + sid)
        db.end_session(sid, 'ws_orphan_reap')
    # hermes-mobile installed registers Platform('mobile'): a routeless row with that label still adopts.
    from gateway.config import Platform
    Platform._add_pseudo_member('mobile')
    from hermes_cli.sessions_adopt import adoptable_sessions
    offered = {row['id'] for row in adoptable_sessions(db.db_path)}
    assert {'phone-mobile', 'phone-ios'} <= offered and not offered & {'webui-row', 'teams-row', 'curator-row'}
    for i, sid in enumerate(('phone-mobile', 'phone-ios')):
        if sid == 'phone-mobile':
            # A metadata read first opens a read-only view; the resume then adopts it in place.
            await desktop.dispatch({'id': 50, 'method': 'session.info', 'params': {'session_id': sid}})
        resumed = await desktop.dispatch({'id': i, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert resumed.get('result', {}).get('session_id') == sid, resumed
        assert [m['content'] for m in resumed['result']['messages']] == ['hi from ' + sid]
        sent = await desktop.dispatch({'id': 10 + i, 'method': 'prompt.submit',
                                       'params': {'session_id': sid, 'text': 'go on', 'submission_id': 's-' + sid}})
        assert 'error' not in sent, sent
        # Bound, it is the Desktop chat it resumes as; a second open still finds it.
        assert db.get_session(sid)['source'] == 'desktop'
        again = await desktop.dispatch({'id': 20 + i, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert again.get('result', {}).get('session_id') == sid, again
    # A foreign writer's rows, a bundled plugin platform's label and automation stay read-only.
    for i, sid in enumerate(('webui-row', 'teams-row', 'curator-row')):
        await desktop.dispatch({'id': 30 + i, 'method': 'session.resume', 'params': {'session_id': sid}})
        sent = await desktop.dispatch({'id': 40 + i, 'method': 'prompt.submit',
                                       'params': {'session_id': sid, 'text': 'go on', 'submission_id': 's-' + sid}})
        assert (sent.get('error', {}).get('data') or {}).get('reason') == 'session_read_only', (sid, sent)
        assert db.get_session(sid)['source'] == sid.split('-')[0]


@pytest.mark.asyncio
async def test_a_plugin_platform_sharing_the_label_keeps_its_messaging_route(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource

    store, desktop = await _owner(tmp_path, monkeypatch, {})
    Platform._add_pseudo_member('mobile')  # what the hermes-mobile plugin's platform registration yields
    entry = store.get_or_create_session(SessionSource(platform=Platform('mobile'), chat_id='device-1',
                                                      chat_type='dm', user_id='device-1'))
    store._db.append_message(entry.session_id, role='user', content='from the plugin app')
    resumed = await desktop.dispatch({'id': 1, 'method': 'session.resume',
                                      'params': {'session_id': entry.session_id}})
    assert [m['content'] for m in resumed.get('result', {}).get('messages', [])] == ['from the plugin app'], resumed
    assert store._db.get_session(entry.session_id)['source'] == 'mobile'
