"""The owner advertises service tiers for the selected route and account, never another session."""
import pytest

from tests.gateway.test_session_create_tier_cwd import _owner


@pytest.mark.asyncio
@pytest.mark.parametrize(('model', 'provider', 'base_url', 'entitled', 'expected'), [
    ('gpt-6.1-sol', 'openai', 'https://api.openai.com/v1', True, {'normal', 'fast', 'auto', 'cold', 'ultrafast'}),
    ('gpt-5.4', 'openai', 'https://api.openai.com/v1', True, {'normal', 'fast', 'auto', 'cold'}),
    ('gpt-6.1-sol', 'openai-codex', 'https://chatgpt.com/backend-api/codex', False,
     {'normal', 'fast', 'auto', 'cold'}),
    ('gpt-6.1-sol', 'openai-codex', 'https://chatgpt.com/backend-api/codex', True,
     {'normal', 'fast', 'auto', 'cold', 'ultrafast'}),
    ('gpt-6.1-sol', 'custom', 'http://127.0.0.1:12345/v1', True, set()),
])
async def test_service_tier_choices_use_the_sessions_model_route_and_account(
    tmp_path, monkeypatch, model, provider, base_url, entitled, expected,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('agent.codex_service_tiers.codex_account_offers_tier', lambda *a, **k: entitled)
    store, _, _, owner = _owner(tmp_path / 'owner', monkeypatch, {'model': {'default': 'fixture'}})
    try:
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'create', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
            'model': model, 'provider': provider, 'base_url': base_url,
        }})
        sid = created['result']['session_id']
        reply = await owner.dispatch({'id': 'choices', 'method': 'config.get', 'params': {
            'session_id': sid, 'key': 'service_tier',
        }})
        assert 'result' in reply, reply
        assert reply['result']['value'] == 'normal'
        assert set(reply['result']['options']) == expected
        if 'ultrafast' not in expected:
            rejected = await owner.dispatch({'id': 'pick', 'method': 'config.set', 'params': {
                'session_id': sid, 'key': 'fast', 'value': 'ultrafast',
            }})
            assert rejected['error']['message'] == 'fast_unavailable'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_service_tier_config_publishes_route_gated_fast_on_session_info(tmp_path, monkeypatch):
    from types import MethodType

    from gateway.run import GatewayRunner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('agent.codex_service_tiers.codex_account_offers_tier', lambda *a, **k: True)
    store, runner, authority, owner = _owner(tmp_path / 'owner', monkeypatch, {'model': {'default': 'fixture'}})
    for name in ('_sessions_map', '_session_state', '_peek_session_state', '_set_session_service_tier_override'):
        setattr(runner, name, MethodType(getattr(GatewayRunner, name), runner))
    try:
        supported = await owner.dispatch({'id': 'c1', 'method': 'session.create', 'params': {
            'request_id': 'c1', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
            'model': 'gpt-6.1-sol', 'provider': 'openai', 'base_url': 'https://api.openai.com/v1',
        }})
        unsupported = await owner.dispatch({'id': 'c2', 'method': 'session.create', 'params': {
            'request_id': 'c2', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
            'model': 'gemini-4-neon', 'provider': 'gemini',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        }})
        sid_ok = supported['result']['session_id']
        sid_no = unsupported['result']['session_id']

        await owner.dispatch({'id': 's1', 'method': 'config.set', 'params': {
            'session_id': sid_ok, 'key': 'fast', 'value': 'ultrafast',
        }})
        await owner.dispatch({'id': 's2', 'method': 'config.set', 'params': {
            'session_id': sid_ok, 'key': 'fast', 'value': 'normal',
        }})
        await owner.dispatch({'id': 's3', 'method': 'config.set', 'params': {
            'session_id': sid_no, 'key': 'service_tier', 'value': 'ultrafast',
        }})

        ok_stream = authority.sessions[sid_ok].event_stream
        no_stream = authority.sessions[sid_no].event_stream
        ok_events = [
            e['payload'] for e in ok_stream.since(ok_stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and 'service_tier' in e.get('payload', {})
        ]
        no_events = [
            e['payload'] for e in no_stream.since(no_stream.watermark()[0], 0)['events']
            if e.get('type') == 'session.info' and 'service_tier' in e.get('payload', {})
        ]
        assert ok_events[-2:] == [
            {'service_tier': 'ultrafast', 'fast': True},
            {'service_tier': '', 'fast': False},
        ]
        assert no_events[-1] == {'service_tier': 'ultrafast', 'fast': False}
    finally:
        await owner.close()
        store.close_all_db_handles()
