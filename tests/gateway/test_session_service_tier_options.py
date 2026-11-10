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
