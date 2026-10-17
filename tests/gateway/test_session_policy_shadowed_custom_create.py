"""A Desktop pick of a custom endpoint named like a built-in freezes the endpoint's own route (#140107).

The Desktop picker identifies a custom endpoint's row by its bare key (``hermes_cli.inventory``), and
``session.create`` carries that pick. An entry named ``deepseek`` is shadowed by the built-in in raw
provider resolution, so a bare ``deepseek`` frozen as the session's route ran on the native provider
(no key, or an OpenRouter fallback with the private model name).
"""
import json
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection

_ENTRY = {'name': 'deepseek', 'api': 'https://api.deepseek.com', 'key_env': 'DEEPSEEK_ENTRY_KEY',
          'transport': 'chat_completions', 'default_model': 'deepseek-flash'}


@pytest.mark.asyncio
@pytest.mark.parametrize('pick', ['deepseek', 'custom:deepseek'])
async def test_session_create_with_the_entrys_bare_pick_freezes_its_custom_route(tmp_path, monkeypatch, pick):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from hermes_state_local import local_receipt

    for var in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    config = {'platform_toolsets': {'cli': []}, 'providers': {'deepseek': dict(_ENTRY)},
              'model': {'default': 'deepseek-flash', 'provider': 'custom:deepseek',
                        'base_url': 'https://api.deepseek.com', 'key_env': 'DEEPSEEK_ENTRY_KEY'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: json.loads(json.dumps(config)))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        # What desktopSessionCreateParams() sends for a manual pick of the endpoint's row.
        reply = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'desktop-pick', 'source': 'gui', 'cwd': str(tmp_path),
            'model': 'deepseek-flash', 'provider': pick}})
        assert 'error' not in reply, reply
        policy = local_receipt(authority.db, reply['result']['session_id'])['policy']
        model = json.loads(policy['config_json'])['model']
        assert (model.get('provider'), model.get('base_url'), model.get('key_env')) == \
            ('custom:deepseek', 'https://api.deepseek.com', 'DEEPSEEK_ENTRY_KEY'), model
    finally:
        await connection.close()
        authority.db.close()
