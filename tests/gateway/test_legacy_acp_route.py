"""A thread main's in-process ACP server wrote resumes on the route it saved (``model_config``).

Main's ``SessionManager._persist`` stored the session's provider, base_url and api_mode in
``model_config`` (and, on older rows, the workspace too) and ``_restore`` rebuilt the agent from them.
Adoption under the gateway must keep that route rather than the profile's current endpoint.
"""
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal


@pytest.mark.asyncio
async def test_adopted_acp_thread_keeps_its_saved_route_and_workspace(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    from hermes_state_local import local_receipt

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {
        'model': {'default': 'profile-model', 'provider': 'openrouter',
                  'base_url': 'https://openrouter.ai/api/v1', 'api_mode': 'chat_completions'},
        'platform_toolsets': {'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    actor = Principal('uid:1000', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native')
    bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': str(tmp_path),
                                             'instance_id': 'fixture'})
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    # An older row: the workspace only in model_config (the cwd column was added later).
    authority.db.create_session('old-editor-thread', source='acp', model='saved-model', model_config={
        'cwd': str(workspace), 'provider': 'custom', 'base_url': 'http://127.0.0.1:8080/v1',
        'api_mode': 'anthropic_messages'})
    try:
        resolve_local_target(authority, actor, 'old-editor-thread')
        policy = local_receipt(authority.db, 'old-editor-thread')['policy']
        route = json.loads(policy['config_json'])['model']
        assert policy['source'] == 'acp' and policy['model'] == 'saved-model'
        assert policy['cwd'] == str(workspace)
        assert (route['provider'], route['base_url'], route['api_mode']) == (
            'custom', 'http://127.0.0.1:8080/v1', 'anthropic_messages'), route
    finally:
        store._db.close()
