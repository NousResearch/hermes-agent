"""A repeated `-m alias` on resume is judged by the owner's creation provenance, not the client's
own alias map (andrexibiza C19): the owner maps `fast` to model A, the remote client's home maps it
to model B; the same alias resumes, a genuinely different model is still refused."""
import argparse
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_repeated_alias_resumes_from_a_client_with_a_different_alias_map(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session, local_session_info
    from hermes_cli import gateway_chat, model_switch
    import hermes_yaml as yaml

    owner, client = tmp_path / 'owner', tmp_path / 'client'
    owner.mkdir(), client.mkdir()
    alias_cfg = lambda model: {
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'default': 'model-main'},
        'model_aliases': {'fast': {'model': model, 'provider': 'custom', 'base_url': 'http://127.0.0.1:3/v1'}}}
    (client / 'config.yaml').write_text(yaml.safe_dump(alias_cfg('client-model')))
    (owner / 'config.yaml').write_text(yaml.safe_dump(alias_cfg('owner-model')))
    monkeypatch.setenv('HERMES_HOME', str(owner))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: alias_cfg('owner-model'))
    model_switch.DIRECT_ALIASES.clear()
    store = SessionStore(owner / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    try:
        actor = AuthorityConnection(authority, object(), {'user_id': 'u', 'profile_id': 'fixture'}).actor
        ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'cli', 'cwd': str(tmp_path),
                                                      'model': 'fast', 'toolsets': []})
        snapshot = {'info': local_session_info(authority, ref)}
        assert snapshot['info']['launch_request']['model'] == 'owner-model'
        monkeypatch.setenv('HERMES_HOME', str(client))  # the remote client's own home
        model_switch.DIRECT_ALIASES.clear()
        gateway_chat.check_resume_policy(argparse.Namespace(resume=ref.session_id, model='fast'), snapshot)
        gateway_chat.check_resume_policy(argparse.Namespace(resume=ref.session_id, model='owner-model'), snapshot)
        for model in ('client-model', 'other'):
            with pytest.raises(gateway_chat.GatewayClientError):
                gateway_chat.check_resume_policy(argparse.Namespace(resume=ref.session_id, model=model), snapshot)
        with pytest.raises(Exception, match='invalid_params'):  # provenance is owner-stamped only
            create_local_session(authority, actor, {'request_id': 'x', 'source': 'cli', 'cwd': str(tmp_path),
                                                    'model': 'm', 'model_requested': 'fast', 'toolsets': []})
    finally:
        store.close_all_db_handles()
