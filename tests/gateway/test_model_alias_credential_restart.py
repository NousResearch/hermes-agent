"""N48/N30/E02: ``/model <alias>`` with a LITERAL alias key commits a reference to the alias entry,
so a restarted owner re-derives it (no ``launch_credentials_unavailable`` wedge), and the key is
read from this owner's profile, never from the process-global alias cache another served profile
may have refilled."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_literal_alias_key_survives_owner_restart_and_stays_in_profile(tmp_path, monkeypatch):
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch
    old_url, far_url = 'http://127.0.0.1:9/v1', 'http://127.0.0.3:9/v1'
    (tmp_path / 'config.yaml').write_text(
        f'model_aliases:\n  Far: {{model: far-model, provider: custom, base_url: "{far_url}", api_key: sk-far-literal}}\n')
    # What another served profile's own /model left in the process-global cache.
    monkeypatch.setitem(model_switch.DIRECT_ALIASES, 'far', model_switch.DirectAlias(
        'far-model', 'custom', far_url, api_key='sk-other-profile'))

    def owner(db):
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        authority.runner = SimpleNamespace(session_authority=authority,
                                           _resolve_session_agent_runtime=lambda **k: (None, {}))
        return authority

    def switch(alias, base_url):
        monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
            success=True, new_model=alias + '-model', target_provider='custom', base_url=base_url,
            api_key='resolved-at-switch', resolved_via_alias=alias))

    with SessionDB(tmp_path / 'state.db') as db:
        first = owner(db)
        policy = build_policy({'cwd': str(tmp_path), 'model': 'a', 'provider': 'custom', 'base_url': old_url},
                              {'model': {'provider': 'custom', 'default': 'a', 'base_url': old_url}})
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        switch('far', far_url)
        committed = (await prepare_model(first, SimpleNamespace(source=None, route='r'), {'model': 'far'},
                                         prepared))['policy']
        assert 'sk-far-literal' not in str(committed) and 'sk-other-profile' not in str(committed)
        switched = restore_policy(committed)
        assert _resolve_policy_agent_runtime(first.runner, switched)[1]['api_key'] == 'sk-far-literal'
        # Owner restart: a fresh authority over the same durable receipt, nothing in memory.
        restarted = owner(db)
        assert _resolve_policy_agent_runtime(restarted.runner, switched)[1]['api_key'] == 'sk-far-literal'
        switch('back', old_url)
        again = await prepare_model(restarted, SimpleNamespace(source=None, route='r'), {'model': 'back'},
                                    {'snapshot': {'receipt': {'session_id': 'sid', 'policy': committed}}})
        back = restore_policy(again['policy'])
        assert back.base_url == old_url and 'api_key_from' not in back.config()['model']
