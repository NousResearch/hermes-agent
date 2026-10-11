"""``/model <alias>`` keeps the alias endpoint's own credential on the next turn."""
import pytest


@pytest.mark.asyncio
async def test_model_switch_to_a_keyed_direct_alias_runs_the_next_turn_with_its_own_key(tmp_path, monkeypatch):
    """A direct alias declares its endpoint's credential (``key_env`` or a literal ``api_key``). The
    route change drops the old endpoint's credentials; the alias's own must take their place in the
    committed selection, or the next turn sends ``no-key-required`` to the alias host."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import bind_launch_key, build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch
    old_url, far_url = 'http://127.0.0.1:9/v1', 'http://127.0.0.3:9/v1'
    monkeypatch.setenv('FAR_ALIAS_KEY', 'sk-far-env')
    monkeypatch.setitem(model_switch.DIRECT_ALIASES, 'far-env', model_switch.DirectAlias(
        'far-model', 'custom', far_url, key_env='FAR_ALIAS_KEY'))
    monkeypatch.setitem(model_switch.DIRECT_ALIASES, 'far-literal', model_switch.DirectAlias(
        'far-model', 'custom', far_url, api_key='sk-far-literal'))
    monkeypatch.setitem(model_switch.DIRECT_ALIASES, 'far-ref', model_switch.DirectAlias(
        'far-model', 'custom', far_url, api_key='${FAR_ALIAS_KEY}'))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        runner = SimpleNamespace(session_authority=authority,
                                 _resolve_session_agent_runtime=lambda **k: (None, {'api_key': 'sk-endpoint-a'}))
        authority.runner = runner
        config = {'model': {'provider': 'custom', 'default': 'a', 'base_url': old_url, 'key_env': 'ENDPOINT_A_KEY'}}
        private = {}
        policy = build_policy({'cwd': str(tmp_path), 'model': 'a', 'provider': 'custom', 'base_url': old_url,
                               'api_key': 'sk-endpoint-a'}, config, private_secrets=private)
        policy = bind_launch_key(authority, 'sid', policy, 'sk-endpoint-a', config_secrets=private)
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        # Every credential shape a direct alias can declare (DirectAlias / direct_alias_api_key).
        for alias, expected in (('far-env', 'sk-far-env'), ('far-literal', 'sk-far-literal'),
                                ('far-ref', 'sk-far-env')):
            monkeypatch.setattr(model_switch, 'switch_model', lambda alias=alias, **k: model_switch.ModelSwitchResult(
                success=True, new_model='far-model', target_provider='custom', base_url=far_url,
                api_key='resolved-at-switch', resolved_via_alias=alias))
            switched = restore_policy((await prepare_model(authority, SimpleNamespace(source=None, route='r'),
                                                           {'model': alias}, prepared))['policy'])
            assert switched.credential_ref is None, "endpoint A's launch key crossed to the alias host"
            assert 'sk-far-literal' not in switched.config_json, 'a literal alias key must stay private'
            runtime = _resolve_policy_agent_runtime(runner, switched)[1]
            assert runtime['base_url'].rstrip('/') == far_url
            assert runtime['api_key'] == expected, f'{alias}: the next turn ran without the alias key'
