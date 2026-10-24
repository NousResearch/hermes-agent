"""A following session (``follow_profile_config``) whose frozen config carries private secrets can
still pick a model.

Regression: ``prepare_model`` names the ``{old (section, index): new}`` secret-relocation map
``moved``, and the following branch shadowed that name with a tuple of route-bound key names. The
crash (``'tuple' object has no attribute 'get'``) only fires when the session has config secrets to
relocate — an inline ``api_key`` on a named/custom provider — which is why the suite's
empty-secret following picks never reached it. Every fresh Bot Chat / Desktop chat follows the
profile, so on a profile with an inline provider key every composer pick died as an ``internal
error`` (ws ``session.mutate`` dispatch crash).
"""
from dataclasses import asdict
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_a_following_session_with_config_secrets_can_pick_a_model(tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import bind_launch_key, build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch

    # The following read resolves the profile home through ``gateway.run`` (as the follow-profile
    # tests' ``home`` fixture does): point it at this test's profile, never the real one.
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    # The owning profile: an MCP server with a token (a private secret OUTSIDE the followed sections,
    # so a following session still carries it) beside a provider with an inline key.
    config = {'model': {'provider': 'openrouter', 'default': 'a'},
              'providers': {'mine': {'api_key': 'sk-mine', 'base_url': 'http://127.0.0.1:9/v1'}},
              'mcp_servers': {'notes': {'command': 'notes-mcp', 'env': {'NOTES_TOKEN': 'tok-notes'}}}}
    (tmp_path / 'config.yaml').write_text(
        'model:\n  provider: openrouter\n  default: a\n'
        'providers:\n  mine:\n    api_key: sk-mine\n    base_url: http://127.0.0.1:9/v1\n'
        'mcp_servers:\n  notes:\n    command: notes-mcp\n    env:\n      NOTES_TOKEN: tok-notes\n', encoding='utf-8')
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='b', target_provider='anthropic', provider_changed=True,
        base_url='https://api.anthropic.com'))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db, runner=SimpleNamespace(
            _resolve_session_agent_runtime=lambda **k: (None, {'api_key': 'sk-launch'})))
        private = {}
        policy = build_policy({'cwd': str(tmp_path), 'model': 'a', 'api_key': 'sk-launch',
                               'follow_profile_config': True}, config, private_secrets=private)
        policy = bind_launch_key(authority, 'sid', policy, 'sk-launch', config_secrets=private)
        assert policy.follows_profile
        assert policy.config_secret_ref, 'the fixture must carry config secrets for the regression to bite'
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}

        response = await prepare_model(authority, SimpleNamespace(source=None, route='r'),
                                       {'model': 'b', 'provider': 'anthropic'}, prepared)

        switched = restore_policy(response['policy'])
        assert switched.model == 'b'
        # The secret outside the followed sections rode along to the new route intact …
        assert switched.config(authority)['mcp_servers']['notes']['env']['NOTES_TOKEN'] == 'tok-notes'
        # … and the profile's own provider key still comes from the profile, never re-committed here.
        assert switched.config(authority)['providers']['mine']['api_key'] == 'sk-mine'
