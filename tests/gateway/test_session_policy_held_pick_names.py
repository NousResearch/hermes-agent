"""A held pick only holds for the model name it recorded.

Regression: a profile-following session whose policy.model was inherited from an earlier pick on
another provider (e.g. a continued chat carrying ``grok-4.7``) but whose frozen ``model:`` block came
from a different route (anthropic) used to be treated as "held". ``effective()`` then paired the
inherited foreign model name with the frozen block's endpoint and posted one provider's model to
another's API (``grok-4.7`` -> ``api.anthropic.com`` -> 404 -> silent fallback). A committed ``/model``
records the name as the block's ``default``; a create-time pick records it as the request's ``model``.
Any other name was never chosen for this route, so the pick does not hold and the session follows the
profile. No live services, no network, no credentials.
"""
import json
from types import SimpleNamespace

from gateway.session_policy import LocalSessionPolicy, model_route

PROFILE = {'model': {'default': 'claude-opus-5-5', 'provider': 'anthropic',
                     'base_url': 'https://api.anthropic.com', 'api_mode': 'anthropic_messages'}}


def _authority(tmp_path):
    (tmp_path / 'config.yaml').write_text(json.dumps(PROFILE))
    return SimpleNamespace(db=SimpleNamespace(db_path=str(tmp_path / 'state.db')))


def _policy(tmp_path, model, block, request):
    return LocalSessionPolicy(source='tui', platform='tui', cwd=str(tmp_path), model=model, toolsets=(),
                              config_json=json.dumps({'model': block}),
                              request_json=json.dumps({'source': 'tui', 'follow_profile_config': True, **request}),
                              terminal_json=json.dumps({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': str(tmp_path)}))


def test_an_inherited_foreign_model_is_not_paired_with_the_profiles_route(tmp_path):
    # The live receipt: policy.model 'grok-4.7' inherited from an earlier xai pick, but the frozen
    # block is anthropic's and its selection_profile still matches the profile, so the old code held it.
    policy = _policy(tmp_path, 'grok-4.7',
                     {'default': 'claude-opus-5-5', 'provider': 'anthropic',
                      'selection_profile': ['claude-opus-5-5', 'anthropic']}, {})
    config, model = policy.effective(_authority(tmp_path), secrets=False)
    assert model == 'claude-opus-5-5'
    assert model_route(config['model'])[0] == 'anthropic'
    assert (model, model_route(config['model'])[0]) != ('grok-4.7', 'anthropic')


def test_a_committed_model_pick_still_holds_on_its_own_route(tmp_path):
    # What prepare_model writes for `/model grok-4.7` on xai: the block's default IS the policy model.
    policy = _policy(tmp_path, 'grok-4.7',
                     {'default': 'grok-4.7', 'provider': 'xai', 'base_url': 'https://api.x.ai/v1',
                      'api_mode': 'codex_responses', 'selection_profile': ['claude-opus-5-5', 'anthropic']}, {})
    config, model = policy.effective(_authority(tmp_path), secrets=False)
    provider, endpoint = model_route(config['model'])
    assert model == 'grok-4.7'
    assert provider == 'xai'
    assert 'api.x.ai' in endpoint


def test_a_create_time_model_pick_still_holds(tmp_path):
    # A create-time pick records the name as the request's `model`, on the profile's own route.
    policy = _policy(tmp_path, 'claude-sonnet-5',
                     {**PROFILE['model'], 'selection_profile': ['claude-opus-5-5', 'anthropic']},
                     {'model': 'claude-sonnet-5'})
    config, model = policy.effective(_authority(tmp_path), secrets=False)
    assert model == 'claude-sonnet-5'
    assert model_route(config['model'])[0] == 'anthropic'


def test_a_profile_edit_still_supersedes_a_committed_pick(tmp_path):
    # Same committed xai pick, but its selection_profile records a profile that has since moved (to the
    # anthropic route below): a later profile edit supersedes the pick and the session follows the profile.
    policy = _policy(tmp_path, 'grok-4.7',
                     {'default': 'grok-4.7', 'provider': 'xai', 'base_url': 'https://api.x.ai/v1',
                      'api_mode': 'codex_responses', 'selection_profile': ['grok-4.7', 'xai']}, {})
    config, model = policy.effective(_authority(tmp_path), secrets=False)
    assert model == 'claude-opus-5-5'
    assert model_route(config['model'])[0] == 'anthropic'
