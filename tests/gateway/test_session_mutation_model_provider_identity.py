"""A model pick freezes only the picked provider: an unrelated legacy entry it displaces keeps its
own secret, and a respelled ``providers:`` key replaces its stale spelling (real policy/resolver)."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest


async def _pick(tmp_path, monkeypatch, before, after, provider):
    """``(warm_config, warm_runtime, cold_config, cold_runtime, frozen_config_json)``."""
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, bind_launch_key, restore_policy
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps(before))
    private = {}
    policy = build_policy({'cwd': str(tmp_path), 'model': 'original'}, deepcopy(before), private_secrets=private)
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='review', epoch=1, profile_id='review', db=db)
        runner = SimpleNamespace(session_authority=authority)
        authority.runner = runner
        policy = bind_launch_key(authority, 'sid', policy, None, config_secrets=private)
        runner._resolve_session_agent_runtime = lambda **k: _resolve_policy_agent_runtime(runner, policy, fallback=False)
        (tmp_path / 'config.yaml').write_text(json.dumps(after))
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        committed = await prepare_model(authority, SimpleNamespace(source=None, route='r'),
                                        {'model': 'switched', 'provider': provider}, prepared)
        new = restore_policy(committed['policy'])
        warm_cfg = new.config(authority)
        _, warm = _resolve_policy_agent_runtime(runner, new, fallback=False)
        authority._local_config_secrets = {}  # a cold owner re-reads the profile's own paths
        cold_cfg = new.config(authority)
        _, cold = _resolve_policy_agent_runtime(runner, new, fallback=False)
        return warm_cfg, warm, cold_cfg, cold, json.loads(new.config_json)


def _entry(cfg, name):
    return next(e for e in cfg['custom_providers'] if isinstance(e, dict) and e.get('name') == name)


@pytest.mark.asyncio
async def test_displaced_unchanged_legacy_entry_keeps_its_key(tmp_path, monkeypatch):
    """Frozen [X]; the profile inserts Y before an unchanged X and the user picks Y: X (still a
    fallback) keeps its own key warm and cold, and Y runs with Y's key."""
    x = {'name': 'x-prov', 'base_url': 'http://127.0.0.1:9/x', 'api_key': 'x-review-key'}
    y = {'name': 'y-prov', 'base_url': 'http://127.0.0.1:9/y', 'api_key': 'y-review-key'}
    before = {'model': {'provider': 'custom:x-prov', 'default': 'original'}, 'custom_providers': [x],
              'fallback_providers': [{'provider': 'custom:x-prov', 'model': 'original'}]}
    after = {**before, 'custom_providers': [y, x]}
    warm_cfg, warm, cold_cfg, cold, _ = await _pick(tmp_path, monkeypatch, before, after, 'custom:y-prov')
    assert (warm['base_url'], warm['api_key']) == (y['base_url'], 'y-review-key')
    assert (cold['base_url'], cold['api_key']) == (y['base_url'], 'y-review-key')
    assert _entry(warm_cfg, 'x-prov')['api_key'] == 'x-review-key'
    assert _entry(cold_cfg, 'x-prov')['api_key'] == 'x-review-key'


@pytest.mark.asyncio
async def test_respelled_providers_key_replaces_the_stale_spelling(tmp_path, monkeypatch):
    """Frozen ``providers: {'Review Prov': A}``, profile now ``{'review-prov': B}``: the pick runs B
    and the stale spelling (and its secret) is gone from the frozen session."""
    before = {'model': {'provider': 'review-prov', 'default': 'original'},
              'providers': {'Review Prov': {'base_url': 'http://127.0.0.1:9/a', 'api_key': 'a-review-key',
                                            'models': {'switched': {}}}}}
    after = {**before, 'providers': {'review-prov': {'base_url': 'http://127.0.0.1:9/b',
                                                     'api_key': 'b-review-key', 'models': {'switched': {}}}}}
    warm_cfg, warm, cold_cfg, cold, frozen = await _pick(tmp_path, monkeypatch, before, after, 'review-prov')
    assert (warm['base_url'], warm['api_key']) == ('http://127.0.0.1:9/b', 'b-review-key')
    assert (cold['base_url'], cold['api_key']) == ('http://127.0.0.1:9/b', 'b-review-key')
    assert list(frozen['providers']) == ['review-prov']
    assert list(warm_cfg['providers']) == list(cold_cfg['providers']) == ['review-prov']
