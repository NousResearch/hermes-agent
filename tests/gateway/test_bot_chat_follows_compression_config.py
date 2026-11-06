"""Bot Chat follows compression settings, not frozen auxiliary routing or session pins."""
import json

import pytest

from tests.gateway.test_session_follow_profile_config import _owner


def _profile(home, compression, auxiliary):
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'default': 'profile-model'},
        'compression': compression, 'auxiliary': auxiliary,
    }), encoding='utf-8')


@pytest.mark.asyncio
@pytest.mark.parametrize('follows_profile', [True, False], ids=['follow', 'frozen'])
async def test_compression_edit_respects_follow_policy_and_keeps_auxiliary_frozen(monkeypatch, follows_profile):
    from gateway import run
    from gateway.session_policy import policy_for_source, policy_scope
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    before = {'enabled': True, 'threshold': 0.5, 'threshold_tokens': 10000,
              'model_thresholds': {'profile-model': 0.6}, 'target_ratio': 0.2,
              'protect_last_n': 4, 'tail_mode': 'assistant', 'in_place': False,
              'codex_responses_native': False, 'codex_responses_compact_threshold': 0.8}
    after = {**before, 'threshold': 0.7, 'threshold_tokens': 20000,
             'model_thresholds': {'profile-model': 0.75}, 'target_ratio': 0.3,
             'protect_last_n': 8, 'tail_mode': 'turn', 'in_place': True,
             'codex_responses_native': True, 'codex_responses_compact_threshold': 0.9}
    auxiliary = {'compression': {'provider': 'main', 'timeout': 30}}
    _profile(home, before, auxiliary)
    store, runner, authority, owner = _owner(home)
    try:
        response = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'create', 'source': 'gui', 'cwd': str(home), 'toolsets': [],
            'model': 'session-pick', 'reasoning': 'low', 'follow_profile_config': follows_profile,
        }})
        assert 'result' in response, response
        live = authority.sessions[response['result']['session_id']]
        policy = policy_for_source(runner, live.source)
        assert policy is not None
        with policy_scope(policy, authority=authority):
            initial = policy.config(authority)
            assert initial['compression'] == before
            _profile(home, after, {'compression': {'provider': 'main', 'timeout': 90}})
            # One profile read per turn: edits apply on the next scope, not halfway through a turn.
            assert policy.config(authority)['compression'] == before
        with policy_scope(policy, authority=authority):
            current = policy.config(authority)
            assert current['auxiliary'] == auxiliary
            assert policy.current_model(authority) == 'session-pick'
            assert current['agent']['reasoning_effort'] == 'low'
            assert policy.current_toolsets(current) == ()
            assert current['compression'] == (after if follows_profile else before)
            initial_signature = run.GatewayRunner._extract_cache_busting_config(initial)
            current_signature = run.GatewayRunner._extract_cache_busting_config(current)
            assert (initial_signature != current_signature) is follows_profile
        # Removing the section must drop followed overrides instead of reviving the launch snapshot.
        _profile(home, None, auxiliary)
        with policy_scope(policy, authority=authority):
            assert policy.config(authority).get('compression') == (None if follows_profile else before)
    finally:
        await owner.close()
        store.close_all_db_handles()
