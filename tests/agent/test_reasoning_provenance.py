"""Session reasoning provenance through gateway, fallback, and route restoration.

Regression coverage for four defects, each of which lost an explicit reasoning
choice:

* a fallback route's re-resolution replaced the session's own setting, and
  restoring the primary route resurrected a stale snapshot over a later edit
  (including an equal value, ``None``, and a plain attribute assignment);
* a model switch re-resolved reasoning from the process profile instead of the
  session policy bound to the turn;
* a delegated child inherited the parent's copied policy instead of using its
  own (or its process config);
* a voice turn's effort override hid the session's edit, and a fallback that
  activated during the voice turn restored over it;
* the gateway's per-turn reasoning assignment recorded the caller's live dict as
  the session edit, so a later in-place mutation rewrote the record (the setter
  now snapshots it, and the runner's dead pre-write is gone);
* a primary-runtime restore with no fallback markers (a one-turn ``/model --once``
  restore, an init-time fallback) let the snapshot's older value overwrite the
  session's own choice.

The fixes are unconditional: no feature flag, no classifier.
"""
import copy
import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import build_api_kwargs, _reresolve_fallback_reasoning_config
from gateway.session_policy import build_policy, pin_reasoning_effort, unpin_reasoning
from hermes_constants import ProfileReasoningConfig, resolve_reasoning_config
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve
from tests.agent.test_run_agent_codex_responses import _build_agent

PROFILE = {
    "model": {"default": "gpt-6.1-terra", "provider": "openai-codex"},
    "agent": {"reasoning_effort": "medium",
              "reasoning_overrides": {"openai-codex/gpt-6.1-terra": "high",
                                      "opencode-go/deepseek-v4.1-flash": "high"}},
}


# ── gateway pins are explicit, not profile baselines ──────────────────────────

@pytest.mark.parametrize('following', [False, True])
@pytest.mark.parametrize('pin', ['high', 'none'])
def test_gateway_launch_pin_is_explicit_on_wire_and_fallback(tmp_path, monkeypatch, following, pin):
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'gpt-6.1-terra',
               'reasoning': pin, 'follow_profile_config': following}
    policy = build_policy(request, PROFILE, profile_terminal=False)
    agent = _build_agent(monkeypatch)
    agent.model, agent.provider, agent.api_mode = 'gpt-6.1-terra', 'openai-codex', 'codex_responses'
    agent.reasoning_config = policy.current_reasoning_config(None)
    agent.request_overrides = None
    explicit = agent.reasoning_config.copy()
    assert type(agent.reasoning_config) is dict
    before = build_api_kwargs(agent, [{'role': 'user', 'content': 'hi'}], [])
    with patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        _reresolve_fallback_reasoning_config(agent)
    after = build_api_kwargs(agent, [{'role': 'user', 'content': 'hi'}], [])
    assert agent.reasoning_config == explicit
    assert before.get('reasoning') == after.get('reasoning')
    assert before.get('reasoning', {}).get('effort') != 'low'
    if pin == 'high':
        assert before['reasoning']['effort'] == 'high'


@pytest.mark.parametrize('following', [False, True])
def test_committed_pin_and_reset_provenance(tmp_path, following):
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'gpt-6.1-terra',
               'reasoning': 'high', 'follow_profile_config': following}
    policy = build_policy(request, PROFILE, profile_terminal=False)
    cfg = policy.config()
    unpin_reasoning(cfg, PROFILE)
    policy = replace(policy, config_json=json.dumps(cfg))
    assert isinstance(policy.current_reasoning_config(None), ProfileReasoningConfig)
    pin_reasoning_effort(cfg, 'high', session=True)
    policy = replace(policy, config_json=json.dumps(cfg))
    assert type(policy.current_reasoning_config(None)) is dict


# ── fix 1: an explicit edit survives fallback and primary restoration ─────────

@pytest.mark.parametrize('new_value', [{'enabled': False}, {'enabled': True, 'effort': 'low'}, None, {}])
def test_edit_during_fallback_survives_restore(new_value):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        agent.reasoning_config = new_value
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == new_value


@pytest.mark.parametrize('reuse_object', [False, True])
@pytest.mark.parametrize('second_fallback', [False, True])
def test_equal_assignment_during_fallback_is_a_session_edit(reuse_object, second_fallback):
    """A fresh value equal to the fallback's own is still a deliberate assignment."""
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config({'agent': {'reasoning_effort': 'high'}}, agent.model)
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    config = {'agent': {'reasoning_effort': 'medium'}}
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        fresh = agent.reasoning_config if reuse_object else resolve_reasoning_config(config, 'primary')
        agent.reasoning_config = fresh
        if second_fallback:
            assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'fallback-2'},
                                    'openrouter', 'fallback-2') is not None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == fresh
    assert type(agent.reasoning_config) is type(fresh)


def test_reassigning_none_during_fallback_is_a_session_edit():
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config({'agent': {'reasoning_effort': 'high'}}, agent.model)
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config is None
        agent.reasoning_config = None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


def test_unmodified_fallback_restores_primary_profile_provenance():
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4'})
    primary = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    agent._primary_runtime['reasoning_config'] = primary
    agent.reasoning_config = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'medium'
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert isinstance(agent.reasoning_config, ProfileReasoningConfig)
    assert agent.reasoning_config == primary


@pytest.mark.parametrize('fresh', ['medium', 'low', None, 'disabled'])
def test_cached_gateway_turn_setting_survives_fallback_restoration(fresh):
    """The gateway re-assigns the turn's own setting on a reused (fallback) agent."""
    from tests.gateway.test_display_null_turn_wiring import _wire
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    config = {'agent': {'reasoning_effort': 'medium'}}
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'medium'
    if fresh == 'disabled':
        value = {'enabled': False}
    elif fresh is None:
        value = None
    else:
        value = resolve_reasoning_config({'agent': {'reasoning_effort': fresh}}, 'gpt-6.1-terra')
    _wire({}, agent=agent, reasoning_config=value)
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == value
    assert type(agent.reasoning_config) is type(value)


@pytest.mark.parametrize('before_fallback', [False, True])
def test_edit_survives_multiple_fallbacks(before_fallback):
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    if before_fallback:
        agent.reasoning_config = {'enabled': False}
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        if not before_fallback:
            agent.reasoning_config = {'enabled': False}
        assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'model-b'},
                                'openrouter', 'model-b') is not None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == {'enabled': False}


def test_none_edit_between_fallbacks_is_not_replaced_by_fallback_default():
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        agent.reasoning_config = None
        assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'model-b'},
                                'openrouter', 'model-b') is not None
        assert agent.reasoning_config is None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


@pytest.mark.parametrize('choice', ['none', 'same-baseline', 'new-baseline', 'in-place'])
def test_nested_fallback_keeps_explicit_session_edit_on_wire(choice):
    from agent.route_binding import bind_route_entry
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    config = {'agent': {'reasoning_effort': 'low',
                        'reasoning_overrides': {'openrouter/model-b': 'high',
                                                'openrouter/model-c': 'high'}}}
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(config, 'primary')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        if choice == 'none':
            agent.reasoning_config = None
        elif choice == 'same-baseline':
            agent.reasoning_config = agent.reasoning_config
        elif choice == 'new-baseline':
            agent.reasoning_config = resolve_reasoning_config(config, 'primary')
        else:
            agent.reasoning_config['effort'] = 'medium'
        expected = copy.deepcopy(agent.reasoning_config)
        for model in ('model-b', 'model-c'):
            assert bind_route_entry(agent, {'provider': 'openrouter', 'model': model}, 'openrouter', model)
            assert agent.reasoning_config == expected
            assert type(agent.reasoning_config) is type(expected)
            wire = _reasoning_config_for_wire(agent)
            if expected is not None:
                assert wire == expected
            else:
                assert wire is None or wire.get('effort') != 'high'
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == expected


def test_unedited_nested_fallback_still_resolves_its_own_profile():
    from agent.route_binding import bind_route_entry
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    config = {'agent': {'reasoning_effort': 'low',
                        'reasoning_overrides': {'openrouter/model-b': 'high'}}}
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    agent.reasoning_config = resolve_reasoning_config(config, 'primary')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'model-b'}, 'openrouter', 'model-b')
        assert agent.reasoning_config['effort'] == 'high'
        assert _reasoning_config_for_wire(agent)['effort'] == 'high'


@pytest.mark.parametrize('snapshot_key', [False, True])
def test_unset_primary_restored_after_real_fallback(snapshot_key):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = None
    if snapshot_key:
        agent._primary_runtime['reasoning_config'] = None
    else:
        agent._primary_runtime.pop('reasoning_config', None)
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'high'}}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'high'
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


@pytest.mark.parametrize("second_fallback", [False, True])
def test_in_place_edit_during_fallback_survives_primary_restore(second_fallback):
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}):
        assert agent._try_activate_fallback()
        agent.reasoning_config["effort"] = "low"
        if second_fallback:
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "fallback-2"},
                                    "openrouter", "fallback-2") is not None
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config["effort"] == "low"


def test_empty_switch_snapshot_preserves_explicit_mapping():
    from agent.primary_runtime_snapshot import _build_primary_runtime_snapshot
    agent = _make_agent()
    agent.reasoning_config = {}
    snapshot = _build_primary_runtime_snapshot(agent, agent.api_mode)
    assert snapshot['reasoning_config'] == {}
    assert snapshot['reasoning_config'] is not agent.reasoning_config


# ── fix 2: model switches resolve from the bound session policy ──────────────

@pytest.mark.parametrize('explicit,expected', [(True, 'low'), (False, 'xhigh')])
def test_switch_uses_bound_policy_not_process_config(explicit, expected):
    agent = _make_agent()
    process = {'agent': {'reasoning_effort': 'high'}}
    policy = {'agent': {'reasoning_effort': 'medium',
                        'reasoning_overrides': {'new-model': 'xhigh'}}}
    agent.reasoning_config = ({'enabled': True, 'effort': 'low'} if explicit
                              else resolve_reasoning_config(process, 'old-model'))
    with patch('gateway.session_policy.turn_policy_config', return_value=policy), \
         patch('hermes_cli.config.load_config', return_value=process), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        agent.switch_model('new-model', 'custom', api_key='sk-test',
                           base_url='https://my-llm.example.com/v1')
    assert agent.reasoning_config['effort'] == expected


def test_switch_ends_stale_fallback_reasoning_markers():
    agent = _make_agent()
    agent._fallback_reasoning_at_bind = {'enabled': True, 'effort': 'medium'}
    agent._fallback_reasoning_primary = {'enabled': True, 'effort': 'high'}
    agent._fallback_reasoning_session_edit = ({'enabled': False},)
    with patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'high'}}), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        agent.switch_model('new-model', 'custom', api_key='sk-test',
                           base_url='https://my-llm.example.com/v1')
    assert '_fallback_reasoning_at_bind' not in vars(agent)
    assert '_fallback_reasoning_primary' not in vars(agent)
    assert '_fallback_reasoning_session_edit' not in vars(agent)
    assert agent._primary_runtime['reasoning_config']['effort'] == 'high'


@pytest.mark.parametrize('following', [False, True])
def test_gateway_loader_passes_effective_model_to_local_policy(tmp_path, following):
    from types import SimpleNamespace
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    config = {'model': {'default': 'model-a', 'provider': 'openrouter'},
              'agent': {'reasoning_effort': 'medium',
                        'reasoning_overrides': {'model-a': 'high', 'model-b': 'low'}}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'model-a',
                           'follow_profile_config': following}, config, profile_terminal=False)
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *_: 'route',
                             _peek_session_state=lambda *_: None)
    with patch('gateway.session_setting_commands.local_policy', return_value=policy), \
         patch('gateway.session_authorities.active_authority', return_value=None):
        resolved = GatewayConfigLoadersMixin._resolve_session_reasoning_config(runner, model='model-b')
    assert resolved['effort'] == 'low'


@pytest.mark.parametrize('following', [False, True])
@pytest.mark.parametrize('pin', [None, 'low'])
def test_policy_uses_actual_turn_model_not_frozen_model(tmp_path, following, pin):
    config = {'model': {'default': 'model-a', 'provider': 'openrouter'},
              'agent': {'reasoning_effort': 'medium',
                        'reasoning_overrides': {'model-a': 'high', 'model-b': 'xhigh'}}}
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'model-a',
               'follow_profile_config': following}
    if pin:
        request['reasoning'] = pin
    policy = build_policy(request, config, profile_terminal=False)
    with patch('gateway.session_authorities.active_authority', return_value=None):
        assert policy.current_reasoning_config(None, model='model-a')['effort'] == (pin or 'high')
        assert policy.current_reasoning_config(None, model='model-b')['effort'] == (pin or 'xhigh')
        assert policy.current_reasoning_config(None, model='model-a')['effort'] == (pin or 'high')


# ── fix 3: a delegated child does not inherit a copied parent policy ─────────

def test_delegated_child_ignores_copied_parent_policy(tmp_path):
    from agent.delegation_context import delegated_child_context
    from gateway.session_policy import policy_scope, turn_policy_config
    parent_cfg = {'agent': {'reasoning_effort': 'high'}}
    child_cfg = {'agent': {'reasoning_effort': 'medium',
                           'reasoning_overrides': {'new-model': 'xhigh', 'fallback': 'xhigh'}}}
    process = {'agent': {'reasoning_effort': 'low'}}
    request = {'source': 'cli', 'cwd': str(tmp_path)}
    parent = build_policy(request, parent_cfg, profile_terminal=False)
    child = build_policy(request, child_cfg, profile_terminal=False)
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = resolve_reasoning_config(child_cfg, agent.model)
    with patch('hermes_cli.config.load_config', return_value=process), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         policy_scope(parent):
        assert turn_policy_config() == parent.config()
        with delegated_child_context('child'):
            # The parent's policy was copied into this context, not bound here.
            assert turn_policy_config() is None
            with policy_scope(child):
                assert turn_policy_config() == child.config()
                agent.reasoning_config = resolve_reasoning_config(child_cfg, agent.model)
                agent.switch_model('new-model', 'custom', api_key='sk-test',
                                   base_url='https://my-llm.example.com/v1')
                assert agent.reasoning_config['effort'] == 'xhigh'
                assert agent._try_activate_fallback()
                assert agent.reasoning_config['effort'] == 'xhigh'
            assert turn_policy_config() is None
        assert turn_policy_config() == parent.config()


def test_delegation_worker_copies_parent_but_accepts_child_binding(tmp_path):
    from gateway.session_policy import policy_scope, turn_policy_config
    from tests.tools.test_delegate import _make_mock_parent
    from tests.tools.test_delegate_child_transcript_release import _fake_child
    from tools.delegate_tool import _run_single_child
    cfg = {'agent': {'reasoning_effort': 'high'}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path)}, cfg, profile_terminal=False)
    child = _fake_child([])
    child._delegate_profile_home = None
    seen = []

    def run_child(**kwargs):
        seen.append(turn_policy_config())
        # Even rebinding the very same policy is a new scope owned by this child.
        with policy_scope(policy):
            seen.append(turn_policy_config())
        seen.append(turn_policy_config())
        return {'final_response': 'ok', 'completed': True, 'api_calls': 1, 'messages': []}

    child.run_conversation.side_effect = run_child
    with patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'low'}}), \
         policy_scope(policy):
        entry = _run_single_child(task_index=0, goal='work', child=child, parent_agent=_make_mock_parent())
        assert turn_policy_config() == policy.config()
    assert seen[0] is None
    assert seen[1] == policy.config()
    assert seen[2] is None
    assert entry['summary'] == 'ok'


# ── fix 4: voice turns retain and restore explicit edits ─────────────────────

@pytest.mark.parametrize('existing', [False, True])
def test_voice_route_restores_fallback_tracking(existing):
    from agent.voice_turn_route import _capture, _reinstall
    agent = _make_agent()
    if existing:
        agent._fallback_reasoning_at_bind = {'enabled': False}
        agent._fallback_reasoning_primary = None
    before = {key: value for key, value in vars(agent).items() if key.startswith('_fallback_reasoning_')}
    state = _capture(agent)
    agent._fallback_reasoning_at_bind = {'enabled': True}
    agent._fallback_reasoning_primary = {'effort': 'high'}
    agent._fallback_reasoning_session_edit = ({},)
    with patch('agent.route_binding.reinstall_runtime_snapshot'):
        _reinstall(agent, state)
    after = {key: value for key, value in vars(agent).items() if key.startswith('_fallback_reasoning_')}
    assert after == before


def test_voice_effort_only_restoration_does_not_create_session_edit():
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent()
    agent._fallback_reasoning_at_bind = {'enabled': True, 'effort': 'medium'}
    agent.reasoning_config = {'enabled': True, 'effort': 'medium'}
    agent.__dict__.pop('_fallback_reasoning_session_edit', None)
    agent._voice_turn_pending = True
    with patch('agent.auxiliary_task_config._get_auxiliary_task_config', return_value={'reasoning_effort': 'low'}), \
         patch('agent.voice_turn_route._route_target', return_value=None):
        begin_voice_turn_route(agent, [], None)
        end_voice_turn_route(agent)
    assert '_fallback_reasoning_session_edit' not in vars(agent)


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("edit", [{"enabled": False}, None, {"enabled": True, "effort": "low"}])
def test_session_edit_during_voice_on_fallback_survives_all_restores(separate_voice, edit):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "none"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = edit
        end_voice_turn_route(agent)
        assert agent.reasoning_config == edit
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == edit


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("mode", ["unchanged", "in_place", "nested_fallback"])
def test_voice_override_does_not_hide_or_invent_session_edits(separate_voice, mode):
    from agent.route_binding import bind_route_entry
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        if mode == "in_place":
            agent.reasoning_config["effort"] = "xhigh"
        elif mode == "nested_fallback":
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "voice-fallback"},
                                    "openrouter", "voice-fallback") is not None
        end_voice_turn_route(agent)
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config["effort"] == ("xhigh" if mode == "in_place" else "high")


@pytest.mark.parametrize("edit", [None, {}, {"enabled": False}, {"enabled": True, "effort": "low"}])
@pytest.mark.parametrize("mutate", [False, True])
def test_same_route_voice_edit_without_fallback(edit, mutate):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent()
    agent.reasoning_config = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    with patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=None):
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = edit.copy() if isinstance(edit, dict) else edit
        if mutate and isinstance(edit, dict):
            agent.reasoning_config["custom"] = "edited"
        end_voice_turn_route(agent)
    expected = {**edit, "custom": "edited"} if mutate and isinstance(edit, dict) else edit
    assert agent.reasoning_config == expected
    assert type(agent.reasoning_config) is type(expected)


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("prior_fallback", [False, True])
def test_voice_assignment_then_in_place_edit_survives_nested_fallback(separate_voice, prior_fallback):
    from agent.route_binding import bind_route_entry
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        if prior_fallback:
            assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent.reasoning_config["effort"] = "xhigh"
        assert bind_route_entry(agent, {"provider": "openrouter", "model": "voice-fallback"},
                                "openrouter", "voice-fallback") is not None
        end_voice_turn_route(agent)
        assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
        if prior_fallback:
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("prior_fallback", [False, True])
@pytest.mark.parametrize("nested_fallback", [False, True])
@pytest.mark.parametrize("edit_mode", ["none", "assign", "mutate_to_marker"])
def test_voice_provenance_across_full_primary_restore(separate_voice, prior_fallback, nested_fallback, edit_mode):
    from agent.route_binding import bind_route_entry
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route, session_runtime_view

    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        if prior_fallback:
            assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        if edit_mode != "none":
            agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
            if edit_mode == "mutate_to_marker":
                agent.reasoning_config["effort"] = "low"
            assert session_runtime_view(agent).reasoning_config == agent.reasoning_config
        if nested_fallback:
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "voice-fallback"},
                                    "openrouter", "voice-fallback") is not None
            if edit_mode != "none":
                assert session_runtime_view(agent).reasoning_config["effort"] == (
                    "low" if edit_mode == "mutate_to_marker" else "xhigh")
        end_voice_turn_route(agent)
        if agent._fallback_activated:
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config["effort"] == {
        "none": "high", "assign": "xhigh", "mutate_to_marker": "low",
    }[edit_mode]


# ── fix 5: an explicit reasoning config needs no capability probe ────────────

def test_explicit_ollama_reasoning_does_not_probe_capabilities():
    from tests.agent.test_unset_reasoning_effort_default import _Agent
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    agent = _Agent(reasoning_config={'enabled': True, 'effort': 'high'})
    agent.api_mode = 'chat_completions'
    agent._ollama_num_ctx = 4096
    agent.reasoning_config = {'enabled': True, 'effort': 'high'}
    agent._ollama_supports_thinking_cached = MagicMock(side_effect=AssertionError('unexpected probe'))
    assert _reasoning_config_for_wire(agent) == agent.reasoning_config
    agent._ollama_supports_thinking_cached.assert_not_called()


# ── fix 6: the recorded session edit is a snapshot, not the caller's object ──

@pytest.mark.parametrize('value', [{'enabled': True, 'effort': 'high'}, {}, None])
def test_reasoning_setter_records_a_snapshot_of_the_assigned_value(value):
    """The setter is the single place a fallback-bound agent records a session edit."""
    agent = _make_agent()
    agent._fallback_reasoning_at_bind = {'enabled': True, 'effort': 'medium'}
    agent.reasoning_config = value
    recorded = agent._fallback_reasoning_session_edit[0]
    assert recorded == value
    assert recorded is not value or value is None
    if isinstance(value, dict):
        value['custom'] = 'mutated by the caller'
    assert agent._fallback_reasoning_session_edit[0] == recorded


def test_gateway_turn_assignment_records_a_snapshot_not_the_turn_dict():
    """``run_turn_runner`` hands the setter the turn's live policy dict every turn.

    The runner used to pre-write the marker and then have the setter overwrite it
    with that live dict, so mutating the turn's policy rewrote the recorded edit.
    """
    from tests.gateway.test_display_null_turn_wiring import _wire
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
    turn = {'enabled': True, 'effort': 'high'}
    _wire({}, agent=agent, reasoning_config=turn)
    recorded = agent._fallback_reasoning_session_edit
    assert recorded is not None
    assert recorded[0] == turn
    assert recorded[0] is not turn
    turn['effort'] = 'low'
    assert recorded[0] == {'enabled': True, 'effort': 'high'}


# ── fix 7: a missing provenance marker must not resurrect the snapshot ───────

def _activate_fallback_with_markers(agent, *, keep_markers):
    """Bind a fallback, then optionally drop the markers while it stays active."""
    primary = resolve_reasoning_config({'agent': {'reasoning_effort': 'medium'}}, agent.model)
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
    if not keep_markers:
        for name in ('_fallback_reasoning_at_bind', '_fallback_reasoning_primary',
                     '_fallback_reasoning_session_edit'):
            vars(agent).pop(name, None)


@pytest.mark.parametrize('keep_markers', [False, True])
def test_turn_choice_survives_restore_without_provenance_markers(keep_markers):
    """Init-time fallbacks and end-of-voice marker restores leave no markers."""
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    _activate_fallback_with_markers(agent, keep_markers=keep_markers)
    choice = resolve_reasoning_config({'agent': {'reasoning_effort': 'high'}}, 'gpt-6.1-terra')
    agent.reasoning_config = choice
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == choice
    assert type(agent.reasoning_config) is type(choice)


def test_restore_without_markers_keeps_an_unchanged_session_value():
    """No markers, and the value already matches the snapshot: nothing to protect."""
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    _activate_fallback_with_markers(agent, keep_markers=False)
    saved = agent._primary_runtime['reasoning_config']
    agent.reasoning_config = copy.deepcopy(saved)
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == saved


def test_one_turn_model_override_restore_keeps_pre_override_reasoning():
    """Reachable production path: the ``/model ... --once`` restore.

    ``hermes_cli.cli_model_switch_mixin._restore_model_runtime_snapshot`` assigns the
    session's pre-override ``reasoning_config`` and THEN forces ``_fallback_activated``
    with an older ``_primary_runtime`` and no fallback markers — a one-turn override never
    binds through ``bind_route_entry``. Its TUI twin (``tui_gateway/model_switch.py``)
    re-assigns the value after a successful restore to paper over this clobber.
    """
    agent = _make_agent()
    stale = resolve_reasoning_config({'agent': {'reasoning_effort': 'medium'}}, agent.model)
    session = {'enabled': True, 'effort': 'high'}
    assert stale != session
    agent._primary_runtime['reasoning_config'] = stale
    agent.reasoning_config = session             # pre-restore assignment, as the CLI does
    agent._fallback_activated = True             # forced by the one-turn restore
    assert not any(name in vars(agent) for name in
                   ('_fallback_reasoning_at_bind', '_fallback_reasoning_primary',
                    '_fallback_reasoning_session_edit'))
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == session


# ── provenance class: conservative by design, documented here ───────────────

PROFILE_RERESOLVE = {'agent': {'reasoning_effort': 'medium',
                               'reasoning_overrides': {'openrouter/fallback-model': 'low'}}}


def test_baseline_class_is_the_fallback_reresolve_signal():
    """Only a profile-baseline object may be re-resolved for a fallback route.

    The signal is a Python type, and a JSON round trip loses it. That asymmetry is
    deliberate and fails CLOSED: a value whose provenance is unknown (a JSON
    round-tripped plain ``dict``) is kept as the session's own decision instead
    of being silently re-derived. Copying preserves the class, so nothing else
    loses the signal.
    """
    fresh = resolve_reasoning_config(PROFILE_RERESOLVE, 'primary-model')
    resumed = json.loads(json.dumps(fresh))
    assert type(fresh) is ProfileReasoningConfig

    assert type(resumed) is dict
    assert resumed == fresh
    for clone in (copy.deepcopy(fresh), copy.copy(fresh)):
        assert type(clone) is ProfileReasoningConfig
        assert clone.is_baseline()

    def after_reresolve(value):
        agent = _make_agent()
        agent.model, agent.provider = 'fallback-model', 'openrouter'
        agent.reasoning_config = value
        with patch('hermes_cli.config.load_config', return_value=PROFILE_RERESOLVE):
            _reresolve_fallback_reasoning_config(agent)
        return agent.reasoning_config

    fresh_after = after_reresolve(fresh)
    assert fresh_after['effort'] == 'low'
    assert type(fresh_after) is ProfileReasoningConfig
    # A plain dict is the session's value: never replaced, and never upgraded to
    # the baseline class (which would let the NEXT route change replace it).
    resumed_after = after_reresolve(resumed)
    assert resumed_after == resumed
    assert type(resumed_after) is dict


def test_legacy_persisted_baseline_and_explicit_choice_are_indistinguishable(tmp_path):
    """Legacy adoption pins both; it cannot safely infer an unrecorded origin."""
    from gateway.session_local_migration import legacy_launch_params
    baseline = resolve_reasoning_config(PROFILE_RERESOLVE, 'primary-model')
    explicit = dict(baseline)
    params_list = []
    for value in (baseline, explicit):
        from types import SimpleNamespace
        row = SimpleNamespace(source='cli', model='primary-model', cwd=str(tmp_path),
                              model_config=json.loads(json.dumps({'reasoning_config': value})),
                              billing_provider=None, billing_base_url=None)
        params, _ = legacy_launch_params(row)
        params_list.append(params)
        policy = build_policy(params, PROFILE_RERESOLVE, profile_terminal=False)
        actual = policy.current_reasoning_config(None)
        assert type(actual) is dict
        agent = _make_agent()
        agent.model, agent.provider, agent.reasoning_config = 'fallback-model', 'openrouter', actual
        with patch('hermes_cli.config.load_config', return_value=PROFILE_RERESOLVE):
            _reresolve_fallback_reasoning_config(agent)
        assert agent.reasoning_config == explicit
    assert params_list[0] == params_list[1]


@pytest.mark.parametrize("nested", [False, True])
def test_restored_explicit_none_survives_later_fallbacks(nested):
    from agent.route_binding import bind_route_entry

    cfg = {"agent": {"reasoning_effort": "low",
                     "reasoning_overrides": {"openrouter/model-b": "high"}}}
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "model-a"})
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = resolve_reasoning_config(cfg, "primary")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        agent.reasoning_config = None
        assert agent._restore_primary_runtime()
        for _ in range(2):
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "model-b"},
                                    "openrouter", "model-b")
            assert agent.reasoning_config is None
            if nested:
                assert bind_route_entry(agent, {"provider": "openrouter", "model": "model-a"},
                                        "openrouter", "model-a")
                assert agent.reasoning_config is None
            assert agent._restore_primary_runtime()
            assert agent.reasoning_config is None
        # A new profile baseline replaces the prior explicit None.
        agent.reasoning_config = resolve_reasoning_config(cfg, "primary")
        assert bind_route_entry(agent, {"provider": "openrouter", "model": "model-b"},
                                "openrouter", "model-b")
        assert agent.reasoning_config["effort"] == "high"


def test_unset_profile_none_still_resolves_for_fallback():
    from agent.route_binding import bind_route_entry

    cfg = {"agent": {"reasoning_overrides": {"openrouter/model-b": "high"}}}
    agent = _make_agent()
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = None
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        for _ in range(2):
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "model-b"},
                                    "openrouter", "model-b")
            assert agent.reasoning_config["effort"] == "high"
            assert agent._restore_primary_runtime()
            assert agent.reasoning_config is None


@pytest.mark.parametrize("nested", [False, True])
def test_assigned_profile_mutated_to_bind_value_survives_restore(nested):
    from agent.route_binding import bind_route_entry

    cfg = {"agent": {"reasoning_effort": "low"}}
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "model-a"})
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = resolve_reasoning_config(cfg, "primary")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        previous = agent.reasoning_config.copy()
        assigned = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, "primary")
        agent.reasoning_config = assigned
        assigned.clear()
        assigned.update(previous)
        if nested:
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "model-b"},
                                    "openrouter", "model-b")
            assert agent.reasoning_config == previous
        assert agent._restore_primary_runtime()
        assert agent.reasoning_config == previous
        assert type(agent.reasoning_config) is type(assigned)
