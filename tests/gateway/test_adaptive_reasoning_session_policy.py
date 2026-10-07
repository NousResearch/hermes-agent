"""What decides ``agent.adaptive_reasoning`` for an owner-run turn: the session's policy (its config and
whether it chose its level), the turn's own level (one-turn pick, API request), and a Kanban attempt.
The daemon-level behaviour is in ``test_adaptive_reasoning_local_turn.py``."""
import json
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner

ADAPTIVE = {'enabled': True, 'max_effort': 'high'}


def _policy(tmp_path, config=None, **params):
    from gateway.session_policy import build_policy
    return build_policy({'cwd': str(tmp_path), 'source': 'gui', **params}, config or {})


def test_reasoning_chosen_tracks_the_launch_level_a_session_pick_and_a_reset(tmp_path):
    from dataclasses import replace

    from gateway.session_policy import pin_reasoning_effort, unpin_reasoning
    inherited = _policy(tmp_path, {'agent': {'reasoning_effort': 'medium'}})
    assert inherited.reasoning_chosen() is False
    assert _policy(tmp_path, reasoning='low').reasoning_chosen() is True
    config = inherited.config()
    pin_reasoning_effort(config, 'low', session=True)  # a committed session /reasoning low
    picked = replace(inherited, config_json=json.dumps(config))
    assert picked.reasoning_chosen() is True
    unpin_reasoning(config, {'agent': {'reasoning_effort': 'medium'}})  # /reasoning reset
    assert replace(inherited, config_json=json.dumps(config)).reasoning_chosen() is False
    # A reset also gives back a create-time level.
    launched = _policy(tmp_path, reasoning='low')
    config = launched.config()
    unpin_reasoning(config, {})
    assert replace(launched, config_json=json.dumps(config)).reasoning_chosen() is False


def test_following_session_reads_the_profiles_current_adaptive_policy(tmp_path, monkeypatch):
    """A Bot Chat (``follow_profile_config``) re-reads ``agent.*`` from its profile every turn: turning
    ``adaptive_reasoning`` on or off in config.yaml reaches its next turn, like ``reasoning_effort``."""
    from gateway import session_policy
    policy = _policy(tmp_path, {'agent': {'reasoning_effort': 'medium', 'adaptive_reasoning': ADAPTIVE}},
                     follow_profile_config=True)
    live = {'agent': {'reasoning_effort': 'medium'}}
    monkeypatch.setattr(session_policy, 'profile_config', lambda authority: live)
    authority = SimpleNamespace()
    assert 'adaptive_reasoning' not in policy.config(authority, secrets=False)['agent']
    live['agent']['adaptive_reasoning'] = {'enabled': True, 'max_effort': 'xhigh'}
    assert policy.config(authority, secrets=False)['agent']['adaptive_reasoning'] == {
        'enabled': True, 'max_effort': 'xhigh'}
    # A frozen session keeps its creation snapshot.
    frozen = _policy(tmp_path, {'agent': {'reasoning_effort': 'medium', 'adaptive_reasoning': ADAPTIVE}})
    assert frozen.config(authority, secrets=False)['agent']['adaptive_reasoning'] == ADAPTIVE


def _wire(user_config, runner_extra=None, **kwargs):
    agent = SimpleNamespace()
    ctx = SimpleNamespace(
        progress_callback=None, native_tool_start_callback=None, voice_ack_callback=None, _voice_ack_guild=[None],
        voice_turn=False, _native_slack_task_cards=False, native_tool_complete_callback=None, _step_callback_sync=None,
        _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None, _event_callback_sync=None,
        _status_adapter=None, session_key='route', user_config=user_config,
        source=SimpleNamespace(platform='local'), mute_notification_reply=False, _thinking_enabled=False,
        agent_holder=[None], tools_holder=[None], process_task_id=None, process_baseline=None, run_generation=0)
    holder = SimpleNamespace(
        _ctx=ctx, _approval_owner=None,
        _runner=SimpleNamespace(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: [],
                                **(runner_extra or {})),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None, _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None)
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, {'enabled': True, 'effort': 'medium'},
                                          None, None, False, **kwargs)
    return agent


def test_turn_wiring_arms_the_policy_and_an_explicit_turn_level_wins():
    config = {'agent': {'adaptive_reasoning': ADAPTIVE}}
    agent = _wire(config)  # a runner without the messaging override detector (local owner stubs)
    assert agent.adaptive_reasoning == ADAPTIVE and agent.reasoning_user_override is False
    assert _wire(config, reasoning_explicit=True).reasoning_user_override is True
    typed = _wire(config, {'_session_reasoning_override_active': lambda key: key == 'route'})
    assert typed.reasoning_user_override is True
    assert _wire({}).adaptive_reasoning is None


def test_managed_worker_arms_the_policy_except_for_a_kanban_attempt(tmp_path):
    from dataclasses import replace

    from gateway.session_worker_construct import arm_adaptive_reasoning
    config = {'agent': {'reasoning_effort': 'medium', 'adaptive_reasoning': ADAPTIVE}}
    agent = SimpleNamespace()
    arm_adaptive_reasoning(agent, _policy(tmp_path, config))
    assert agent.adaptive_reasoning == ADAPTIVE and agent.reasoning_user_override is False
    arm_adaptive_reasoning(agent, _policy(tmp_path, config, reasoning='low'))
    assert agent.reasoning_user_override is True
    arm_adaptive_reasoning(agent, replace(_policy(tmp_path, config), kanban_json=json.dumps({'task_id': 't'})))
    assert agent.adaptive_reasoning is None
