"""Native Goal ownership must survive state reloads and bypass the Hermes judge."""
import json

from hermes_cli.goals import GoalManager, GoalState


def test_native_goal_state_survives_reload():
    state = GoalState.from_json(json.dumps({'goal':'verify tests','runtime':'codex',
        'goal_id':'revision-1','token_budget':90000,'native_goal':{'status':'active','tokensUsed':1200}}))
    assert state.runtime == 'codex'
    assert state.goal_id == 'revision-1'
    assert state.token_budget == 90000
    assert state.native_goal['tokensUsed'] == 1200
    assert GoalState.from_json(state.to_json()).native_goal == state.native_goal


def test_old_goals_keep_hermes_ownership():
    assert GoalState.from_json('{"goal":"existing"}').runtime == 'hermes'


def test_native_goal_does_not_run_aux_judge(monkeypatch):
    import hermes_cli.goals as goals
    state = GoalState.from_json('{"goal":"work", "runtime":"codex", "turns_used":25}')
    monkeypatch.setattr(goals, 'load_goal', lambda _: state)
    monkeypatch.setattr(goals, 'judge_goal', lambda *a,**k: (_ for _ in ()).throw(AssertionError('second scheduler called judge')))
    mgr=GoalManager('sid')
    decision=mgr.evaluate_after_turn('intermediate progress')
    assert not decision['should_continue'] and mgr.state.turns_used==25


def test_factory_selects_native_for_codex_route(monkeypatch):
    from hermes_cli.codex_goals import goal_manager_for_session, CodexGoalManager
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, 'load_config', lambda: {'model':{'openai_runtime':'codex_app_server'},'goals':{'runtime':'codex','codex_token_budget':90000}})
    assert isinstance(goal_manager_for_session('new-session'), CodexGoalManager)


def test_late_completion_does_not_accept_changed_criteria():
    from hermes_cli.codex_goals import CodexGoalManager, record_native_goal
    from hermes_cli.goals import load_goal
    mgr = CodexGoalManager('criteria'); mgr.set('old objective')
    mgr.add_subgoal('new requirement')
    view = record_native_goal('criteria', mgr.state.goal_id, {'objective':'old objective','status':'complete'})
    assert view.status == 'paused' and 'criteria changed' in view.paused_reason
    assert load_goal('criteria').status != 'done'


def test_pause_preserves_latest_native_usage():
    from hermes_cli.codex_goals import CodexGoalManager, record_native_goal
    from hermes_cli.goals import load_goal
    mgr = CodexGoalManager('stale-pause'); mgr.set('work')
    record_native_goal(mgr.session_id, mgr.state.goal_id, {'objective':'work','status':'active','tokensUsed':4321})
    mgr.pause()
    assert load_goal(mgr.session_id).native_goal['tokensUsed'] == 4321
    record_native_goal(mgr.session_id, mgr.state.goal_id, {'objective':'work','status':'complete','tokensUsed':4567})
    assert load_goal(mgr.session_id).status == 'paused'


def test_resumed_native_goal_uses_original_budget_thread():
    from hermes_cli.codex_goals import CodexGoalManager
    from agent.codex_runtime_goals import native_resume_thread
    from types import SimpleNamespace
    mgr = CodexGoalManager('resume-thread'); mgr.set('work')
    mgr.state.native_goal = {'threadId':'original', 'status':'paused','tokensUsed':8000}; mgr._save()
    mgr.pause(); mgr.resume()
    assert native_resume_thread(SimpleNamespace(session_id=mgr.session_id)) == 'original'


def test_existing_hermes_goal_is_not_stolen_on_config_switch(monkeypatch):
    from hermes_cli.codex_goals import goal_manager_for_session, CodexGoalManager
    import hermes_cli.config as cfg
    GoalManager('legacy-owner').set('ongoing work')
    monkeypatch.setattr(cfg, 'load_config', lambda: {'model':{'openai_runtime':'codex_app_server'},'goals':{'runtime':'codex'}})
    assert not isinstance(goal_manager_for_session('legacy-owner'), CodexGoalManager)


def test_profile_config_drives_goal_owner_budget_and_wait_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.codex_goals import goal_manager_for_session, CodexGoalManager
    from agent.codex_runtime_goals import run_app_server_work
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'model:\n  openai_runtime: codex_app_server\n'
        'goals:\n  runtime: codex\n  codex_token_budget: 91000\n'
        'agent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 123\n'
    )
    mgr = goal_manager_for_session('real-config-reader')
    assert isinstance(mgr, CodexGoalManager) and mgr.token_budget == 91000
    observed = {}
    def run_turn(**options):
        observed.update(options)
        return 'accepted'
    agent = SimpleNamespace(session_id=None, _codex_session=SimpleNamespace(run_turn=run_turn))
    assert run_app_server_work(agent, 'test', messages=[]) == 'accepted'
    assert observed == {'user_input':'test', 'turn_timeout':0, 'idle_timeout':123}
