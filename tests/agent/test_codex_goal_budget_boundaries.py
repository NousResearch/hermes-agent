"""Goal text edits cannot replenish usage or replace its budget with a profile default."""
from types import SimpleNamespace

import pytest

from agent.codex_runtime_goals import run_app_server_work
from agent.transports.codex_app_server_goals import run_native_goal
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from tests.agent.transports.test_codex_native_goals import NativeClient


class CumulativeClient(NativeClient):
    def request(self, method, params=None, timeout=30):
        used = (self.goal or {}).get('tokensUsed', 0)
        result = super().request(method, params, timeout)
        if method == 'thread/goal/set':
            self.goal['tokensUsed'] = used
            result['goal'] = dict(self.goal)
        return result


def bound_manager(sid, *, status='paused', budget=5000000, used=4372489):
    mgr = CodexGoalManager(sid, token_budget=budget); mgr.set('old objective')
    mgr.state.native_goal = {'threadId':'thread-fake-001', 'objective':'old objective',
                             'status':status, 'tokenBudget':budget, 'tokensUsed':used}
    mgr.state.turns_used = 24; mgr._save(); mgr.pause()
    return mgr


def run(client, mgr, **extra):
    session = CodexAppServerSession(client_factory=lambda **kw: client)
    result = run_native_goal(session, 'continue', session_id=mgr.session_id, state=mgr.state,
                             turn_timeout=0, idle_timeout=2, **extra)
    return result


def test_revision_keeps_budget_thread_and_usage_despite_lower_profile_default():
    old = bound_manager('revision')
    mgr = CodexGoalManager(old.session_id, token_budget=1000000)
    state = mgr.set('solve all issues')
    assert state.token_budget == 5000000 and state.turns_used == 24
    assert state.native_goal['tokensUsed'] == 4372489
    client = CumulativeClient(stages=1); client.goal = dict(state.native_goal)
    result = run(client, mgr)
    assert result.error is None
    sets = [p for method,p in client.requests if method == 'thread/goal/set']
    assert sets[0] == {'threadId':'thread-fake-001','status':'active','objective':'solve all issues'}
    assert not any(method == 'thread/goal/clear' for method,_ in client.requests)
    assert load_goal(mgr.session_id).native_goal['tokensUsed'] == 4372489


def test_completed_goal_replacement_does_not_inherit_old_usage():
    mgr = bound_manager('fresh', status='complete'); mgr.state.status = 'done'; mgr._save()
    mgr = CodexGoalManager(mgr.session_id, token_budget=90000); mgr.set('new work')
    client = CumulativeClient(stages=1)
    client.goal = {'threadId':'thread-fake-001','objective':'old objective','status':'complete',
                   'tokenBudget':5000000,'tokensUsed':4372489}
    result = run(client, mgr)
    assert result.error is None
    assert any(method == 'thread/goal/clear' for method,_ in client.requests)
    assert load_goal(mgr.session_id).native_goal['tokensUsed'] == 0


@pytest.mark.parametrize('status', ['budgetLimited','usageLimited'])
def test_limited_bound_goal_is_reported_without_starting_or_resetting(status):
    mgr = bound_manager('limited-'+status, status=status, budget=1000000)
    mgr = CodexGoalManager(mgr.session_id, token_budget=5000000); mgr.set('revised work')
    client = CumulativeClient(); client.goal = dict(mgr.state.native_goal)
    result = run(client, mgr)
    assert result.error and status in result.error and result.should_retire
    assert not any(m in ('turn/start','thread/goal/clear','thread/goal/set') for m,_ in client.requests)
    state = load_goal(mgr.session_id)
    assert state.status == 'paused' and state.token_budget == 1000000
    assert state.native_goal['tokensUsed'] == 4372489


@pytest.mark.parametrize('error,interrupted,status', [('429 Too Many Requests',False,'active'),(None,True,'active'),(None,False,'active'),(None,False,'complete')])
def test_empty_turn_is_not_mislabeled_as_transcript_write_failure(tmp_path, monkeypatch, error, interrupted, status):
    (tmp_path/'config.yaml').write_text('goals:\n  runtime: codex\nagent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 2\n')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    mgr = CodexGoalManager('empty'); mgr.set('work')
    client = CumulativeClient(); session = CodexAppServerSession(client_factory=lambda **kw:client)
    def empty_turn(*a, **kw):
        client.goal['status'] = status
        return TurnResult(thread_id=session._thread_id, error=error, interrupted=interrupted)
    session.run_turn = empty_turn
    agent = SimpleNamespace(session_id=mgr.session_id,_codex_session=session)
    monkeypatch.setattr('agent.codex_runtime._persist_projected_messages',lambda *a:False)
    result = run_app_server_work(agent,'work',messages=[])
    assert result.error and 'durably mirrored' not in result.error and result.should_retire
    if error: assert error in result.error
    assert load_goal(mgr.session_id).status == 'paused'


def test_adopted_turn_output_is_saved_even_when_native_goal_has_just_hit_limit(tmp_path, monkeypatch):
    from tests.agent.test_codex_model_created_goal import ModelGoalClient
    (tmp_path/'config.yaml').write_text('goals:\n  runtime: codex\nagent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 2\n')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    client = ModelGoalClient(stages=1,final_status='budgetLimited')
    session = CodexAppServerSession(client_factory=lambda **kw:client)
    saved = []
    def persist(agent, turn, messages):
        saved.append(turn.final_text)
        return True
    monkeypatch.setattr('agent.codex_runtime._persist_projected_messages',persist)
    monkeypatch.setattr('agent.codex_runtime._store_codex_thread_id',lambda *a:None)
    result = run_app_server_work(SimpleNamespace(session_id='adopt-limited',_codex_session=session),
                                'authorized work',messages=[])
    assert saved == ['STAGE-1'] and result.native_turns == 1
    assert result.error and 'budgetLimited' in result.error
    assert load_goal('adopt-limited').status == 'paused'
