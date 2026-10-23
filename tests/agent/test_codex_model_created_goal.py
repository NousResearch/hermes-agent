"""A model-created native Goal must stay supervised after the first ordinary turn."""
from types import SimpleNamespace

import pytest

from agent.codex_runtime_goals import run_app_server_work
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from tests.agent.transports.test_codex_native_goals import NativeClient


class ModelGoalClient(NativeClient):
    def request(self, method, params=None, timeout=30):
        result = super().request(method, params, timeout)
        if method == 'turn/start' and self.goal is None:
            self.goal = {'threadId': 'thread-fake-001', 'objective': 'new authorized work',
                         'status': 'active', 'tokenBudget': 5000000, 'tokensUsed': 644603}
        return result

    def take_notification(self, timeout=0):
        result = super().take_notification(timeout)
        if (result and result['method'] == 'turn/completed' and self.delivered == 1
                and getattr(self, 'on_first_complete', None)):
            self.on_first_complete()
        return result


def run_model_goal(tmp_path, monkeypatch, *, commit=None, final_status='complete',
                   during_turn=None, persist_ok=True):
    (tmp_path / 'config.yaml').write_text(
        'goals:\n  runtime: codex\nagent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 2\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager = CodexGoalManager('model-created', token_budget=1000000)
    manager.set('old work')
    manager.state.native_goal = {'threadId': 'old-thread', 'objective': 'old work',
                                 'status': 'budgetLimited', 'tokensUsed': 1002439}
    manager._save()
    manager.pause('Codex Goal budgetLimited')
    client = ModelGoalClient(final_status=final_status)
    client.on_first_complete = (lambda: during_turn(manager)) if during_turn else None
    session = CodexAppServerSession(client_factory=lambda **kwargs: client)
    agent = SimpleNamespace(session_id=manager.session_id, _codex_session=session)
    mirrored = []

    def persist(agent, turn, messages):
        mirrored.append(turn.final_text)
        if commit:
            commit(manager, turn)
        return persist_ok

    monkeypatch.setattr('agent.codex_runtime._persist_projected_messages', persist)
    monkeypatch.setattr('agent.codex_runtime._store_codex_thread_id', lambda *a: None)
    monkeypatch.setattr('agent.codex_runtime._record_codex_app_server_usage', lambda *a, **k: {})
    result = run_app_server_work(agent, 'Add the authorized budget and continue', messages=[])
    return result, client, session, load_goal(manager.session_id), mirrored


def test_model_created_goal_adopts_new_thread_budget_and_consumes_all_turns(tmp_path, monkeypatch):
    result, client, session, state, mirrored = run_model_goal(tmp_path, monkeypatch)
    assert result.error is None and result.native_turns == 3
    assert mirrored == ['STAGE-1', 'STAGE-2', 'STAGE-3']
    assert state.goal == 'new authorized work' and state.status == 'done'
    assert state.token_budget == 5000000 and state.native_goal['tokensUsed'] == 644603
    assert state.native_goal['threadId'] == 'thread-fake-001'
    assert [m for m, _ in client.requests].count('turn/start') == 1
    assert not any(m in ['thread/goal/set', 'thread/goal/clear'] for m, _ in client.requests)
    assert getattr(session, '_native_goal_running', False) is False


def test_user_pause_after_adoption_stops_native_continuation(tmp_path, monkeypatch):
    result, client, _, state, mirrored = run_model_goal(
        tmp_path, monkeypatch, commit=lambda mgr, turn: mgr.pause('user-paused'))
    assert result.interrupted and result.should_retire
    assert state.status == 'paused' and state.paused_reason == 'user-paused'
    assert mirrored == ['STAGE-1'] and client.delivered == 1


@pytest.mark.parametrize('status', ['budgetLimited', 'usageLimited', 'blocked'])
def test_adopted_goal_limits_are_not_success(tmp_path, monkeypatch, status):
    result, _, _, state, _ = run_model_goal(tmp_path, monkeypatch, final_status=status)
    assert result.interrupted and state.status == 'paused'
    assert state.native_goal['status'] == status
    assert state.token_budget == 5000000


def test_new_user_goal_during_discovery_is_not_replaced(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='changed while discovering'):
        run_model_goal(tmp_path, monkeypatch, during_turn=lambda mgr: mgr.set('new user criteria'))
    assert load_goal('model-created').goal == 'new user criteria'


def test_failed_transcript_persistence_pauses_without_replay(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='not durably mirrored'):
        run_model_goal(tmp_path, monkeypatch, persist_ok=False)
    state = load_goal('model-created')
    assert state.status == 'paused' and state.token_budget == 5000000


def test_existing_terminal_goal_allows_ordinary_chat_without_revival(tmp_path, monkeypatch):
    (tmp_path / 'config.yaml').write_text('goals:\n  runtime: codex\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager = CodexGoalManager('terminal-chat'); manager.set('old work')
    manager.state.native_goal = {'threadId': 'thread-fake-001', 'objective': 'old work',
                                 'status': 'budgetLimited'}
    manager._save(); manager.pause('budget limited')
    turn = SimpleNamespace(thread_id='thread-fake-001', error=None, interrupted=False)
    client = NativeClient(); client.goal = manager.state.native_goal
    session = SimpleNamespace(run_turn=lambda **kw: turn, _client=client, _thread_id=turn.thread_id)
    assert run_app_server_work(SimpleNamespace(session_id=manager.session_id, _codex_session=session),
                               'What remains?', messages=[]) is turn
    assert load_goal(manager.session_id).status == 'paused'
    assert not any(m == 'thread/goal/set' for m, _ in client.requests)
