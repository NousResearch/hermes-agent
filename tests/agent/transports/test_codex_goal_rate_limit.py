"""Terminal 429 recovery must use the native ledger, never replay a user/tool turn."""

import pytest

from agent.transports.codex_app_server_goals import run_native_goal
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_cli.codex_goals import CodexGoalManager
from hermes_cli.goals import load_goal
from tests.agent.transports.test_codex_native_goals import NativeClient


class LimitedClient(NativeClient):
    def __init__(self, *, failures=1, code=429, quota=False):
        super().__init__(stages=1)
        self.failures, self.code, self.quota = failures, code, quota
        self.rearms = 0
        self.failed = 0

    def enqueue(self, tid):
        failed = self.failed < self.failures
        turn = {'id': tid, 'status': 'failed' if failed else 'completed'}
        if failed:
            turn['error'] = {'message': 'insufficient_quota' if self.quota else f'exceeded retry limit, last status: {self.code}',
                             'codexErrorInfo': {'responseTooManyFailedAttempts': {'httpStatusCode': self.code}}}
        self._notifications.extend([
            {'method': 'turn/started', 'params': {'threadId': 'thread-fake-001', 'turn': {'id': tid}}},
            {'method': 'item/completed', 'params': {'threadId': 'thread-fake-001', 'turnId': tid,
             'item': {'type': 'commandExecution', 'id': 'tool-'+tid, 'command': 'printf unique',
                      'status': 'completed', 'aggregatedOutput': tid, 'exitCode': 0}}},
            {'method': 'item/completed', 'params': {'threadId': 'thread-fake-001', 'turnId': tid,
             'item': {'type': 'agentMessage', 'id': 'msg-'+tid, 'text': 'unique-'+tid, 'phase': 'final_answer'}}},
            {'method': 'turn/completed', 'params': {'threadId': 'thread-fake-001', 'turn': turn}},
        ])

    def request(self, method, params=None, timeout=30):
        rearm = method == 'thread/goal/set' and (params or {}).get('status') == 'active' and (self.goal or {}).get('status') == 'blocked'
        used = (self.goal or {}).get("tokensUsed", 0)
        result = super().request(method, params, timeout)
        if method == "thread/goal/set":
            self.goal["tokensUsed"] = used
            result["goal"] = dict(self.goal)
        if method == 'turn/start':
            self._notifications.clear(); self.enqueue('turn-fake-001')
        elif rearm:
            self.rearms += 1
            # A delayed status notification from the FAILED turn cannot defeat recovery.
            self._notifications.append({'method': 'thread/goal/updated', 'params': {'threadId': 'thread-fake-001', 'goal': {'status': 'blocked'}}})
            self.enqueue('recovery-'+str(self.rearms))
        return result

    def take_notification(self, timeout=0):
        note = super().take_notification(timeout)
        if note and note['method'] == 'turn/completed':
            failed = note['params']['turn']['status'] == 'failed'
            self.failed += int(failed)
            self.goal['status'] = 'blocked' if failed else 'complete'
            self.goal['tokensUsed'] = 1000 + self.delivered * 100
        return note


def run(client, *, delays=(.001,), on_turn=None, on_recovery=None):
    mgr = CodexGoalManager('limited', token_budget=90000); mgr.set('work')
    session = CodexAppServerSession(client_factory=lambda **kw: client)
    result = run_native_goal(session, 'kickoff', session_id=mgr.session_id, state=mgr.state,
                             turn_timeout=0, idle_timeout=2, rate_limit_delays=delays,
                             on_turn=on_turn, on_recovery=on_recovery)
    return result, load_goal(mgr.session_id), client


def test_terminal_429_rearms_after_durable_commit_without_replay():
    client = LimitedClient(); commits = []; notices = []
    def commit(turn, continuing):
        commits.append((turn.turn_id, continuing))
        if turn.error: assert client.rearms == 0
    result, state, _ = run(client, on_turn=commit, on_recovery=notices.append)
    assert result.error is None and state.status == 'done' and state.turns_used == 1
    assert commits == [('turn-fake-001', True), ('recovery-1', False)]
    assert result.native_turns == 2 and state.native_goal['tokensUsed'] == 1200
    assert notices and len(notices) == 1
    methods = [m for m, p in client.requests]
    assert methods.count('turn/start') == 1 and methods.count('thread/start') == 1
    assert 'thread/goal/clear' not in methods
    sets = [p for m, p in client.requests if m == 'thread/goal/set']
    assert sets[-1] == {'threadId': 'thread-fake-001', 'status': 'active'}
    assert state.token_budget == 90000
    assert [m["content"] for m in result.projected_messages if m["role"] == "tool"] == [
        '{"exit_code": 0, "output": "turn-fake-001"}', '{"exit_code": 0, "output": "recovery-1"}']


def test_repeated_429_is_bounded_and_paused_not_success():
    result, state, client = run(LimitedClient(failures=9), delays=(.001, .001))
    assert client.rearms == 2 and result.error and result.should_retire
    assert state.status == 'paused' and state.turns_used == 0
    assert '429' in result.error


@pytest.mark.parametrize('code,quota', [(401, False), (400, False), (500, False), (429, True)])
def test_other_errors_or_quota_never_auto_resume(code, quota):
    result, state, client = run(LimitedClient(code=code, quota=quota))
    assert result.error and state.status == 'paused' and client.rearms == 0


def test_user_stop_during_backoff_wins():
    def stop(notice): CodexGoalManager('limited').pause('user-paused')
    result, state, client = run(LimitedClient(), on_recovery=stop)
    assert client.rearms == 0 and result.error and state.paused_reason == 'user-paused'


def test_persistence_failure_is_never_retried():
    client = LimitedClient()
    def fail(*args): raise RuntimeError('not durably mirrored')
    with pytest.raises(RuntimeError, match='durably mirrored'): run(client, on_turn=fail)
    assert client.rearms == 0 and load_goal('limited').status == 'paused'


def test_default_disabled_without_new_policy():
    result, state, client = run(LimitedClient(), delays=())
    assert result.error and state.status == 'paused' and client.rearms == 0


def test_completed_error_does_not_attach_old_plugin_stderr():
    client = LimitedClient(); client.set_stderr_tail(['WARN plugin prewarm 401 Unauthorized', 'old parse error question/title'])
    result, state, _ = run(client, delays=())
    assert '429' in result.error and '401' not in result.error and 'parse error' not in result.error
    assert state.status == 'paused'


def test_configured_runtime_uses_native_recovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.codex_runtime_goals import run_app_server_work
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('goals:\n  runtime: codex\nagent:\n  codex_turn_timeout: 0\n  codex_idle_timeout: 2\n  codex_goal_rate_limit_delays: [0.001]\n')
    mgr = CodexGoalManager('runtime', token_budget=90000); mgr.set('work')
    client = LimitedClient(); notices = []
    agent = SimpleNamespace(session_id='runtime', _codex_session=CodexAppServerSession(client_factory=lambda **kw: client),
                            _emit_diagnostic_status=notices.append)
    saved = []
    monkeypatch.setattr('agent.codex_runtime._persist_projected_messages', lambda a, t, m: saved.append(t.turn_id) or True)
    monkeypatch.setattr('agent.codex_runtime._store_codex_thread_id', lambda *a: None)
    monkeypatch.setattr('agent.codex_runtime._record_codex_app_server_usage', lambda *a, **kw: {})
    monkeypatch.setattr('agent.codex_runtime._record_codex_app_server_compaction', lambda *a: None)
    result = run_app_server_work(agent, 'start', messages=[])
    assert result.error is None and load_goal('runtime').status == 'done'
    assert saved == ['turn-fake-001', 'recovery-1'] and notices


@pytest.mark.parametrize('value', [None, True, [True], [-1], [0], [float('nan')], [3601], [1]*7])
def test_invalid_cooldown_configuration_rejected(value):
    from agent.transports.codex_goal_recovery import recovery_delays
    with pytest.raises(ValueError): recovery_delays(value)


@pytest.mark.parametrize('action', ['clear', 'replace'])
def test_goal_change_during_cooldown_cancels_recovery(action):
    def change(notice):
        mgr = CodexGoalManager('limited')
        if action == 'clear': mgr.clear()
        else: mgr.set('new goal')
    result, state, client = run(LimitedClient(), on_recovery=change)
    assert result.error and client.rearms == 0
    assert (state.status == 'cleared') if action == 'clear' else (state.goal == 'new goal' and state.status == 'active')


def test_missing_completed_status_does_not_recover_uncertain_transport():
    client = LimitedClient()
    enqueue = client.enqueue
    def uncertain(tid):
        enqueue(tid)
        client._notifications[-1]['params']['turn']['status'] = 'interrupted'
    client.enqueue = uncertain
    result, state, client = run(client)
    assert result.error and client.rearms == 0 and state.status == 'paused'


def test_chat_error_is_bounded_and_drops_ambient_diagnostics():
    from agent.codex_runtime import _codex_terminal_response
    from agent.transports.codex_app_server_session import TurnResult
    result = _codex_terminal_response(TurnResult(error='turn failed: 429\ncodex stderr (last 2 lines):\n401 old startup\nparse old error', final_text='x'*5000))
    assert '429' in result and '401' not in result and len(result) < 1000
    assert 'not a completion confirmation' in result


@pytest.mark.parametrize('status', ['budgetLimited','usageLimited','paused'])
def test_native_terminal_limits_do_not_get_rearmed(status):
    client = LimitedClient(); get = client.take_notification
    def note(timeout=0):
        result = get(timeout)
        if result and result['method'] == 'turn/completed': client.goal['status'] = status
        return result
    client.take_notification = note
    result, state, client = run(client)
    assert result.error and client.rearms == 0 and state.status == 'paused'


def test_error_message_or_old_stderr_without_structured_429_is_not_retried():
    client = LimitedClient(); enqueue = client.enqueue
    def plain(tid):
        enqueue(tid); client._notifications[-1]['params']['turn']['error'].pop('codexErrorInfo')
    client.enqueue = plain
    client.set_stderr_tail(['last unrelated HTTP429'])
    result, state, client = run(client)
    assert result.error and client.rearms == 0 and state.status == 'paused'


def test_stop_after_rearm_rpc_still_cancels_next_native_turn():
    client = LimitedClient(); request = client.request
    def race(method, params=None, timeout=30):
        result = request(method, params, timeout)
        if client.rearms: CodexGoalManager('limited').pause('user-paused')
        return result
    client.request = race
    result, state, client = run(client)
    assert result.error and state.paused_reason == 'user-paused'
    # No active turn id exists in the between-turn gap: pause the native scheduler
    # and require caller retirement, rather than inventing a turn/interrupt target.
    assert client.delivered == 1 and client.goal['status'] == 'paused' and result.should_retire


def test_existing_pause_reason_is_compact_on_goal_status_without_rewriting_ledger():
    mgr = CodexGoalManager('old-error', token_budget=90000); mgr.set('work')
    original = 'turn failed: 429\ncodex stderr (last 6 lines):\nold plugin startup warning'
    mgr.pause(original)
    assert '429' in mgr.status_line() and 'plugin startup' not in mgr.status_line()
    assert load_goal('old-error').paused_reason == original
