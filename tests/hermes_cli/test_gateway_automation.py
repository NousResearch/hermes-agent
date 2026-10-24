"""The classic CLI's /goal, /loop, /heartbeat managers on a gateway session are proxies over
``session.automation``: no local GoalManager (so no read-write state.db in the CLI pid, no judge model
call), main's read helpers run on the mirrored state, and manager refusals keep main's exceptions."""
import json
from types import SimpleNamespace

import pytest


class _Conn:
    """A fake gateway: runs every call against main's in-process managers on a private store."""

    def __init__(self):
        self.calls = []
        self.closed = False

    def call(self, method, **params):
        from gateway.session_viewer_automation import invoke
        self.calls.append((params['kind'], params['call']))
        return invoke(SimpleNamespace(), params['kind'], params['session_id'], params['call'],
                      params['args'], params['kwargs'])


@pytest.fixture
def cli(tmp_path, monkeypatch):
    from hermes_cli import goals
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / 'owner.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)  # the "gateway's" store
    conn = _Conn()
    store = GatewaySessionStore.__new__(GatewaySessionStore)
    store.connection = conn
    from hermes_cli.cli_loops_mixin import CLILoopsMixin
    obj = CLILoopsMixin.__new__(CLILoopsMixin)
    obj._session_db, obj._gateway_session_open, obj.session_id = store, True, 'sid-1'
    yield SimpleNamespace(cli=obj, conn=conn, db=db)
    db.close()


def test_goal_manager_is_the_gateways(cli):
    mgr = cli.cli._get_goal_manager()
    assert type(mgr).__name__ == 'GatewayGoalManager'
    assert cli.conn.calls == [('goal', 'state')]  # loaded from the gateway, not this process's store
    state = mgr.set('ship it')
    assert state.goal == 'ship it' and mgr.is_active()
    assert mgr.status_line() == '⊙ Goal (active, 0/20 turns): ship it'
    assert json.loads(cli.db.get_meta('goal:sid-1'))['goal'] == 'ship it'
    with pytest.raises(IndexError, match='index out of range'):
        mgr.remove_subgoal(4)
    assert cli.cli._get_goal_manager() is mgr  # cached per session id, like main


def test_no_manager_before_the_gateway_session_is_open(cli):
    cli.cli._gateway_session_open = False
    assert cli.cli._get_goal_manager() is None and cli.conn.calls == []


def test_judge_decision_comes_from_the_gateway(cli, monkeypatch):
    from hermes_cli import goals
    monkeypatch.setattr(goals, 'judge_goal', lambda *a, **k: ('done', 'the report is written', False, None, False))
    mgr = cli.cli._get_goal_manager()
    mgr.set('write the report')
    decision = mgr.evaluate_after_turn('the report is written', user_initiated=True)
    assert decision['message'] == '✓ Goal achieved: the report is written'
    assert ('goal', 'evaluate_after_turn') in cli.conn.calls
    assert mgr.state.status == 'done'


def test_idle_polls_cost_no_round_trip(cli):
    hb = cli.cli._get_heartbeat_manager()
    loop = cli.cli._get_loop_manager()
    goal = cli.cli._get_goal_manager()
    before = len(cli.conn.calls)
    assert hb.due_prompt() is None and not loop.is_due() and goal.is_waiting() is False
    assert len(cli.conn.calls) == before
    hb.set('pulse', 60)
    assert hb.is_active() and hb.status_line().startswith('♥ Heartbeat (every 1m')
    loop.set('check', interval_seconds=60, times=1)
    assert loop.is_due() and loop.fire_tick().startswith('[/loop wakeup #1')
    assert loop.complete_tick('done')['message'] == '✓ Loop finished — ran 1/1 times.'


def test_goal_draft_drafts_on_the_gateway(cli, monkeypatch):
    from hermes_cli import goals
    from hermes_cli.gateway_automation import dispatch_goal_command
    drafted = []
    monkeypatch.setattr(goals, 'draft_contract', lambda objective, **k: drafted.append(objective) or None)
    mgr = cli.cli._get_goal_manager()
    lines = []
    result = dispatch_goal_command(mgr, 'draft ship the API', authorize_gate=lambda: None, progress=lines.append)
    assert drafted == ['ship the API'] and ('goal', 'command') in cli.conn.calls
    assert lines == ['Drafting completion contract…']
    assert result.output.startswith('⊙ Goal set (20-turn budget): ship the API') and result.kickoff
    assert mgr.state.goal == 'ship the API'
