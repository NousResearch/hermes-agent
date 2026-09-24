"""Every goal surface applies the same commands to real persisted state."""
import asyncio
import os
import queue
from types import SimpleNamespace

import pytest

from hermes_cli import goals


def _surface(surface, mgr, monkeypatch, prompts=None):
    prompts = prompts if prompts is not None else []
    if surface == "cli":
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        cli = object.__new__(CLICommandsMixin)
        cli._get_goal_manager = lambda: mgr
        cli._pending_input = queue.Queue()
        cli.conversation_history = []
        def execute(arg):
            cli._handle_goal_command('/goal ' + arg)
            while not cli._pending_input.empty():
                prompts.append(cli._pending_input.get_nowait())
        return execute
    if surface == "gateway":
        from gateway.run_busy import GatewayBusySessionMixin
        from gateway.slash_commands_goals import GatewayGoalCommandsMixin

        class Runner(GatewayBusySessionMixin, GatewayGoalCommandsMixin):
            pass

        runner = object.__new__(Runner)
        async def manager(event):
            return mgr, None
        async def execute(fn, *args):
            return fn(*args)
        runner._get_goal_manager_for_event = manager
        runner._run_in_executor_with_context = execute
        runner._adapter_and_key_for = lambda event: (None, None)
        runner._enqueue_goal_turn = lambda event, text, **kwargs: prompts.append(text)
        runner._resume_caller_is_admin = lambda source: True
        def execute(arg):
            event = SimpleNamespace(get_command_args=lambda: arg, source=None)
            if arg == 'show':
                return asyncio.run(runner._busy_goal_command(event, mgr.session_id, None))
            return asyncio.run(runner._handle_goal_command(event))
        return execute
    from tui_gateway import server
    server._sessions[mgr.session_id] = {'session_key': mgr.session_id}
    def execute(arg):
        result = server._methods['command.dispatch'](1, {
            'session_id': mgr.session_id, 'name': 'goal', 'arg': arg})
        if result.get('result', {}).get('type') == 'send':
            prompts.append(result['result']['message'])
        return result
    return execute


@pytest.mark.parametrize('surface', ['cli', 'gateway', 'tui'])
@pytest.mark.parametrize('command', [
    'show', 'draft', 'draft build it', 'drafting docs', 'wait', 'wait nope',
    'wait {pid} build', 'unwait', 'gate add true', 'gate remove 1',
    'gate clear', 'gate list', 'pause', 'resume', 'clear', 'stop', 'done',
    'build it\nverify: test passes', 'status', '',
])
def test_surface_goal_state_matches_cli(surface, command, monkeypatch):
    monkeypatch.setattr(goals, 'draft_contract', lambda objective: goals.GoalContract())
    goals._DB_CACHE.clear()
    command = command.format(pid=os.getpid())
    snapshots = []
    for name in ('cli', surface):
        mgr = goals.GoalManager(session_id=name + '-parity-' + surface)
        mgr.set('original objective')
        mgr.add_gate('original gate')
        mgr.wait_on(os.getpid(), reason='existing barrier')
        result = _surface(name, mgr, monkeypatch)(command)
        if name == 'gateway' and command == 'show':
            assert mgr.state.goal in result
        state = goals.load_goal(mgr.session_id)
        if state:
            from dataclasses import asdict
            state = asdict(state)
            for key in ('created_at', 'updated_at', 'waiting_since'):
                state.pop(key, None)
        snapshots.append(state)
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize('surface', ['cli', 'gateway', 'tui'])
@pytest.mark.parametrize('draft_result', ['contract', 'unavailable', 'error'])
def test_drafts_start_work_but_inspection_and_literal_prefixes_do_not_draft(
    surface, draft_result, monkeypatch,
):
    calls = []
    def draft(objective):
        calls.append(objective)
        if draft_result == 'error':
            raise RuntimeError('aux offline')
        return goals.GoalContract(verification='tests pass') if draft_result == 'contract' else None
    monkeypatch.setattr(goals, 'draft_contract', draft)
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id='draft-' + surface)
    prompts = []
    execute = _surface(surface, mgr, monkeypatch, prompts)
    execute('draft build it')
    state = goals.load_goal(mgr.session_id)
    assert state.goal == 'build it'
    assert state.has_contract() == (draft_result == 'contract')
    assert calls == ['build it']
    assert prompts == ['build it']
    execute('show')
    execute('draft')
    execute('wait invalid')
    assert goals.load_goal(mgr.session_id).goal == 'build it'
    assert prompts == ['build it']
    execute('drafting docs')
    assert goals.load_goal(mgr.session_id).goal == 'drafting docs'
    assert calls == ['build it']
    assert prompts[-1] == 'drafting docs'
