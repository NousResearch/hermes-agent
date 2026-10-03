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


def test_gateway_goal_kickoff_preserves_auto_skill_but_continuation_does_not(monkeypatch):
    """Only the first queued goal turn consumes the triggering channel skill binding."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin

    runner = object.__new__(GatewayGoalCommandsMixin)
    queued = []
    adapter = object()
    runner._adapter_and_key_for = lambda _event: (adapter, "sk")
    runner._enqueue_fifo = lambda key, turn, target: queued.append((key, turn, target))

    event = MessageEvent(
        text="/goal build it",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="group",
            user_id="U123",
        ),
        message_id="171.001",
        channel_prompt="Answer in haiku.",
        auto_skill=["triage"],
    )

    runner._enqueue_goal_turn(event, "kickoff", label="test", kickoff=True)
    runner._enqueue_goal_turn(event, "continuation", label="test", kickoff=False)

    kickoff = queued[0][1]
    continuation = queued[1][1]
    assert kickoff.channel_prompt == "Answer in haiku."
    assert kickoff.auto_skill == ["triage"]
    assert kickoff.message_id == "171.001"
    assert continuation.channel_prompt is None
    assert continuation.auto_skill is None
    assert continuation.message_id is None

    # Consumer boundary: the queued kickoff is the event _hmwa_prepare_turn feeds to the
    # auto-skill loader on a fresh session. Exercise that real loader, not only the copied field.
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, task_id=None: (object(), "/tmp/triage", name),
    )
    monkeypatch.setattr(
        "agent.skill_commands._build_skill_message",
        lambda _skill, _skill_dir, _header: "[SKILL SENTINEL: triage]",
    )
    from gateway.run import GatewayRunner
    consumer = object.__new__(GatewayRunner)
    consumer._hmwa_auto_load_skills(
        kickoff, kickoff.auto_skill, "quick-key", "session-key"
    )
    assert kickoff.text == "[SKILL SENTINEL: triage]\n\nkickoff"

    # Direct free-form turn is the positive sibling through the same consumer.
    direct = MessageEvent(
        text="direct",
        message_type=MessageType.TEXT,
        source=event.source,
        auto_skill=["triage"],
    )
    consumer._hmwa_auto_load_skills(
        direct, direct.auto_skill, "quick-key", "session-key"
    )
    assert direct.text == "[SKILL SENTINEL: triage]\n\ndirect"


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
