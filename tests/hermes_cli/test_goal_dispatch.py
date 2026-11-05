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


def _dispatch(mgr, arg):
    from hermes_cli.goal_command import dispatch_goal_command
    return dispatch_goal_command(mgr, arg, authorize_gate=lambda: None)


@pytest.mark.parametrize('command, paused, status', [
    ('resume last goal', False, 'active'), ('continue the work', True, 'active'),
    ('pause for now', False, 'paused'), ('status please', False, 'active'),
])
def test_control_verb_with_trailing_words_never_replaces_the_goal(command, paused, status):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id='verb-wins-' + command.replace(' ', '-'))
    mgr.set('original objective')
    if paused:
        mgr.pause(reason='user-paused')
    from hermes_cli.goal_command import is_goal_control
    assert is_goal_control(command)  # gateway busy path dispatches instead of rejecting
    result = _dispatch(mgr, command)
    state = goals.load_goal(mgr.session_id)
    assert (state.goal, state.status) == ('original objective', status)
    assert f"(ignored {command.split(' ', 1)[1]!r}" in result.output



def test_double_dash_sets_control_word_goal_and_announces_the_replace():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id='verb-escape')
    mgr.set('original objective')
    result = _dispatch(mgr, '-- pause the nightly cron')
    assert goals.load_goal(mgr.session_id).goal == 'pause the nightly cron'
    assert 'was: original objective' in result.output
    mgr.clear()
    fresh = goals.GoalManager(session_id=mgr.session_id)  # gateway/TUI reload per command
    assert 'replaced' not in _dispatch(fresh, 'fresh objective').output
    _dispatch(fresh, '--dry-run the migration')
    assert goals.load_goal(mgr.session_id).goal == '--dry-run the migration'


def test_progress_asks_the_judge_and_leaves_state_alone(monkeypatch):
    from dataclasses import asdict
    from hermes_cli.goal_command import is_goal_control

    goals._DB_CACHE.clear()
    calls = []
    monkeypatch.setattr(
        goals, 'judge_progress',
        lambda goal, response, **kw: calls.append((goal, response, kw)) or '📊 Goal progress: 42%',
    )
    monkeypatch.setattr(goals, 'last_assistant_message_from_db', lambda sid, **kw: 'the agent said this')
    mgr = goals.GoalManager(session_id='progress-query')
    mgr.set('original objective')
    mgr.add_subgoal('also update the docs')
    before = asdict(goals.load_goal(mgr.session_id))
    assert is_goal_control('progress') and is_goal_control('progress please')

    result = _dispatch(mgr, 'progress')
    assert '42%' in result.output
    assert result.prompt is None and not result.error
    goal, _response, kw = calls[0]
    assert goal == 'original objective'
    assert kw['subgoals'] == ['also update the docs']
    assert (kw['turns_used'], kw['max_turns']) == (0, 20)
    assert asdict(goals.load_goal(mgr.session_id)) == before

    assert '(ignored' in _dispatch(mgr, 'progress please').output
    assert goals.load_goal(mgr.session_id).goal == 'original objective'


def test_progress_without_a_goal_never_calls_the_judge(monkeypatch):
    goals._DB_CACHE.clear()
    monkeypatch.setattr(goals, 'judge_progress', lambda *a, **k: pytest.fail('judge called with no goal'))
    mgr = goals.GoalManager(session_id='progress-no-goal')
    assert _dispatch(mgr, 'progress').output == 'No active goal. Set one with /goal <text>.'


def test_last_assistant_message_content_skips_empty_and_pre_goal_rows():
    history = [
        {'role': 'assistant', 'content': 'before the goal', 'timestamp': 100.0},
        {'role': 'user', 'content': 'keep going', 'timestamp': 110.0},
        {'role': 'assistant', 'content': '', 'timestamp': 120.0},
        {'role': 'tool', 'content': 'tool output', 'timestamp': 130.0},
        {'role': 'assistant', 'content': None, 'timestamp': 140.0},
        {'role': 'assistant', 'content': 'after the goal', 'timestamp': 200.0},
    ]
    assert goals.last_assistant_message_content(history) == 'after the goal'
    assert goals.last_assistant_message_content(history, since=150.0) == 'after the goal'
    assert goals.last_assistant_message_content(history, since=250.0) == ''
    assert goals.last_assistant_message_content(history[:5], since=150.0) == ''
    assert goals.last_assistant_message_content([]) == ''
    assert goals.last_assistant_message_content(None) == ''


def test_progress_reads_the_newest_assistant_text_from_the_db(monkeypatch):
    import time as _time

    goals._DB_CACHE.clear()
    session_id = 'progress-db-read'
    db = goals._get_session_db()
    assert db is not None, 'no session DB in this environment'
    db.create_session(session_id, 'test')
    seen = []
    monkeypatch.setattr(goals, 'judge_progress', lambda goal, response, **kw: seen.append(response) or 'report')

    mgr = goals.GoalManager(session_id=session_id)
    mgr.set('ship the widget')
    base = _time.time()
    db.append_message(session_id, 'assistant', 'reply from before the goal', timestamp=base - 600)
    db.append_message(session_id, 'tool', 'tool output', timestamp=base - 500)
    db.append_message(session_id, 'assistant', '', timestamp=base - 400)
    db.append_message(session_id, 'assistant', 'newest reply after the goal', timestamp=base)

    assert mgr.progress_report() == 'report'
    assert seen == ['newest reply after the goal']


def test_progress_without_any_assistant_text_never_calls_the_judge(monkeypatch):
    goals._DB_CACHE.clear()
    monkeypatch.setattr(goals, 'judge_progress', lambda *a, **k: pytest.fail('judge called with no evidence'))
    monkeypatch.setattr(goals, 'last_assistant_message_from_db', lambda sid, **kw: '')
    mgr = goals.GoalManager(session_id='progress-no-evidence')
    mgr.set('ship the widget')
    report = mgr.progress_report()
    assert 'No agent response to assess yet' in report


def test_progress_report_renders_the_judge_estimate(monkeypatch):
    goals._DB_CACHE.clear()
    seen = {}

    def fake_llm(call_llm, system_prompt, user_prompt, timeout):
        seen['system'], seen['user'] = system_prompt, user_prompt
        return ('{"percent": 60, "state": "halfway", "remaining": "wire up the CLI", '
                '"reason": "two of four files done"}')

    monkeypatch.setattr(goals, '_call_goal_judge_llm', fake_llm)
    monkeypatch.setattr(goals, 'last_assistant_message_from_db', lambda sid, **kw: 'the agent said this')
    mgr = goals.GoalManager(session_id='progress-render')
    mgr.set('ship the widget', contract=goals.GoalContract(verification='pytest -q tests/widget'))
    report = mgr.progress_report()
    lines = report.splitlines()
    assert lines[0].endswith('halfway')
    assert lines[1] == "  What's left: wire up the CLI"
    assert 'two of four files done' in lines[2]
    assert '0/20' in report
    assert '~60% done' in lines[3] and 'rough estimate' in lines[3]
    assert 'ship the widget' in seen['user'] and 'Verification' in seen['user']
    assert 'percent' in seen['system']

    monkeypatch.setattr(
        goals, '_call_goal_judge_llm',
        lambda *a, **k: '{"state": "early", "remaining": "start the migration"}',
    )
    no_number = mgr.progress_report()
    assert 'early' in no_number and '%' not in no_number

    monkeypatch.setattr(goals, '_call_goal_judge_llm', lambda *a, **k: 'no idea, sorry')
    assert 'did not return the expected JSON' in mgr.progress_report()
    monkeypatch.setattr(goals, '_call_goal_judge_llm', lambda *a, **k: '')
    assert 'empty reply' in mgr.progress_report()

    def boom(*a, **k):
        raise RuntimeError('aux down')

    monkeypatch.setattr(goals, '_call_goal_judge_llm', boom)
    assert 'Progress unavailable' in mgr.progress_report()
