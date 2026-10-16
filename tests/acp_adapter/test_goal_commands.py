"""Goals use the shared command/manager and the existing ACP prompt queue."""
import asyncio
import threading

import acp
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_cli.goals import GoalManager
from tests.acp_adapter.test_acp_commands import CaptureConn, FakeAgent
from tests.acp_adapter.test_control_runtime_context import RUNTIME_CONTEXT


@pytest.mark.asyncio
async def test_inprocess_goal_dispatch_continuation_and_cold_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {'goals': {'max_turns': 2}})
    judgments = []

    def judge(goal, response, **kwargs):
        judgments.append((goal, response))
        return 'continue', 'fixture needs another step', False, None, False

    monkeypatch.setattr('hermes_cli.goals.judge_goal', judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    conn = CaptureConn()
    agent.on_connect(conn)
    state = manager.create_session(cwd=str(tmp_path))
    sid = state.session_id
    assert 'goal' in {cmd.name for cmd in agent._available_commands()}

    async def slash(text):
        return await agent.prompt([acp.text_block(text), acp.text_block(RUNTIME_CONTEXT)], sid)

    await slash('/help')
    assert '/goal' in conn.updates[-1][1].content.text
    await slash('/goal')
    await slash('/goal status')
    assert state.agent.runs == []
    body = 'Finish WS_SHARED\n  keep indentation and --session\n<example>user XML</example>'
    assert (await slash('/goal ' + body)).stop_reason == 'end_turn'
    saved = GoalManager(sid).state
    assert saved.goal == body
    assert (saved.status, saved.turns_used, saved.max_turns) == ('paused', 2, 2)
    assert state.agent.runs[0] == body
    assert len(state.agent.runs) == len(judgments) == 2
    assert 'standing goal' in state.agent.runs[1]
    assert RUNTIME_CONTEXT not in str(state.history)
    assert len([u for _, u in conn.updates if u.session_update == 'agent_message_chunk'
                and 'ran:' in u.content.text]) == 2
    await slash('/goal pause')
    cold = HermesACPAgent(session_manager=SessionManager(agent_factory=FakeAgent))
    cold.on_connect(conn)
    await cold.resume_session(cwd=str(tmp_path), session_id=sid)
    await cold.load_session(cwd=str(tmp_path), session_id=sid)
    before = len(judgments)
    await cold.prompt([acp.text_block('/goal status')], sid)
    assert 'paused' in conn.updates[-1][1].content.text
    assert len(judgments) == before
    await cold.prompt([acp.text_block('/goal resume')], sid)
    assert len(judgments) > before
    await cold.prompt([acp.text_block('/goal clear')], sid)
    assert not GoalManager(sid).has_goal()


@pytest.mark.asyncio
async def test_inprocess_interruption_pauses_goal_without_followup(tmp_path, monkeypatch):
    def no_judge(*args, **kwargs):
        pytest.fail('Interrupted output must not drive the goal judge')

    monkeypatch.setattr('hermes_cli.goals.judge_goal', no_judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    GoalManager(state.session_id).set('Leave the cancelled turn stopped')
    state.is_running = True
    state.cancel_event = threading.Event()
    state.cancel_event.set()
    await agent._finish_turn(state, state.session_id, None,
                             {'final_response': 'partial', 'interrupted': True}, None, False)
    assert GoalManager(state.session_id).state.status == 'paused'
    assert not state.queued_prompts and state.agent.runs == []


@pytest.mark.asyncio
async def test_cancel_during_goal_judgment_does_not_restart_the_turn(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    judgments = []

    def judge(*args, **kwargs):
        judgments.append(args)
        started.set()
        assert release.wait(5), 'fixture judge never released'
        return 'continue', 'another step', False, None, False

    monkeypatch.setattr('hermes_cli.goals.judge_goal', judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    GoalManager(state.session_id).set('Stay stopped after cancellation')
    state.is_running = True
    task = asyncio.create_task(agent._finish_turn(state, state.session_id, None,
        {'final_response': 'finished inference'}, None, False))
    try:
        assert await asyncio.to_thread(started.wait, 5), 'goal judge never started'
        await agent.cancel(state.session_id)
    finally:
        release.set()
        response = await task
    assert response.stop_reason == "cancelled"
    assert len(judgments) == 1 and state.agent.runs == []
    assert GoalManager(state.session_id).state.status == 'paused'


@pytest.mark.asyncio
@pytest.mark.parametrize('queued_input', [None, 'ordinary follow-up'])
async def test_cancel_during_goal_notice_does_not_restart_the_turn(tmp_path, monkeypatch, queued_input):
    started, release = asyncio.Event(), asyncio.Event()
    judgments = []

    def judge(*args, **kwargs):
        judgments.append(args)
        return 'continue', 'another step', False, None, False

    class BlockedNotice(CaptureConn):
        async def session_update(self, session_id, update):
            if (state.pending_goal_prompt and not started.is_set()
                    and update.session_update == 'agent_message_chunk'):
                started.set()
                await release.wait()
            await super().session_update(session_id, update)

    monkeypatch.setattr('hermes_cli.goals.judge_goal', judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    conn = BlockedNotice()
    agent.on_connect(conn)
    GoalManager(state.session_id).set('Stay stopped after cancellation')
    state.is_running = True
    task = asyncio.create_task(agent._finish_turn(state, state.session_id, conn,
        {'final_response': 'finished inference'}, None, False))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert state.pending_goal_prompt
        if queued_input:
            await agent.prompt([acp.text_block('/queue ' + queued_input)], state.session_id)
            assert state.queued_prompts == [queued_input]
        await agent.cancel(state.session_id)
    finally:
        release.set()
        response = await asyncio.wait_for(task, 5)
    assert state.agent.runs == ([queued_input] if queued_input else [])
    assert response.stop_reason == 'cancelled'
    assert len(judgments) == 1
    assert GoalManager(state.session_id).state.status == 'paused'
    assert state.pending_goal_prompt is None and not state.queued_prompts


@pytest.mark.asyncio
@pytest.mark.parametrize('queued_input', [None, 'ordinary follow-up'])
async def test_cancel_during_goal_user_notification_keeps_admission_cancelled(tmp_path, monkeypatch, queued_input):
    started, release = asyncio.Event(), asyncio.Event()
    judgments = []

    def judge(*args, **kwargs):
        judgments.append(args)
        return ('continue' if len(judgments) == 1 else 'done'), 'fixture verdict', False, None, False

    class BlockedUserNotification(CaptureConn):
        async def session_update(self, session_id, update):
            if update.session_update == 'user_message_chunk' and not started.is_set():
                started.set()
                await release.wait()
            await super().session_update(session_id, update)

    monkeypatch.setattr('hermes_cli.goals.judge_goal', judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    conn = BlockedUserNotification()
    agent.on_connect(conn)
    GoalManager(state.session_id).set('Stay stopped after cancellation')
    state.is_running = True
    task = asyncio.create_task(agent._finish_turn(state, state.session_id, conn,
        {'final_response': 'finished inference'}, None, False))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert state.pending_goal_prompt is None  # already removed by the drain
        await agent.cancel(state.session_id)
        if queued_input:
            # An explicit queue request may start a fresh human turn while the
            # automatic notification is blocked. It must not revive that admission.
            await agent.prompt([acp.text_block('/queue ' + queued_input)], state.session_id)
    finally:
        release.set()
        response = await asyncio.wait_for(task, 5)
    assert state.agent.runs == ([queued_input] if queued_input else [])
    assert response.stop_reason == 'cancelled'
    assert len(judgments) == 1
    assert GoalManager(state.session_id).state.status == 'paused'
    assert state.pending_goal_prompt is None and not state.queued_prompts
    # A later explicit resume authorizes new automatic work, using a fresh scope.
    before = len(state.agent.runs)
    assert (await agent.prompt([acp.text_block('/goal resume')], state.session_id)).stop_reason == 'end_turn'
    assert len(state.agent.runs) == before + 1
    assert len(judgments) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('commands', [
    ('/goal pause',),
    ('/goal clear',),
    ('/goal pause', '/goal resume'),
    ('/goal Replacement goal',),
])
async def test_goal_commands_revoke_detached_user_notification(tmp_path, monkeypatch, commands):
    started, release = asyncio.Event(), asyncio.Event()
    judgments = []
    blocked_text = []

    def judge(*args, **kwargs):
        judgments.append(args)
        return ('continue' if len(judgments) == 1 else 'done'), 'fixture verdict', False, None, False

    class BlockedContinuation(CaptureConn):
        async def session_update(self, session_id, update):
            if (update.session_update == 'user_message_chunk' and state.agent.runs
                    and not started.is_set()):
                blocked_text.append(update.content.text)
                started.set()
                await release.wait()
            await super().session_update(session_id, update)

    monkeypatch.setattr('hermes_cli.goals.judge_goal', judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    agent.on_connect(BlockedContinuation())

    async def slash(text):
        response = await agent.prompt([acp.text_block(text)], state.session_id)
        assert response.stop_reason == 'end_turn'

    task = asyncio.create_task(slash('/goal Original goal'))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert state.agent.runs == ['Original goal']
        assert state.pending_goal_prompt is None and not state.is_running
        for command in commands:
            await slash(command)
        # Goal controls must not cancel ordinary human input or change stopReason.
        await slash('/queue ordinary follow-up')
        expected_runs = list(state.agent.runs)
        assert expected_runs[-1] == 'ordinary follow-up'
        assert 'standing goal' in blocked_text[0]
        expected_judgments = len(judgments)
        assert not state.cancel_event.is_set()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)

    assert state.agent.runs == expected_runs  # no stale inference after command success
    assert len(judgments) == expected_judgments
    assert state.pending_goal_prompt is None and not state.queued_prompts
    saved = GoalManager(state.session_id).state
    if commands == ('/goal pause',):
        assert saved.status == 'paused'
    elif commands == ('/goal clear',):
        assert saved.status == 'cleared' and not GoalManager(state.session_id).has_goal()
    else:
        assert len(expected_runs) == 3  # original, explicitly authorized goal work, human input
        assert saved.goal == ('Original goal' if commands[-1] == '/goal resume' else 'Replacement goal')


@pytest.mark.asyncio
async def test_cancel_during_queued_human_notification_preserves_fifo_and_pauses_goal(tmp_path, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    def no_judge(*args, **kwargs):
        pytest.fail('Cancelled goal must be paused before queued human turns run')

    class BlockedUserNotification(CaptureConn):
        async def session_update(self, session_id, update):
            if update.session_update == 'user_message_chunk' and not started.is_set():
                started.set()
                await release.wait()
            await super().session_update(session_id, update)

    monkeypatch.setattr('hermes_cli.goals.judge_goal', no_judge)
    manager = SessionManager(agent_factory=FakeAgent)
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=str(tmp_path))
    conn = BlockedUserNotification()
    agent.on_connect(conn)
    GoalManager(state.session_id).set('Pause before running queued human input')
    state.is_running = True
    for text in ('first human input', 'second human input'):
        await agent.prompt([acp.text_block('/queue ' + text)], state.session_id)
    task = asyncio.create_task(agent._finish_turn(state, state.session_id, conn,
        {'final_response': 'finished inference'}, None, False))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await agent.cancel(state.session_id)
    finally:
        release.set()
        response = await asyncio.wait_for(task, 5)
    assert state.agent.runs == ['first human input', 'second human input']
    assert response.stop_reason == 'cancelled'
    assert GoalManager(state.session_id).state.status == 'paused'
    assert state.pending_goal_prompt is None and not state.queued_prompts


@pytest.mark.asyncio
async def test_inprocess_goal_follows_compression_target_on_cold_load(tmp_path):
    from hermes_cli.goals import migrate_goal_to_session
    from tests.acp_adapter.test_config_options import runtime, values

    def runtime_agent():
        agent = runtime()
        agent._session_db, agent._session_db_created = db, True
        return agent

    manager = SessionManager(agent_factory=runtime_agent)
    db = manager._get_db()
    agent = HermesACPAgent(session_manager=manager)
    agent.on_connect(CaptureConn())
    state = manager.create_session(cwd=str(tmp_path))
    root, child = state.session_id, 'compressed-goal-tip'
    await agent.set_config_option('reasoning', root, 'high')
    await agent.set_config_option('fast', root, 'fast')
    db.append_message(root, role='user', content='archived parent context')
    GoalManager(root).set('Finish after compaction')
    # Production publishes the agent's init metadata, without ACP selector choices.
    holder = 'compression-selector-fixture'
    assert db.try_acquire_compression_lock(root, holder)
    try:
        db.publish_compression_child(parent_session_id=root, child_session_id=child, source='acp',
            model=state.model, model_config={'max_iterations': 90,
                'reasoning_config': {'enabled': True, 'effort': 'high'}, 'max_tokens': None,
                'service_tier': 'priority'},
            messages=[{'role': 'user', 'content': 'compressed context'}],
            compression_lock_holder=holder)
    finally:
        db.release_compression_lock(root, holder)
    assert migrate_goal_to_session(root, child, reason='fixture compression')
    state.agent.session_id = child
    agent._handle_slash_command('/goal pause', state)
    assert GoalManager(child).state.status == 'paused'
    cold = HermesACPAgent(session_manager=SessionManager(agent_factory=runtime_agent))
    conn = CaptureConn()
    cold.on_connect(conn)
    loaded = await cold.load_session(cwd=str(tmp_path), session_id=root)
    assert values(loaded)['reasoning'] == 'high'
    assert values(loaded)['fast'] == 'fast'
    await cold.prompt([acp.text_block('/goal status')], root)
    assert 'Finish after compaction' in conn.updates[-1][1].content.text
    restored = cold.session_manager.get_session(root)
    assert restored.agent.reasoning_config == {'enabled': True, 'effort': 'high'}
    assert restored.agent.service_tier == 'priority'
    assert restored.history[0]['content'] == 'compressed context'
    assert db.get_session(root)['end_reason'] == 'compression'
