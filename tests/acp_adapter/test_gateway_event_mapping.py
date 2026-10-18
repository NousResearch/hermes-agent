"""The gateway's turn events reach the editor as the in-process ACP agent projected them (#134889
ACP-5, ACP-15): native plan updates, the executor's tool verdict, closed tool and permission bubbles,
and ``messageId`` on assistant chunks."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.schema import AgentPlanUpdate, AllowedOutcome, DeniedOutcome

from acp_adapter.gateway_server import GatewayACPAgent

TODOS = [{'id': '1', 'content': 'write the test', 'status': 'completed'},
         {'id': '2', 'content': 'fix the bug', 'status': 'in_progress'},
         {'id': '3', 'content': 'drop the hack', 'status': 'cancelled'}]


def agent():
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    return agent


def updates(agent):
    return [call.kwargs['update'] for call in agent._conn.session_update.await_args_list]


async def project(agent, kind, payload, aid='a'):
    await agent._project({'session_id': 's', 'type': kind, 'admission_id': aid, 'payload': payload})


@pytest.mark.asyncio
async def test_todo_updated_becomes_the_native_plan_update():
    """The owner publishes ``todo.updated`` for a local viewer's turn (an ACP session is LOCAL);
    the in-process agent sent Zed's task panel a ``plan`` update from the todo result."""
    editor = agent()
    await project(editor, 'todo.updated', {'todos': TODOS, 'revision': 2})
    (plan,) = updates(editor)
    assert isinstance(plan, AgentPlanUpdate)
    assert [(e.content, e.status) for e in plan.entries] == [
        ('write the test', 'completed'), ('fix the bug', 'in_progress'), ('[cancelled] drop the hack', 'completed')]


def test_an_acp_session_turn_publishes_todo_updated():
    """The real owner path: a LOCAL (ACP-created) turn's todo tool completion publishes ``todo.updated``."""
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    published = []
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL, chat_id='chat'),
                                _voice_ack_guild=[None], _native_slack_task_cards=False, progress_mode='all')
    turn._approval_owner = (SimpleNamespace(sessions={'s': SimpleNamespace(tool_progress_mode=None)}), 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    turn._publish_api_tool = lambda *a, **k: None
    turn.combined_tool_start_callback('t1', 'todo', {'todos': TODOS})
    turn.combined_tool_complete_callback('t1', 'todo', {'todos': TODOS}, json.dumps({'todos': TODOS, 'revision': 2}))
    assert ('todo.updated', {'todos': TODOS, 'revision': 2}) in published


@pytest.mark.asyncio
async def test_tool_complete_honours_the_executors_is_error_verdict():
    """A failed tool whose plain-text result the heuristic misses still shows ``failed``: the owner sends
    the executor's ``is_error`` on every ``tool.complete``, as the in-process callback passed it on."""
    editor = agent()
    args = {'command': 'make'}
    await project(editor, 'tool.start', {'tool_call_id': 'c1', 'tool_name': 'terminal', 'args': args})
    await project(editor, 'tool.complete', {'tool_call_id': 'c1', 'tool_name': 'terminal', 'args': args,
                                            'is_error': True, 'result': 'cancelled by the user'})
    assert updates(editor)[-1].status == 'failed'


@pytest.mark.asyncio
async def test_a_tool_call_left_open_is_closed_when_the_turn_completes():
    """A tool blocked by a guard or a denied approval never publishes ``tool.complete``; the in-process
    agent flushed such calls as ``failed`` at turn end so no bubble spins forever."""
    editor = agent()
    await project(editor, 'tool.start', {'tool_call_id': 'blocked', 'tool_name': 'terminal', 'args': {'command': 'rm'}})
    await project(editor, 'tool.start', {'tool_call_id': 'other', 'tool_name': 'terminal', 'args': {}}, aid='b')
    await project(editor, 'message.complete', {'text': 'done', 'outcome': 'completed'})
    closed = [u for u in updates(editor) if getattr(u, 'session_update', None) == 'tool_call_update']
    assert [(u.tool_call_id, u.status) for u in closed] == [('blocked', 'failed')]
    assert ('s', 'other') in editor._tool_args  # another admission's call is not this turn's to close


@pytest.mark.asyncio
@pytest.mark.parametrize('choice, status', [('allow_once', 'completed'), ('deny', 'failed'), (None, 'failed')])
async def test_an_answered_permission_card_is_closed(choice, status):
    """The editor materialises a permission request's ``tool_call`` as a pending bubble; the in-process
    agent closed it once answered (``completed`` for an allow, ``failed`` otherwise)."""
    editor = agent()
    outcome = (AllowedOutcome(outcome='selected', option_id=choice) if choice
               else DeniedOutcome(outcome='cancelled'))
    editor._conn.request_permission.return_value = SimpleNamespace(outcome=outcome)
    editor._gateway.rpc.return_value = {'status': 'resolved'}
    prompt = {'kind': 'approval', 'prompt_id': 'p', 'execution_generation': 1, 'command': 'rm -r x',
              'description': 'recursive delete', 'choices': ['once', 'deny']}
    await editor._answer_permission('s', prompt)
    card = editor._conn.request_permission.await_args.kwargs['tool_call']
    (closed,) = updates(editor)
    assert (closed.tool_call_id, closed.status) == (card.tool_call_id, status)


@pytest.mark.asyncio
async def test_an_answered_clarify_card_is_closed():
    editor = agent()
    editor._conn.request_permission.return_value = SimpleNamespace(
        outcome=AllowedOutcome(outcome='selected', option_id='choice-0'))
    editor._gateway.rpc.return_value = {'status': 'resolved'}
    await editor._answer_clarify('s', {'kind': 'clarify', 'prompt_id': 'q', 'execution_generation': 1,
                                       'question': 'Which?', 'choices': ['A', 'B']})
    (closed,) = updates(editor)
    assert (closed.tool_call_id, closed.status) == ('clarify-q', 'completed')


@pytest.mark.asyncio
async def test_assistant_chunks_carry_a_message_id_that_changes_at_each_new_message():
    """ACP clients group chunks into one reply by ``messageId``; the in-process agent closed the id at each
    tool boundary and at turn end, so separate replies never merge into one bubble."""
    editor = agent()
    await project(editor, 'reasoning.delta', {'text': 'think'})
    await project(editor, 'message.delta', {'text': 'Let me look. '})
    await project(editor, 'tool.start', {'tool_call_id': 'c1', 'tool_name': 'read_file', 'args': {'path': '/x'}})
    await project(editor, 'tool.complete', {'tool_call_id': 'c1', 'tool_name': 'read_file', 'args': {'path': '/x'},
                                            'is_error': False, 'result': '{"content": "x"}'})
    await project(editor, 'message.delta', {'text': 'It says '})
    await project(editor, 'message.complete', {'text': 'It says x.', 'outcome': 'completed'})
    await project(editor, 'message.delta', {'text': 'next turn'}, aid='b')
    chunks = [u for u in updates(editor) if u.session_update in ('agent_message_chunk', 'agent_thought_chunk')]
    ids = [u.message_id for u in chunks]
    assert all(ids), ids
    import uuid
    assert all(uuid.UUID(i) for i in ids)
    # think + 'Let me look.' | 'It says ' + remainder 'x.' | next turn
    assert ids[0] == ids[1] != ids[2] == ids[3] != ids[4], ids


@pytest.mark.asyncio
async def test_a_reconnect_never_reuses_the_open_reply_id():
    """A reply cut off by a lost connection never receives its ``message.complete`` (the re-attach starts at
    the owner's current sequence), so the next reply must still open a new ``messageId``."""
    editor = agent()
    await project(editor, 'message.delta', {'text': 'half a reply'})
    await editor._teardown()
    await project(editor, 'message.delta', {'text': 'the next reply'}, aid='b')
    first, second = (u.message_id for u in updates(editor))
    assert first != second


@pytest.mark.asyncio
async def test_a_terminal_receipt_finishes_the_reply_like_message_complete():
    """A retry whose turn already settled is reported from its receipt: that remainder carries the
    ``messageId``, closes it, and fails the turn's still-open tool calls, as ``message.complete`` does."""
    editor = agent()
    await project(editor, 'message.delta', {'text': 'Look'})
    await project(editor, 'tool.start', {'tool_call_id': 'c1', 'tool_name': 'terminal', 'args': {}})
    editor._gateway.rpc.return_value = {'result': {'final_response': 'Done.'}}
    await editor._settled_receipt(editor._gateway, 's', 's', {'admission_id': 'a', 'outcome': 'completed'})
    await project(editor, 'message.delta', {'text': 'next'}, aid='b')
    sent = updates(editor)
    chunks = [u for u in sent if u.session_update == 'agent_message_chunk']
    assert [c.content.text for c in chunks] == ['Look', 'Done.', 'next']
    assert chunks[1].message_id and chunks[1].message_id != chunks[2].message_id
    closed = [u for u in sent if u.session_update == 'tool_call_update']
    assert [(u.tool_call_id, u.status) for u in closed] == [('c1', 'failed')]
