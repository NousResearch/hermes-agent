"""Live ACP tool events carry the same identity, arguments and result the replay shows."""
from unittest.mock import AsyncMock

import pytest
from acp.schema import ToolCallProgress, ToolCallStart

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.server import _history_replay_updates


def agent():
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    return agent


ARGS = {'path': '/tmp/owned/notes.txt'}
RESULT = '{"content": "owned line", "total_lines": 1}'


async def live(agent):
    await agent._project({'session_id': 's', 'type': 'tool.start', 'admission_id': 'a',
                          'payload': {'tool_call_id': 'call_1', 'tool_name': 'read_file', 'args': ARGS}})
    await agent._project({'session_id': 's', 'type': 'tool.complete', 'admission_id': 'a',
                          'payload': {'tool_call_id': 'call_1', 'tool_name': 'read_file', 'is_error': False,
                                      'args': ARGS, 'result': RESULT}})
    return [call.kwargs['update'] for call in agent._conn.session_update.await_args_list]


def replay():
    history = [
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "/tmp/owned/notes.txt"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'tool_name': 'read_file', 'content': RESULT},
    ]
    return list(_history_replay_updates(history))


@pytest.mark.asyncio
async def test_live_tool_events_project_exactly_like_replay():
    updates = await live(agent())
    assert [type(u) for u in updates] == [ToolCallStart, ToolCallProgress]
    assert [u.model_dump(by_alias=True, exclude_none=True) for u in updates] == [
        u.model_dump(by_alias=True, exclude_none=True) for u in replay()]
    assert updates[0].tool_call_id == 'call_1' and updates[0].locations[0].path == ARGS['path']
    assert updates[1].status == 'completed'


def test_gateway_publishes_tool_arguments_and_result_for_live_viewers():
    from gateway.run_turn_runner import TurnRunner
    published = []

    class Authority:
        def publish_execution(self, session_id, generation, event_type, payload):
            published.append((event_type, payload))
            return True

    class Ctx:
        _voice_ack_guild = [None]
        _native_slack_task_cards = False

    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = Ctx()
    runner._approval_owner = (Authority(), 's', 3)
    runner.combined_tool_start_callback('call_1', 'read_file', ARGS)
    runner.combined_tool_complete_callback('call_1', 'read_file', ARGS, RESULT)
    start, complete = published
    assert start == ('tool.start', {'tool_call_id': 'call_1', 'tool_name': 'read_file', 'args': ARGS})
    assert complete[0] == 'tool.complete'
    assert complete[1]['args'] == ARGS and complete[1]['result'] == RESULT
