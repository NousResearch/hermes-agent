"""A viewer of the parent session follows a delegated child live over the shared gateway, with no file reads.

The owner already publishes ``subagent.*`` to the parent session's viewers. What those frames lack
(a tool's result and outcome, and any tool of a nested child) is in the child's stored rows, because
the tool-call turn is persisted before the tool runs. ``subagent.transcript`` serves those rows, so the
hermes-tern plugin can drop its read of ``cache/delegation/live/<deleg>/task-<n>.log``.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.fixture(scope='module')
def receipt(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('subagent-live')
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/subagent_live_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((home / 'receipt.json').read_text())


def _calls(reply):
    return [call['function']['name'] for m in reply['messages'] for call in m.get('tool_calls') or []]


def test_a_running_tool_is_the_trailing_call_without_a_result(receipt):
    for name, tool in (('leaf_running', 'terminal'), ('child_running', 'delegate_task')):
        reply = receipt[name]
        assert reply['available'] and reply['running'], (name, reply)
        assert _calls(reply) == [tool] and not any(m['role'] == 'tool' for m in reply['messages']), (name, reply)
    [row] = [m for m in receipt['leaf_running']['messages'] if m.get('tool_calls')]
    assert 'LEAF_TOOL_DONE' in json.loads(row['tool_calls'][0]['function']['arguments'])['command']


def test_the_cursor_then_brings_the_result_and_the_answer(receipt):
    after = receipt['leaf_after']
    assert not after['running'] and [m['role'] for m in after['messages']] == ['tool', 'assistant'], after
    result, answer = after['messages']
    assert json.loads(result['content'])['output'] == 'LEAF_TOOL_DONE' and result['tool_name'] == 'terminal'
    assert answer['content'] == 'LEAF_FINAL'
    # A running child only appends: the poll stays on the cursor's delta read.
    assert not after['reset'] and receipt['leaf_running']['reset'], after
    assert receipt['leaf_done']['messages'][-2:] == after['messages']
    # The orchestrator's own result row names how its nested child ended.
    [done] = json.loads(receipt['child_done']['messages'][-2]['content'])['results']
    assert done['status'] == 'completed' and done['exit_reason'] == 'completed', done


def test_parent_viewers_get_subagent_frames_and_another_session_gets_none(receipt):
    child, leaf = receipt['list']
    seen = {(f['subagent_id'], f['type']) for f in receipt['frames']}
    for row in (child, leaf):
        assert {(row['subagent_id'], t) for t in ('subagent.spawn_requested', 'subagent.start',
                                                   'subagent.complete')} <= seen, receipt['frames']
    assert (child['subagent_id'], 'subagent.tool') in seen
    assert all(f['session_id'] == receipt['session_id'] for f in receipt['frames'])
    assert {f['child_session_id'] for f in receipt['frames'] if f['subagent_id'] == leaf['subagent_id']} == {
        leaf['child_session_id']}
    assert next(f for f in receipt['frames'] if f['subagent_id'] == leaf['subagent_id'])['parent_id'] == child['subagent_id']
    assert receipt['foreign_frames'] == []
    assert receipt['foreign_read']['available'] is False and receipt['foreign_read']['messages'] == []
