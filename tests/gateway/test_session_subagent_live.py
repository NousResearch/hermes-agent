"""A viewer of the parent session follows a delegated child live over the shared gateway, with no file reads.

The owner already publishes ``subagent.*`` to the parent session's viewers. What those frames lack
(a tool's result and outcome, and any tool of a nested child) is in the child's stored rows, because
the tool-call turn is persisted before the tool runs. ``session.transcript`` with ``subagent_id`` serves
those rows as display rows (adapted from #140477's ``subagent.transcript`` live test).
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


def _shown(reply):
    return [(m['role'], m.get('name') or m.get('text')) for m in reply['rows'] if m['row_kind'] == 'message']


def test_a_running_child_shows_its_rows_so_far(receipt):
    for name, goal in (('leaf_running', 'LEAF_GOAL'), ('child_running', 'CHILD_GOAL')):
        reply = receipt[name]
        assert reply['available'] and reply['running'], (name, reply)
        # The pending tool-call turn has no text, so the display rows hold only the goal until it ends.
        assert _shown(reply)[0] == ('user', goal) and not any(m['role'] == 'tool' for m in reply['rows']), reply


def test_the_cursor_then_brings_the_result_and_the_answer(receipt):
    after = receipt['leaf_after']
    assert not after['running'] and _shown(after) == [('tool', 'terminal'), ('assistant', 'LEAF_FINAL')], after
    # A running child only appends: the poll stays on the cursor.
    assert not after['reset'] and receipt['leaf_running']['reset'], after
    assert receipt['leaf_done']['rows'][-2:] == after['rows']
    assert ('tool', 'delegate_task') in _shown(receipt['child_done']), receipt['child_done']
    assert _shown(receipt['child_done'])[-1] == ('assistant', 'CHILD_FINAL'), receipt['child_done']


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
    assert receipt['foreign_read']['available'] is False and receipt['foreign_read']['rows'] == []
