"""A viewer of a session reads its delegated children's transcripts over the shared gateway.

Before ``subagent.transcript`` the owner answered ``-32601`` and ``session.resume`` on a child id
answered ``not_found``, so a viewer (the hermes-tern plugin's sidebar) opened ``state.db`` itself
and guessed the child by ``parent_session_id`` + first user message.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.fixture(scope='module')
def receipt(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('subagent-transcript')
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/subagent_transcript_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((home / 'receipt.json').read_text())


def _texts(reply):
    return [(m['role'], m['content']) for m in reply['messages']]


def test_running_child_transcript_is_served_in_order_with_its_session(receipt):
    live, rows = receipt['live'], receipt['rows']['live-child']
    assert live['available'] and live['running'] and live['child_session_id'] == 'live-child', live
    assert _texts(live) == [('user', 'live goal'), ('assistant', 'live step one')]
    assert [m['row_id'] for m in live['messages']] == rows[:2]
    assert live['cursor'] == rows[1]
    # subagent.list now names the child's session, so a viewer need not guess it by goal text.
    [row] = receipt['list']['subagents']
    assert row['subagent_id'] == 'sa-0-live' and row['child_session_id'] == 'live-child', receipt['list']


def test_cursor_returns_only_rows_written_after_it(receipt):
    after, caught_up = receipt['live_after'], receipt['live_caught_up']
    assert _texts(after) == [('assistant', 'live step two')], after
    assert after['cursor'] == receipt['rows']['live-child'][2]
    assert caught_up['messages'] == [] and caught_up['cursor'] == after['cursor'], caught_up


def test_finished_children_are_read_from_the_store(receipt):
    ended = receipt['live_ended']
    assert ended['available'] and not ended['running'] and ended['child_session_id'] == 'live-child', ended
    assert _texts(ended)[-1] == ('assistant', 'live step two')
    # A compressed child resolves to its live tip, never a foreign child sharing its subagent_id.
    done = receipt['sa-1-done']
    assert done['child_session_id'] == 'done-child-2' and not done['running'], done
    assert _texts(done) == [('user', 'done summary'), ('assistant', 'done final')]
    nested = receipt['sa-0-nested']
    assert nested['child_session_id'] == 'nested-child', nested
    assert _texts(nested) == [('user', 'nested goal'), ('assistant', 'nested final')]


def test_children_of_other_conversations_are_never_served(receipt):
    for name in ('sa-0-foreign', 'sa-0-branch', 'sa-9-unknown'):
        reply = receipt[name]
        assert reply == {'subagent_id': name, 'available': False, 'child_session_id': None, 'running': False,
                         'messages': [], 'cursor': 0}, (name, reply)
    assert 'PRIVATE' not in json.dumps({k: v for k, v in receipt.items() if k != 'rows'})


def test_unknown_session_foreign_profile_and_bad_cursor_are_refused(receipt):
    assert receipt['unknown_session']['message'] == 'not_found', receipt['unknown_session']
    assert receipt['foreign_profile']['message'] == 'profile_mismatch', receipt['foreign_profile']
    assert receipt['bad_cursor']['message'] == 'invalid_params', receipt['bad_cursor']


def test_another_authenticated_viewer_reads_the_same_transcript(receipt):
    assert receipt['peer_live'] == receipt['live']


def test_runtime_advertises_the_method(receipt):
    assert 'subagent-transcript-v1' in receipt['capabilities']
