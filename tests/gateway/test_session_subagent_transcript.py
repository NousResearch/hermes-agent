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
    assert live['reset'] and isinstance(live['cursor'], str)
    # subagent.list now names the child's session, so a viewer need not guess it by goal text.
    [row] = receipt['list']['subagents']
    assert row['subagent_id'] == 'sa-0-live' and row['child_session_id'] == 'live-child', receipt['list']


def test_cursor_returns_only_rows_written_after_it(receipt):
    after, caught_up = receipt['live_after'], receipt['live_caught_up']
    assert _texts(after) == [('assistant', 'live step two')], after
    assert [m['row_id'] for m in after['messages']] == receipt['rows']['live-child'][2:3]
    assert caught_up['messages'] == [] and caught_up['cursor'] == after['cursor'], caught_up


def test_finished_children_are_read_from_the_store(receipt):
    ended = receipt['live_ended']
    assert ended['available'] and not ended['running'] and ended['child_session_id'] == 'live-child', ended
    assert _texts(ended)[-1] == ('assistant', 'live step two')
    # A compressed child resolves to its live tip, never a foreign child sharing its subagent_id.
    done = receipt['sa-1-done']
    assert done['child_session_id'] == 'done-child-2' and not done['running'], done
    assert _texts(done) == [('user', 'done goal'), ('user', 'done summary'), ('assistant', 'done final')]
    # The segment before the rotation is display history, as session.resume shows a compressed chat.
    assert [m.get('archived', False) for m in done['messages']] == [True, False, False], done
    nested = receipt['sa-0-nested']
    assert nested['child_session_id'] == 'nested-child', nested
    assert _texts(nested) == [('user', 'nested goal'), ('assistant', 'nested final')]


def test_children_of_other_conversations_are_never_served(receipt):
    for name in ('sa-0-foreign', 'sa-0-branch', 'sa-9-unknown'):
        reply = receipt[name]
        assert reply == {'subagent_id': name, 'available': False, 'child_session_id': None, 'running': False,
                         'messages': [], 'cursor': None, 'reset': True}, (name, reply)
    assert 'PRIVATE' not in json.dumps({k: v for k, v in receipt.items() if k != 'rows'})


def test_unknown_session_foreign_profile_and_bad_cursor_are_refused(receipt):
    assert receipt['unknown_session']['message'] == 'not_found', receipt['unknown_session']
    assert receipt['foreign_profile']['message'] == 'profile_mismatch', receipt['foreign_profile']
    assert receipt['bad_cursor']['message'] == 'invalid_params', receipt['bad_cursor']


def test_another_authenticated_viewer_reads_the_same_transcript(receipt):
    assert receipt['peer_live'] == receipt['live']


def test_runtime_advertises_the_method(receipt):
    assert 'subagent-transcript-v1' in receipt['capabilities']


def test_polls_from_a_cursor_read_only_the_new_rows(receipt):
    # The first read and the peer's are full; the cursored polls read only what was appended.
    assert receipt['full_reads_while_polling'] == 0, receipt['full_reads_while_polling']
    assert not receipt['live_after']['reset'] and not receipt['live_caught_up']['reset']


def test_an_in_place_compaction_resets_the_viewer_instead_of_repeating_rows(receipt):
    compacted = receipt['live_compacted']
    assert compacted['reset'] and compacted['child_session_id'] == 'live-child', compacted
    texts = _texts(compacted)
    assert len(texts) == len(set(texts)) and ('user', 'live summary') in texts, compacted
    assert {m['content'] for m in compacted['messages'] if m.get('archived')} == {'live goal', 'live step one'}
    after = receipt['live_compacted_caught_up']
    assert not after['reset'] and after['messages'] == [], after


def test_rows_written_and_compacted_between_polls_reach_the_viewer(receipt):
    assert receipt['empty']['available'] and receipt['empty']['messages'] == [], receipt['empty']
    compacted = receipt['empty_compacted']
    assert compacted['reset'], compacted
    assert [(m['content'], m.get('archived', False)) for m in compacted['messages']] == [
        ('empty goal', True), ('empty step', True), ('empty summary', False)], compacted


def test_a_cursor_the_owner_never_issued_is_a_fresh_read(receipt):
    for reply in receipt['odd_cursors']:
        assert reply.get('reset') is True and reply['child_session_id'] == 'live-child', reply
        assert ('user', 'live summary') in _texts(reply), reply


def test_a_metadata_only_rewrite_of_a_held_row_resets_the_viewer(receipt):
    reacted = receipt['live_reacted']
    assert reacted['reset'], reacted
    [row] = [m for m in reacted['messages'] if m['content'] == 'live summary']
    assert [r['emoji'] for r in row['display_metadata']['reactions']] == ['\U0001f525'], row


def test_a_running_worker_of_an_orchestrator_stays_running_after_the_parent_is_rebuilt(receipt):
    assert receipt['rebuilt_orch']['running'], receipt['rebuilt_orch']
    deep = receipt['rebuilt_deep']
    assert deep['running'] and deep['child_session_id'] == 'deep-child', deep
    assert receipt['rebuilt_list'] == ['sa-0-deep', 'sa-1-orch'], receipt['rebuilt_list']


def test_a_rewrite_of_a_row_from_before_a_rotation_resets_the_viewer(receipt):
    reacted = receipt['done_ancestor_reacted']
    assert reacted['reset'] and reacted['child_session_id'] == 'done-child-2', reacted
    [row] = [m for m in reacted['messages'] if m['content'] == 'done goal']
    assert row['archived'] and [r['emoji'] for r in row['display_metadata']['reactions']] == ['\U0001f44d'], row


def test_a_queued_child_of_a_rotated_delegator_shows_none_of_the_delegator_rows(receipt):
    queued = receipt['queued_child']
    assert queued['running'] and queued['child_session_id'] == 'queued-child', queued
    assert queued['messages'] == [] and queued['reset'], queued
