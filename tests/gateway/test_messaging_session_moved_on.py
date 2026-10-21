"""Messaging-session reads after the conversation moved on (RT-18, TUI-19 on messaging rows, #134889).

A messaging lane's control card (goal, loop, heartbeat) follows the conversation through compression,
and an old conversation opened read-only for a refused delete is not a live session.
"""
import pytest

from tests.gateway.test_authority_cold_route_recovery import _cold, _compressed_lane, _ref, _superseded


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))


@pytest.mark.asyncio
async def test_the_control_card_of_a_compressed_messaging_session_reads_its_current_segment(tmp_path):
    from hermes_cli.goals import GoalState
    from gateway.session_ancillary import control_snapshot
    _compressed_lane(tmp_path)
    cold = await _cold(tmp_path)
    cold.authority.authorize(cold.actor, _ref('lane-root'), 'session:read')
    # Compression moves automation state to the continuation and archives the root's copy.
    cold.authority.db.set_meta('goal:lane-root', GoalState('ROOT_GOAL', status='cleared').to_json())
    cold.authority.db.set_meta('goal:lane-tip', GoalState('TIP_GOAL').to_json())

    snapshot = control_snapshot(cold.authority, _ref('lane-root'))

    assert snapshot['goal']['title'] == 'TIP_GOAL'


@pytest.mark.asyncio
async def test_an_unviewed_superseded_mount_is_not_listed_live(tmp_path):
    """A refused delete of an old conversation leaves its read-only mount behind with no viewer."""
    from gateway.session_controls import AuthorityConnection
    cold, _successor = await _superseded(tmp_path)
    connection = AuthorityConnection(cold.authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read']})
    resumed = await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old'}})
    assert cold.authority.sessions['old'].superseded, resumed
    listed = await connection.dispatch({'id': 2, 'method': 'session.list', 'params': {}})
    assert 'old' in {row['id'] for row in listed['result']['sessions']}, 'a viewed historic mount stays listed'

    await connection.dispatch({'id': 3, 'method': 'session.detach', 'params': {
        'session_id': 'old', 'subscription_id': resumed['result']['subscription_id']}})
    listed = await connection.dispatch({'id': 4, 'method': 'session.list', 'params': {}})
    active = await connection.dispatch({'id': 5, 'method': 'session.active_list', 'params': {}})

    assert 'old' not in {row['id'] for row in listed['result']['sessions']}
    assert 'old' not in {row.get('id') or row.get('session_id') for row in active['result']['sessions']}


@pytest.mark.asyncio
async def test_an_old_messaging_conversation_can_be_deleted_from_the_picker(tmp_path):
    """The picker reads the revision with ``session.resume`` (a read-only mount for a conversation its
    chat moved past), then deletes by ``session.mutate``; the delete removes this row only."""
    from gateway.session_controls import AuthorityConnection
    cold, successor = await _superseded(tmp_path)
    connection = AuthorityConnection(cold.authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read', 'session:control']})
    resumed = await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old'}})
    assert cold.authority.sessions['old'].superseded, resumed
    cold.authority.runner._evict_cached_agent = lambda key, **kw: pytest.fail('evicted the chat route')

    deleted = await connection.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
        'session_id': 'old', 'request_id': 'delete-old', 'operation': 'delete', 'payload': {},
        'expected_revision': resumed['result']['revision'],
        'expected_generation': resumed['result']['execution_generation']}})

    assert 'result' in deleted, deleted
    assert cold.authority.db.get_session('old') is None
    assert 'old' not in cold.authority.sessions
    assert cold.store.get_or_create_session(successor.origin).session_id == successor.session_id


@pytest.mark.asyncio
async def test_an_old_messaging_conversation_still_refuses_every_other_write(tmp_path):
    from gateway.session_controls import AuthorityConnection
    cold, _successor = await _superseded(tmp_path)
    connection = AuthorityConnection(cold.authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read', 'session:control']})
    resumed = await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old'}})
    renamed = await connection.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
        'session_id': 'old', 'request_id': 'rename-old', 'operation': 'rename', 'payload': {'title': 'x'},
        'expected_revision': resumed['result']['revision']}})
    assert renamed.get('error', {}).get('message') == 'session_superseded', renamed


@pytest.mark.asyncio
async def test_an_unopened_old_messaging_conversation_deletes_on_the_first_attempt(tmp_path):
    """Desktop reads the revision from the stored row without mounting it; the delete itself mounts."""
    from gateway.session_controls import AuthorityConnection
    cold, _successor = await _superseded(tmp_path)
    connection = AuthorityConnection(cold.authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read', 'session:control']})
    row = cold.authority.db.get_session('old')
    assert 'old' not in cold.authority.sessions

    deleted = await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': 'old', 'request_id': 'delete-cold', 'operation': 'delete', 'payload': {},
        'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation']}})

    assert 'result' in deleted, deleted
    assert cold.authority.db.get_session('old') is None


@pytest.mark.asyncio
async def test_a_compressed_messaging_conversation_deletes_by_its_listed_segment(tmp_path):
    """Listings name a compressed conversation by its current segment; the delete authorizes on the
    lineage root, where the lane's session lives, and removes the whole conversation."""
    from gateway.session_controls import AuthorityConnection
    _compressed_lane(tmp_path)
    cold = await _cold(tmp_path)
    connection = AuthorityConnection(cold.authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read', 'session:control']})
    row = cold.authority.db.get_session('lane-tip')

    deleted = await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': 'lane-tip', 'request_id': 'delete-tip', 'operation': 'delete', 'payload': {},
        'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation']}})

    assert 'result' in deleted, deleted
    assert cold.authority.db.get_session('lane-tip') is None
