"""A messaging lane's reset parent is not a live session (#143767).

``/new`` on a messaging chat ends the old row ``session_reset`` and inserts the continuation on the
same routing key. No local policy receipt is written, so the receipt-target read in
``live_listing_facts`` stays empty and the ended parent keeps reading as a live session: the desktop
offers it in the sidebar, and every model switch or resume on it is refused (``session_superseded``).
A compressed root whose receipt is absent hides the same way.

Pinned here: a row a reset or compression child carried on is closed for the live feeds, while the
continuation itself and a row nobody continued stay live.
"""
import pytest

from tests.gateway.test_authority_cold_route_recovery import _compressed_lane, _cold, _source, _superseded
from tests.fixtures.storage_seed import seed_session

ROUTE = 'agent:main:telegram:dm:4242'


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))


def _live(authority, *session_ids):
    """Register the rows as live sessions of the authority (what a restore or a first touch does)."""
    from gateway.session_authority import LiveSession
    for sid in session_ids:
        authority.sessions[sid] = LiveSession(_source(), sid)


def _connection(authority):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'fixture', 'instance_id': 'cold',
        'capabilities': ['session:read']})


async def _feeds(connection, request_id):
    listed = await connection.dispatch({'id': request_id, 'method': 'session.list', 'params': {}})
    active = await connection.dispatch({'id': request_id + 1, 'method': 'session.active_list', 'params': {}})
    assert 'result' in listed, listed
    assert 'result' in active, active
    return ({row['id'] for row in listed['result']['sessions']},
            {row.get('id') or row.get('session_key') for row in active['result']['sessions']})


def _messaging_new(store, child, *, marker):
    """What a messaging ``/new`` leaves: ``old`` ended ``session_reset``, ``child`` on its key.

    ``marker`` False is the pre-marker shape (lineage column only, judged by the shared routing key)."""
    seed_session(store, 'old', 'telegram', session_key=ROUTE, chat_id='4242')
    seed_session(store, child, 'telegram', session_key=ROUTE, chat_id='4242',
                 parent_session_id='old', model_config={'_reset_from': 'old'} if marker else None)
    store.sessions.end('old', 'session_reset')


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', [True, False])
async def test_a_messaging_new_reset_parent_is_not_offered_live(tmp_path, marker):
    cold = await _cold(tmp_path)
    _messaging_new(cold.store.persistence, 'continued', marker=marker)
    _live(cold.authority, 'old', 'continued')
    listed, active = await _feeds(_connection(cold.authority), 1)

    assert 'continued' in listed, 'the continuation is the lane\'s live conversation'
    assert 'continued' in active, active
    assert 'old' not in listed, 'a reset parent stays out of the live feed'
    assert 'old' not in active, active


@pytest.mark.asyncio
async def test_a_compression_root_without_a_receipt_target_is_not_offered_live(tmp_path):
    """``_compressed_lane`` leaves the root ended ``compression`` with no receipt of its own."""
    _compressed_lane(tmp_path)
    cold = await _cold(tmp_path)
    _live(cold.authority, 'lane-root', 'lane-tip')
    listed, active = await _feeds(_connection(cold.authority), 1)

    assert 'lane-tip' in listed, listed
    assert 'lane-root' not in listed, 'a compressed root stays out of the live feed'
    assert 'lane-root' not in active, active


@pytest.mark.asyncio
async def test_a_session_nobody_continued_stays_live(tmp_path):
    """The rule is the successor, not the end reason: a row no continuation followed keeps its place."""
    cold = await _cold(tmp_path)
    seed_session(cold.store.persistence, 'lonely', 'telegram', session_key='agent:main:telegram:dm:9')
    cold.store.persistence.sessions.end('lonely', 'session_reset')
    _live(cold.authority, 'lonely')
    listed, active = await _feeds(_connection(cold.authority), 1)
    assert 'lonely' in listed, listed
    assert 'lonely' in active, active


@pytest.mark.asyncio
async def test_a_viewed_reset_parent_is_not_hidden_from_its_viewer(tmp_path):
    """The closed verdict needs no viewer: a mounted parent keeps its live row until the viewer leaves."""
    cold = await _cold(tmp_path)
    _messaging_new(cold.store.persistence, 'continued', marker=True)
    _live(cold.authority, 'old', 'continued')
    cold.authority.sessions['old'].subscribers['mounted'] = object()
    listed, active = await _feeds(_connection(cold.authority), 1)
    assert 'old' in listed, listed
    assert 'old' in active, active
