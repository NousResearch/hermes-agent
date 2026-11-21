"""A first session.submit on a socket the session was never subscribed on still answers.

``session.create`` may ride one websocket while the client's first prompt leaves on a sibling
connection of the same subject: the server-side resume subscribed only the creating transport, so
the submitter's own connection was never attached and its submit answered a bare
``permission_denied`` — logged only at DEBUG, invisible to the user. ``AuthorityConnection.submit``
now auto-attaches an actor entitled to read the session (the exact grant ``session.resume`` gives
any viewer) before anything else. The refusal an unentitled actor gets must not change.
"""
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, get_session_admission


@pytest.fixture
def authority(tmp_path, monkeypatch):
    """A live authority with one mounted telegram session and a stand-in turn runner.

    Mirrors tests/gateway/test_admission_results.py: the submit path under test is real
    (authorize, attach, admit, the durable FIFO); only the agent turn is a stand-in.
    """
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='telegram', user_id='human')
    epoch = begin_runtime_epoch(db, instance_id='owner')

    async def handle(event):
        return 'ok'

    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(
        SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='human'), 'route')
    yield authority
    db.close()


@pytest.mark.asyncio
async def test_a_first_submit_on_a_sibling_socket_auto_attaches_and_admits(authority):
    """The prompt that lands on a never-subscribed connection is admitted, and the connection ends
    up subscribed exactly as if it had resumed first."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import SessionRef

    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    assert connection.subscriptions == {}  # session.create rode another socket

    result = await connection.submit(SessionRef('p', 's'), {'text': 'hello', 'submission_id': 'r1'})
    assert result['admission_id']
    assert 's' in connection.subscriptions
    row = get_session_admission(authority.db, admission_id=result['admission_id'])
    assert row['status'] in ('queued', 'started')


@pytest.mark.asyncio
async def test_an_actor_without_session_read_is_still_refused(authority):
    """Auto-attach grants session:read's floor, nothing more: an actor the session's read fence
    refuses still gets permission_denied, never an admission."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import SessionRef

    connection = AuthorityConnection(authority, object(),
                                     {'user_id': 'human', 'capabilities': ['session:submit']})
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await connection.submit(SessionRef('p', 's'), {'text': 'hello', 'submission_id': 'r2'})
    assert 's' not in connection.subscriptions
