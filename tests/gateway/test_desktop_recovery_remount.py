"""A cold Desktop row remounts live after a restart, and its first turn is claimed as a viewer's.

A Desktop-client row (``source='desktop'``, NULL session_key/chat_id) matched no resume family:
``restore_api_session`` answers bare ``not_found`` because the row carries no API binding, so the
conversation was unopenable after every gateway restart. ``authorize`` now falls through to the
desktop recovery family, which rebuilds the row as an API-family source. Its admissions are plain
viewer text carrying no ``api_turn_v1`` — so the api turn gate must key on the family marker, not
``Platform.API_SERVER`` alone, or the remounted row's first turn sits admitted but never claimed.
"""
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def authority(tmp_path, monkeypatch):
    """A live authority over a temp SessionDB holding a cold desktop row."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(tmp_path / 'state.db')
    db.create_session(SID, source='desktop', user_id='basic:admin', profile_name='default')
    epoch = begin_runtime_epoch(db, instance_id='owner')

    class _Store:
        def __init__(self):
            import threading
            self._lock = threading.RLock()
            self._entries = {}

        def _ensure_loaded_locked(self):
            pass

        def _generate_session_key(self, source):
            from gateway.session import build_session_key
            return build_session_key(source)

    runner = SimpleNamespace(_draining=False, session_store=_Store())
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db, epoch=epoch)
    adapter = SimpleNamespace(gateway_runner=runner)
    runner._adapter_for_source = lambda source: adapter
    yield authority
    db.close()


SID = '20261030_090000_deskrow'


@pytest.mark.asyncio
async def test_a_cold_desktop_row_mounts_through_authorize(authority):
    """resume of a desktop row mounts it live as an API-family source owned by the row's own
    stamp, and registers the routing entry under the row's api route."""
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef

    actor = SimpleNamespace(profile_id=authority.profile_id,
                            capabilities=frozenset({'session:read'}))
    SessionAuthority.authorize(authority, actor, SessionRef('p', SID), 'session:read')

    live = authority.sessions[SID]
    assert isinstance(live, LiveSession)
    assert live.source.platform == Platform.API_SERVER
    assert live.source.chat_id == SID
    assert live.source.user_id == 'basic:admin'
    route = 'agent:main:api_server:dm:' + SID
    entry = authority.runner.session_store._entries[route]
    assert entry.session_id == SID


@pytest.mark.asyncio
async def test_the_api_turn_gate_keys_on_the_family_marker(authority, monkeypatch):
    """On the remounted row a markerless payload never re-enters restore_api_session (the FIFO head
    stays claimable); a real api_turn_v1 payload still does."""
    import gateway.session_api_turn as api_turn
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError

    actor = SimpleNamespace(profile_id=authority.profile_id,
                            capabilities=frozenset({'session:read'}))
    from gateway.session_authority import SessionAuthority
    SessionAuthority.authorize(authority, actor, SessionRef('p', SID), 'session:read')
    ref = SessionRef('p', SID)

    restores = []

    def spy(authority_, sid):
        restores.append(sid)
        raise RuntimeStoreError('not_found')

    monkeypatch.setattr(api_turn, 'restore_api_session', spy)

    # a remounted desktop row's claimed head: no marker -> no api restore, adapter answers
    back = api_turn.check_api_turn(authority, ref, {'text': 'hello'})
    assert restores == []
    assert back is authority.runner._adapter_for_source(authority.sessions[SID].source)

    # a real API admission carries the marker and keeps its restore path
    payload = {'text': 'hi', 'api_turn_v1': {'history': None, 'settings': {}}}
    with pytest.raises(RuntimeStoreError, match='not_found'):
        api_turn.check_api_turn(authority, ref, payload)
    assert restores == [SID]
