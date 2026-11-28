"""Deleted local receipts are terminal refusals, not retryable storage failures (#155582)."""
import pytest

from tests.gateway.test_authority_cold_route_recovery import _Runner

pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")  # symbolic profile ids are relative homes


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_path: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})


@pytest.mark.asyncio
@pytest.mark.parametrize('reader', ['owned_receipt', 'restore_local_session'])
@pytest.mark.parametrize('deleted', [True, False], ids=['retired', 'missing-live-receipt'])
async def test_local_receipt_refusal_distinguishes_retirement(tmp_path, reader, deleted):
    from gateway import session_local_recovery
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from persistence.core.errors import AuthorityRefusal
    from persistence.sqlite.conversation._store.local import POLICY_PREFIX
    from tests.fixtures.storage_holder import raw_holder

    runner = _Runner(tmp_path)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(authority, owner, {'request_id': 'retired-receipt', 'source': 'gui',
                                                  'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    sid = ref.session_id
    if deleted:
        # A deletion removes the private creation receipt along with the transcript and retires the id.
        assert authority.persistence.sessions.delete(sid, media_root=str(tmp_path / 'media'))
        assert authority.persistence.runtime.retired_session(sid)
    else:
        # A live session the receipt vanished from: an older build's partial write, nothing retired.
        raw_holder(authority.persistence)._execute_write(
            lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)))
        assert not authority.persistence.runtime.retired_session(sid)
    with pytest.raises(AuthorityRefusal) as refused:
        getattr(session_local_recovery, reader)(authority, sid)
    assert refused.value.reason == ('not_found' if deleted else 'storage_unavailable')
