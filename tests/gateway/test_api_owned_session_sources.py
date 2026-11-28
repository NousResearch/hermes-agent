"""The API door runs the first-party rows POST /api/sessions stamps, and no others.

Create accepts a wide label set (cli, telegram, desktop, ...). Bind and admit stay
narrow: ``hermes_browser`` and ``dashboard`` are this door's own clients, and a
foreign surface is ``permission_denied`` before a turn is queued.
"""
import pytest

from gateway.config import Platform
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session

_OWNED = ('hermes_browser', 'dashboard')
_FOREIGN = ('telegram', 'desktop', 'cli')


@pytest.mark.parametrize('source', _OWNED)
def test_bind_adopts_first_party_create_source(owner, source):
    from gateway.session_api import bind_api_session, restore_api_session

    sid = f'{source}-chat'
    seed_session(owner.persistence, sid, source)
    ref = bind_api_session(owner, sid)
    assert session_row(owner.persistence, sid)['source'] == source
    assert owner.sessions[ref.session_id].source.platform == Platform.API_SERVER
    owner.sessions.clear()
    assert restore_api_session(owner, sid) == ref
    assert bind_api_session(owner, sid) == ref
    assert session_row(owner.persistence, sid)['source'] == source


@pytest.mark.parametrize('source', _FOREIGN)
def test_bind_refuses_foreign_surface(owner, source):
    from gateway.session_api import bind_api_session

    sid = f'{source}-chat'
    seed_session(owner.persistence, sid, source)
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        bind_api_session(owner, sid)
    assert sid not in owner.sessions
    assert session_row(owner.persistence, sid)['source'] == source


@pytest.mark.parametrize('source', _OWNED)
def test_admit_runs_first_party_create_source(api, owner, source):
    from gateway.session_api_turn import admit_api_turn

    sid = f'{source}-chat'
    seed_session(owner.persistence, sid, source)
    _, ref, row = admit_api_turn(api, session_id=sid, user_message='hello', conversation_history=[])
    assert ref.session_id == sid and row['status'] == 'queued'
    assert session_row(owner.persistence, sid)['source'] == source
    _, again, second = admit_api_turn(api, session_id=sid, user_message='next', conversation_history=[])
    assert again == ref and second['status'] == 'queued'
    assert second['admission_id'] != row['admission_id']
    assert session_row(owner.persistence, sid)['source'] == source


@pytest.mark.parametrize('source', _FOREIGN)
def test_admit_refuses_foreign_surface(api, owner, source):
    from gateway.session_api_turn import admit_api_turn

    sid = f'{source}-chat'
    seed_session(owner.persistence, sid, source)
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        admit_api_turn(api, session_id=sid, user_message='hello', conversation_history=[])
    assert sid not in owner.sessions
    assert session_row(owner.persistence, sid)['source'] == source
