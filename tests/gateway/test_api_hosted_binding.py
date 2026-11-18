"""Hosted source binding preserves the canonical hidden room identity."""
import hashlib

import pytest

from tests.gateway.test_hosted_room_execution_policy import _dispatch, _policy
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_read import session_row


def test_hosted_binding_requires_exact_server_dispatch(owner):
    from gateway.session_api import bind_api_session, restore_api_session
    dispatch = _dispatch(_policy()).as_mapping()
    seed = '\0'.join(dispatch[k] for k in ('home_install_id', 'room_id', 'member_id', 'target_profile'))
    sid = 'room_' + hashlib.sha256(seed.encode()).hexdigest()[:32]
    ref = bind_api_session(owner, sid, hosted_dispatch=dispatch)
    row = session_row(owner.persistence, sid)
    assert row['source'] == 'bot_room' and row['hidden'] == 1 and row['title'] == 'Group: room-1'
    owner.sessions.clear()
    assert restore_api_session(owner, sid) == ref
    owner.sessions.clear()
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        bind_api_session(owner, sid)
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        bind_api_session(owner, 'different-room-id', hosted_dispatch=dispatch)
    assert session_row(owner.persistence, 'different-room-id') is None
