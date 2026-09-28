"""Hosted source binding preserves the canonical hidden room identity."""
import hashlib

import pytest

from tests.gateway.test_api_source_binding import owner  # noqa: F401
from tests.gateway.test_hosted_room_execution_policy import _dispatch, _policy
from hermes_state_runtime import RuntimeStoreError


def test_hosted_binding_requires_exact_server_dispatch(owner):
    from gateway.session_api import bind_api_session, restore_api_session
    dispatch = _dispatch(_policy()).as_mapping()
    seed = '\0'.join(dispatch[k] for k in ('home_install_id', 'room_id', 'member_id', 'target_profile'))
    sid = 'room_' + hashlib.sha256(seed.encode()).hexdigest()[:32]
    ref = bind_api_session(owner, sid, hosted_dispatch=dispatch)
    row = owner.db.get_session(sid)
    assert row['source'] == 'bot_room' and row['hidden'] == 1 and row['title'] == 'Group: room-1'
    owner.sessions.clear()
    assert restore_api_session(owner, sid) == ref
    owner.sessions.clear()
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        bind_api_session(owner, sid)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_api_session(owner, 'different-room-id', hosted_dispatch=dispatch)
    assert owner.db.get_session('different-room-id') is None
