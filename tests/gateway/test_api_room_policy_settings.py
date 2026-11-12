"""A saved room execution policy is honoured only alongside the room dispatch it was issued for.

These run on the canonical admission path (``admit_api_turn`` / ``check_api_turn`` /
``api_policy_scope``) that ``/v1/runs`` uses, including a turn queued by an older build and
recovered from ``state.db``.
"""
import hashlib
import json

import pytest

from gateway.hosted_room_execution_policy import current_room_execution_policy
from gateway.hosted_room_peer import catalog_mapping
from gateway.session_api_turn import admit_api_turn, api_execution, api_policy_scope, check_api_turn
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_hosted_room_execution_policy import _dispatch, _policy


def test_admission_with_a_policy_but_no_dispatch_is_refused(api, owner):
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        admit_api_turn(api, session_id='policy-only', user_message='hello', conversation_history=[],
                       room_execution_policy=_policy())
    assert list_session_admissions(owner.db, session_id='policy-only') == []


def test_queued_policy_only_admission_from_an_older_build_is_not_recovered(api, owner, monkeypatch):
    from gateway.session_api_turn import recover_api_turns
    _, ref, row = admit_api_turn(api, session_id='legacy-policy', request_id='queued-turn',
                                 user_message='hello', conversation_history=[])
    stored = json.loads(json.dumps(row['payload']))
    stored['api_turn_v1']['settings']['room_execution_policy'] = _policy()
    owner.db._execute_write(lambda conn: conn.execute(
        'UPDATE session_admissions SET payload_json=? WHERE admission_id=?', (json.dumps(stored), row['admission_id'])))
    owner.sessions.clear()
    scheduled = []
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    recover_api_turns(api)
    assert scheduled == []
    # The drain runs the same check before it claims the row.
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        check_api_turn(owner, ref, stored)


def test_dispatch_with_its_catalog_policy_is_admitted_and_bound(api, owner, monkeypatch):
    from gateway import hosted_rooms
    from gateway.platforms import api_server_room_grants
    policy = _policy()
    dispatch = _dispatch(policy).as_mapping()
    catalog = catalog_mapping(installation_id='install-peer', persistent_process=True,
                              target_profile='reviewer', execution_policy=policy)
    monkeypatch.setattr(hosted_rooms, 'local_authority_gateway_id', lambda: 'install-peer')
    monkeypatch.setattr(api_server_room_grants, '_local_room_catalog', lambda *_: (policy, catalog))
    seed = '\0'.join(dispatch[k] for k in ('home_install_id', 'room_id', 'member_id', 'target_profile'))
    sid = 'room_' + hashlib.sha256(seed.encode()).hexdigest()[:32]
    _, ref, row = admit_api_turn(api, session_id=sid, user_message=dispatch['prompt'], conversation_history=[],
                                 room_dispatch=dispatch, room_execution_policy=policy)
    assert row['status'] == 'queued'
    check_api_turn(owner, ref, row['payload'])
    token = api_execution.set({'settings': row['payload']['api_turn_v1']['settings']})
    try:
        with api_policy_scope():
            bound = current_room_execution_policy()
            assert bound is not None and bound.as_mapping() == policy
    finally:
        api_execution.reset(token)
    assert current_room_execution_policy() is None
