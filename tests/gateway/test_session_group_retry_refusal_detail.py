"""``groups.retry`` refuses an unpinned attempt with the keys it needs named (#152324).

The gateway has always required the exact attempt ``groups.state`` publishes — the fence that
stops a stale Retry from re-running a newer one — but its refusal was a bare ``invalid_params``,
while the legacy wire model (``GroupsRetryParams``) called the pair optional. A client following
that model could not tell a contract it was too old for from a malformed request, so the stalled
room's only affordance dead-ended. Both now agree: the pair is required, and the refusal names the
missing key.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB


class RoomService:
    """The retry/discard surface ``_execution_control`` drives, recording the exact kwargs."""

    def __init__(self, path):
        self.db_path = path
        self.calls = []
        self.runtime = SimpleNamespace(status=lambda: {'running': True, 'stopping': False})

    def authorize_room(self, actor_subject, room_id, *, create=False, conn=None):
        assert room_id == 'owned', room_id

    def retry_room_task(self, room_id, *, task_id, member_id=None, execution_generation=None):
        self.calls.append(('retry', room_id, task_id, member_id, execution_generation))
        return {'identity': SimpleNamespace(room_id=room_id, task_id=task_id,
                                           thread_id='thread', turn_id='turn'),
                'status': 'queued', 'execution_generation': 3, 'cancel_generation': 2}

    def discard_room_task(self, **params):
        self.calls.append(('discard', params['room_id'], params['task_id'],
                           params.get('member_id'), params.get('execution_generation')))
        return self.retry_room_task(**params)


def _connection(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({}))
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(profile_id=str(tmp_path), instance_id='owner', db=db, events={})
    service = authority.hosted_room_service = RoomService(db.db_path)
    return AuthorityConnection(authority, object(), {'user_id': 'alice'}), service, db


def _dispatch(connection, method, **params):
    async def call():
        return await connection.dispatch({'id': 1, 'method': method, 'params': params})
    return asyncio.run(call())


@pytest.mark.parametrize('method', ['groups.retry', 'groups.discard'])
def test_an_unpinned_attempt_names_the_keys_the_row_publishes(method, tmp_path, monkeypatch):
    """The Retry row carries all three keys; a request without the pair says which it lacked."""
    connection, service, db = _connection(tmp_path, monkeypatch)
    with db:
        result = _dispatch(connection, method, room_id='owned', task_id='prepare:owned')
        assert result['error']['message'] == 'invalid_params', result
        data = result['error']['data']
        assert data['field'] == 'member_id', data
        assert 'missing: member_id, execution_generation' in data['detail'], data
        assert service.calls == []


def test_a_malformed_coordinate_names_its_own_field(tmp_path, monkeypatch):
    connection, service, db = _connection(tmp_path, monkeypatch)
    with db:
        for params, field in (
                ({'member_id': 'one', 'execution_generation': 0}, 'execution_generation'),
                ({'member_id': 'one', 'execution_generation': '2'}, 'execution_generation'),
                ({'member_id': '', 'execution_generation': 1}, 'member_id')):
            result = _dispatch(connection, 'groups.retry', room_id='owned',
                               task_id='task-exact', **params)
            assert result['error']['message'] == 'invalid_params', (params, result)
            assert result['error']['data']['field'] == field, (params, result)
        # A value of the wrong TYPE never reaches the handler: the declared contract refuses it
        # with the field path, which is the other half of the same diagnosis.
        result = _dispatch(connection, 'groups.retry', room_id='owned', task_id='task-exact',
                           member_id=7, execution_generation=1)
        assert result['error']['data'] == {'reason': 'invalid_params', 'fields': ['member_id']}, result
        assert service.calls == []


def test_the_published_attempt_still_retries_exactly(tmp_path, monkeypatch):
    connection, service, db = _connection(tmp_path, monkeypatch)
    with db:
        result = _dispatch(connection, 'groups.retry', room_id='owned',
                           task_id='task-exact', member_id='one', execution_generation=2)
        assert result['result']['retried'] is True, result
        assert service.calls == [('retry', 'owned', 'task-exact', 'one', 2)]


def test_both_wire_models_require_the_same_attempt_keys():
    """The legacy and canonical declarations of one verb may not disagree (#152324)."""
    from tui_gateway.contracts.canonical_projections import CanonicalRoomAttemptParams
    from tui_gateway.contracts.groups_bot_relay import GroupsRetryParams

    attempt = {'room_id': 'room', 'member_id': 'one', 'task_id': 'task',
               'execution_generation': 1}

    def required(model):
        return set(model.model_json_schema().get('required') or ()) - {'profile'}

    assert required(GroupsRetryParams) == required(CanonicalRoomAttemptParams) == set(attempt)
    assert GroupsRetryParams.model_validate(attempt).room_id == 'room'
    for key in ('member_id', 'execution_generation'):
        without = {name: value for name, value in attempt.items() if name != key}
        with pytest.raises(ValueError):
            GroupsRetryParams.model_validate(without)
