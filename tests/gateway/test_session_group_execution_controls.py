"""Native room controls retain the service protocol and authenticated ownership."""
import asyncio
from types import SimpleNamespace

from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError


class RoomService:
    def __init__(self, path, owner):
        self.db_path = path
        self.owner = owner
        self.calls = []
        self.runtime = SimpleNamespace(status=lambda: {'running': True, 'stopping': False})

    def authorize_room(self, actor_subject, room_id, *, create=False):
        if actor_subject != self.owner or room_id != 'owned':
            raise RuntimeStoreError('permission_denied')

    def create_room(self, *, room_id, name, members):
        from gateway.hosted_rooms import create_room, local_authority_gateway_id
        self.calls.append(('create', room_id))
        return create_room(self.db_path, room_id=room_id, name=name, members=members,
                           authority_gateway_id=local_authority_gateway_id())

    def status(self, room_id):
        return {'room_id': room_id, 'pending_actions': [{'kind': 'retry', 'task_id': 'task'}]}

    def revoke_room_routes(self, room_id):
        self.calls.append(('revoke', room_id))

    def send(self, *, room_id, event_id, payload):
        self.calls.append(('send', room_id, event_id, payload))
        return {'event_id': event_id, 'payload': payload}

    def stop_room(self, room_id, *, cancel_id, require_acknowledged=False):
        self.calls.append(('stop', room_id, cancel_id, require_acknowledged))
        return 1

    def retry_room_task(self, room_id, *, task_id, member_id, execution_generation):
        self.calls.append(('retry', room_id, task_id, member_id, execution_generation))
        return {'identity': SimpleNamespace(room_id=room_id, task_id=task_id,
                                           thread_id='thread', turn_id='turn'),
                'status': 'queued', 'execution_generation': 3, 'cancel_generation': 2}

    def discard_room_task(self, **params):
        return self.retry_room_task(**params)

    def approve_room_task(self, room_id, *, member_id, task_id, execution_generation,
                          choice, request_id=None):
        self.calls.append(('approve', room_id, member_id, task_id, execution_generation,
                           choice, request_id))
        return {'resolved': request_id}


def test_execution_controls_preserve_native_wire_and_exact_task_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), instance_id='owner', db=db, events={})
        connection = AuthorityConnection(authority, object(), {'user_id': 'alice'})
        service = authority.hosted_room_service = RoomService(db.db_path, connection.actor.subject)

        async def probe():
            async def call(method, **params):
                return await connection.dispatch({'id': 1, 'method': method, 'params': params})
            caps = await call('groups.capabilities')
            assert caps['result']['driver'] is True
            created = await call('groups.create', room_id='owned', name='Room', members=[
                {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                {'member_id': 'two', 'profile': 'helper', 'handle': 'two'}])
            assert 'result' in created, created
            assert service.calls[-1] == ('create', 'owned')
            state = await call('groups.state', room_id='owned')
            assert state['result']['driver_status'] == service.status('owned')
            other = AuthorityConnection(authority, object(), {'user_id': 'bob'})
            listed = await other.dispatch({'id': 1, 'method': 'groups.list', 'params': {}})
            assert listed['result']['rooms'] == []

            payload = {'text': '@one hello', 'mentions': ['one']}
            sent = await call('groups.send', room_id='owned', event_id='input', payload=payload)
            assert sent['result']['accepted'] and sent['result']['driver_started']
            assert sent['result']['client_event_id'] == 'input'
            from gateway.hosted_rooms import user_event_id
            assert service.calls[-1] == ('send', 'owned', user_event_id('input'), payload)
            stopped = await call('groups.stop', room_id='owned', cancel_id='cancel-exact')
            assert stopped['result'] == {'cancelled': 1}
            retried = await call('groups.retry', room_id='owned', task_id='task-exact', member_id='one', execution_generation=2)
            assert retried['result']['task'] == {
                'room_id': 'owned', 'task_id': 'task-exact', 'thread_id': 'thread',
                'turn_id': 'turn', 'status': 'queued', 'execution_generation': 3,
                'cancel_generation': 2}
            assert service.calls[-1] == ('retry', 'owned', 'task-exact', 'one', 2)
            discarded = await call('groups.discard', room_id='owned', task_id='task-exact', member_id='one', execution_generation=2)
            assert discarded['result']['discarded'] is True
            approved = await call('groups.approve', room_id='owned', member_id='one',
                                  task_id='task-exact', execution_generation=3,
                                  choice='once', request_id='approval-exact')
            assert approved['result'] == {'approved': True, 'result': {'resolved': 'approval-exact'}}
            assert service.calls[-1] == ('approve', 'owned', 'one', 'task-exact', 3,
                                        'once', 'approval-exact')
            disbanded = await call('groups.disband', room_id='owned', cancel_id='disband-exact')
            assert 'result' in disbanded, disbanded
            assert service.calls[-2:] == [('stop', 'owned', 'disband-exact', True), ('revoke', 'owned')]
        asyncio.run(probe())


def test_execution_controls_reject_foreign_actor_profile_room_and_unready_service(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), instance_id='owner', db=db, events={})
        owner = AuthorityConnection(authority, object(), {'user_id': 'alice'})
        service = authority.hosted_room_service = RoomService(db.db_path, owner.actor.subject)

        async def probe():
            requests = {
                'groups.state': {},
                'groups.log': {},
                'groups.rename': {'event_id': 'rename', 'name': 'No'},
                'groups.disband': {},
                'groups.send': {'event_id': 'input', 'payload': {'text': 'hello'}},
                'groups.stop': {'cancel_id': 'cancel'},
                'groups.retry': {'task_id': 'task'},
                'groups.discard': {'task_id': 'task', 'member_id': 'one', 'execution_generation': 1},
                'groups.approve': {'member_id': 'one', 'task_id': 'task',
                                   'execution_generation': 1, 'request_id': 'approval', 'choice': 'once'},
            }
            for method, params in requests.items():
                for identity, extra, reason in [
                    ({'user_id': 'bob'}, {}, 'permission_denied'),
                    ({'user_id': 'alice', 'capabilities': []}, {}, 'permission_denied'),
                    ({'user_id': 'alice', 'profile_id': 'foreign'}, {}, 'profile_mismatch'),
                    ({'user_id': 'alice'}, {'room_id': 'foreign'}, 'permission_denied'),
                    ({'user_id': 'alice'}, {'actor': 'bob'}, 'invalid_params'),
                ]:
                    peer = AuthorityConnection(authority, object(), identity)
                    result = await peer.dispatch({'id': 1, 'method': method,
                                                  'params': {'room_id': 'owned', **params, **extra}})
                    assert result['error']['message'] == reason, result
            assert service.calls == []
            service.runtime = SimpleNamespace(status=lambda: {'running': True, 'stopping': True})
            result = await owner.dispatch({'id': 1, 'method': 'groups.stop', 'params': {'room_id': 'owned'}})
            assert result['error']['message'] == 'runtime_coordination_required'
            caps = await owner.dispatch({'id': 1, 'method': 'groups.capabilities', 'params': {}})
            assert caps['result']['driver'] is False
            foreign = AuthorityConnection(authority, object(), {'user_id': 'bob'})
            result = await foreign.dispatch({'id': 1, 'method': 'groups.state', 'params': {'room_id': 'owned'}})
            assert result['error']['message'] == 'permission_denied'

            assert service.calls == []
        asyncio.run(probe())
