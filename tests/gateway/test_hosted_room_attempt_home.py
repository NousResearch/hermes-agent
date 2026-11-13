"""The home that admitted a turn answers for its Stop after the profile is moved to another home:
the new home's "no such session" never settles it. The old home's daemon confirming the Stop
settles it, and so does that daemon being gone, so a moved profile does not hold Disband forever."""
import asyncio
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms, session_hosted_transport as transport
from gateway.session_group_controls import _disband
from hermes_agent.hermes_state_runtime import list_session_admissions
from tests.gateway.test_session_hosted_transport import _SourceTask, _server


@pytest.mark.parametrize('old_home', ['answers', 'silent_then_gone'])
def test_a_moved_profile_settles_once_the_home_that_admitted_the_turn_is_gone(hosted_owner, tmp_path, monkeypatch, old_home):
    authority, loop, _, _ = hosted_owner
    home_a = tmp_path / 'profiles' / 'other'
    home_a.mkdir(parents=True, mode=0o700)
    home_b = tmp_path / 'elsewhere' / 'other'
    home_b.mkdir(parents=True, mode=0o700)
    authority.profile_id = str(home_a)
    source = _SourceTask(tmp_path, monkeypatch, b'attachment')
    service, runtime, db = source.service, source.service.runtime, source.db.db_path
    runtime.prepare_room = runtime.publish_terminal = None
    servers = [_server(tmp_path), _server(home_a)]
    transport.install_hosted_transport(servers[0], source.authority, loop, attest=service.attest)
    transport.install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for server in servers:
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    stopped = []
    try:
        binding = service.bindings()[0]
        rpc = service._resolve_member_transport(binding, source.task)
        sid = rpc.create(profile='other', source='bot_room', title='Group: room')['session_id']
        rpc.submit(profile='other', source='bot_room', session_id=sid, prompt='frozen',
                   task=source.identity, execution_generation=1, attachments=source.bound, on_terminal=None)
        tasks.begin_task_cancel(db, source.identity, cancel_id='stop', expected_cancel_generation=0, clock=time.time)
        with monkeypatch.context() as unanswered:
            def no_answer(*_, **__):
                raise TimeoutError('that gateway did not answer')
            unanswered.setattr(transport, 'owner_request', no_answer)
            now = time.time() + runtime.indeterminate_defer_seconds + 1
            runtime.clock = lambda: now
            runtime._retry_stopping_tasks(binding, runtime._ensure_lease(binding))
        assert tasks.get_task(db, source.identity)['status'] == 'deferred'
        source.homes['other'] = str(home_b)  # moved; home_b's daemon would answer "no such session"
        asked = []
        real = transport.owner_request
        def request(home, verb, params, **kw):
            if verb == 'hosted-producer':
                asked.append(str(home))
            return real(home, verb, params, **kw)
        monkeypatch.setattr(transport, 'owner_request', request)
        if old_home == 'answers':
            result = _disband(service, service, db, source.gateway, {'room_id': 'room'})
            assert 'tombstone' in result, result
            assert asked and set(asked) == {str(home_a)}, asked  # its Stop went where it was admitted
            row, = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert row['status'] == 'terminal'
            return
        def silent(home, verb, params, **kw):
            if verb == 'hosted-producer' and str(home) == str(home_a):
                asked.append(str(home))
                raise TimeoutError('the old home is alive but does not answer')
            return request(home, verb, params, **kw)
        monkeypatch.setattr(transport, 'owner_request', silent)
        for _ in range(2):
            with pytest.raises(hosted_rooms.RoomStopPendingError):
                _disband(service, service, db, source.gateway, {'room_id': 'room'})
        assert asked and set(asked) == {str(home_a)}, asked
        assert tasks.get_task(db, source.identity)['status'] == 'deferred'
        asyncio.run_coroutine_threadsafe(servers[1].stop(), loop).result(10)
        stopped.append(servers[1])
        result = _disband(service, service, db, source.gateway, {'room_id': 'room'})
        assert 'tombstone' in result, result
        assert hosted_rooms.room_state(db, room_id='room', include_disbanded=True)['disbanded_at'] is not None
    finally:
        for server in servers:
            if server not in stopped:
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        source.db.close()
