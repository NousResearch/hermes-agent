"""A home the profile now points at can't answer for a turn its old home still holds.

Fails on purpose: it shows the gap. The new home's "no session" answer is stubbed."""
import asyncio
import time

from hermes_agent.gateway import hosted_room_driver as tasks, hosted_rooms, session_hosted_transport as transport
from hermes_agent.gateway.session_group_controls import _disband
from hermes_agent.hermes_state_runtime import list_session_admissions
from tests.gateway.test_session_hosted_transport import _SourceTask, _server


def test_disband_keeps_a_turn_its_old_home_still_holds_after_a_remap(hosted_owner, tmp_path, monkeypatch):
    authority, loop, _, _ = hosted_owner
    home_a = tmp_path / 'profiles' / 'other'
    home_a.mkdir(parents=True, mode=0o700)
    home_b = tmp_path / 'elsewhere' / 'other'
    home_b.mkdir(parents=True, mode=0o700)
    authority.profile_id = str(home_a)
    source = _SourceTask(tmp_path, monkeypatch, b'attachment')
    service, runtime, db = source.service, source.service.runtime, source.db.db_path
    runtime.prepare_room = runtime.publish_terminal = None
    hosted_rooms.append_event(db, room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'frozen'},
        authority_gateway_id=source.gateway, authority_epoch=1)
    servers = [_server(tmp_path), _server(home_a)]
    transport.install_hosted_transport(servers[0], source.authority, loop, attest=service.attest)
    transport.install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for server in servers:
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    try:
        binding = service.bindings()[0]
        rpc = service._resolve_member_transport(binding, source.task)
        sid = rpc.create(profile='other', source='bot_room', title='Group: room')['session_id']
        rpc.submit(profile='other', source='bot_room', session_id=sid, prompt='frozen',
                   task=source.identity, execution_generation=1, attachments=source.bound, on_terminal=None)
        tasks.begin_task_cancel(db, source.identity, cancel_id='stop', expected_cancel_generation=0,
                                clock=time.time)
        with monkeypatch.context() as unanswered:
            def no_answer(*_, **__):
                raise TimeoutError('that gateway did not answer')
            unanswered.setattr(transport, 'owner_request', no_answer)
            now = time.time() + runtime.indeterminate_defer_seconds + 1
            runtime.clock = lambda: now
            runtime._retry_stopping_tasks(binding, runtime._ensure_lease(binding))
        assert tasks.get_task(db, source.identity)['status'] == 'deferred'
        # The profile now points at another home, whose daemon answers but holds no such session.
        source.homes['other'] = str(home_b)
        real = transport.HostedRoomOwnerRPC.resolve_exact
        def resolve(self, **kw):
            return None if str(self.home) == str(home_b) else real(self, **kw)
        monkeypatch.setattr(transport.HostedRoomOwnerRPC, 'resolve_exact', resolve)
        result = _disband(service, service, db, source.gateway, {'room_id': 'room'})
        row, = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert 'tombstone' not in result or row['status'] == 'terminal', (
            f"Disband finished while the old home still holds the turn: {row['status']}")
    finally:
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        source.db.close()
